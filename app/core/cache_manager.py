import os
import json
import logging
from collections import defaultdict
import redis
import jieba
import numpy as np
import hashlib
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from config import settings
from db.vector_client import vector_client

logger = logging.getLogger(__name__)
jieba.setLogLevel(logging.INFO)

class CacheManager:
    def __init__(self):
        logger.info("Connecting to Redis Semantics / Strict QA Cache...")
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
        self.embeddings = vector_client.embeddings
        self.vector_search_algo = settings.REDIS_SEMANTIC_SEARCH_ALGORITHM
        self.vector_dim = settings.EMBEDDING_MODEL_DIMENSION
        self.index_name = "idx:rag_semantic_cache"
        
        # 啟動時自動檢查並建立 Redis 向量索引
        self._init_redis_vector_index()
        

    def _init_redis_vector_index(self):
        """在 Redis 中初始化向量索引"""
        try:
            self.redis_client.ft(self.index_name).info()
            logger.info("Redis Semantic Cache Index already exists.")
        except Exception:
            logger.info("Creating new Redis Vector Index for Semantic Cache...")
            # 先把 Redis索引工具的 schemas寫好
            schema = (
                # 定義欄位（category/question），對應 JSON 中的 $.category/$.question 路徑
                TextField("$.category", as_name="category"),
                TextField("$.question", as_name="question"),
                # 定義「向量欄位」（question_vector）, 對應 JSON 中的 $.question_vector 路徑
                VectorField(
                    "$.question_vector",
                    # FLAT 演算法（暴力搜尋）適合中小規模數據，最準, HNSW適合規模大(記憶體開銷較大)95%準而已
                    self.vector_search_algo,
                    {
                        "TYPE": "FLOAT32",      # 向量數值的資料型態為 32 位元浮點數
                        "DIM": self.vector_dim,    # 設定向量的維度大小
                        "DISTANCE_METRIC": "COSINE"  # 使用「餘弦相似度」來計算兩個問題的語意接近程度
                    },
                    as_name="question_vector"
                )
            )
            # 正式在 Redis 中建立索引工具
            self.redis_client.ft(self.index_name).create_index(
                fields=schema,
                # 指定這個索引工具只會找 Key 開頭為 "rag_cache:" 的 Redis JSON 資料，避免找到其他資料(User, order..)
                definition=IndexDefinition(prefix=["rag_cache:"], index_type=IndexType.JSON)
            )
            logger.info("Redis Vector Index created successfully!")

    def check_semantic_cache(self, question: str, category: str, threshold: float = 0.92) -> dict | None:
        """執行 Redis 語意向量快取比對"""
        try:
          query_vector = self.embeddings.embed_query(question)     # 產生 list格式一維向量
          # 將向量轉成 NumPy 陣列（32位元浮點數），再轉成二進位（Bytes），來符合上方 Redis 向量搜尋規定的格式
          query_vector_bytes = np.array(query_vector, dtype=np.float32).tobytes()
          
          # 限定搜尋特定 category的向量庫就好
          base_query = f"@category:{category}"
          search_query = (
              # Redis Vector Search 語法: 從 question_vector 中，找出與傳入的 $vec_param 最相似的，並把計算出的距離分數命名為 vector_score
              Query(f"({base_query})=>[KNN 1 @question_vector $vec_param AS vector_score]")
              .sort_by("vector_score")  # 依據向量距離由近到遠排序
              .paging(0, 1)       # 分頁設定：只要尋回最接近的第一筆（第 0 到第 1 筆）
              .return_fields("$.answer", "vector_score", "$.question")  # 指定 search後的回傳欄位
              .dialect(2)         # 使用 Redis Search Dialect 2 語法版本（支援向量搜尋）
          )

          res = self.redis_client.ft(self.index_name).search(
              search_query, 
              query_params={"vec_param": query_vector_bytes}
          )                    # res是Result類別物件:{res.total: int, res.doc: List}
          
          if res.docs:
            doc = res.docs[0]   # 從res中取出搜尋結果
            cosine_distance = float(doc.vector_score)
            similarity = 1 - cosine_distance
            
            if similarity >= threshold:
                # 撈出快取中原本的舊問題文字（如果撈不到就顯示 Unknown）
                cached_question = getattr(doc, "$.question", "Unknown")
                logger.info(f"⚡ [Cache Match] Semantic Cache Hit! (Sim: {similarity:.2f})")
                logger.info(f"Matched Question: '{cached_question}'")
                
                # 動態過期機制：若這筆快取被使用，每次命中就重置 24 小時（86400秒）的存活時間（TTL）
                self.redis_client.expire(doc.id, 86400)
                # 回傳最相近的Qustion的快取 answer(包含answer、sources 與 cached, 符合QueryResponse的schemas)
                return json.loads(getattr(doc, "$.answer"))

        except Exception as e:
            logger.error(f"Error during Redis semantic cache search: {str(e)}")
        return None

    def set_semantic_cache(self, question: str, category: str, answer_data: dict):
        """LLM 生成new答案後，把問題與答案向量同步寫入 Redis JSON Cache 中"""
        try:
            # 建立 Redis Key。格式為 rag_cache:{分類}:{問題的雜湊值}, 使用MD5雜湊演算法而非使用hash()導致伺服器重啟後改變雜湊值
            question_hash = hashlib.md5(question.encode('utf-8')).hexdigest()
            cache_key = f"rag_cache:{category}:{question_hash}"
            query_vector = self.embeddings.embed_query(question)  # 產生 list格式一維向量如 [0.01234, -0.05678, 0.98765, ...]
            
            # 打包新產生的回答 
            cache_data = {
                "category": category,
                "question": question,
                "answer": json.dumps(answer_data, ensure_ascii=False),
                "question_vector": query_vector         # 儲存原始list格式的向量, RedisJSON 會變成 JSON 陣列
            }
            # 正式將新產生的資料以 Redis JSON 格式寫入資料庫，"$" 代表寫入 JSON 的根節點
            self.redis_client.json().set(cache_key, "$", cache_data)
            # 設定這筆快取只存活 24 小時（86400 秒）
            self.redis_client.expire(cache_key, 86400)
            logger.info("💾 [Cache Store] Semantic Cache persisted successfully (TTL: 24h).")
        except Exception as e:
            logger.error(f"Failed to write semantic cache: {str(e)}")

# 全域單例
cache_manager = CacheManager()
