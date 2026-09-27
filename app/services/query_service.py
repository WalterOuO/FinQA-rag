import logging
from pathlib import Path
import json
import asyncio  # 💡 用於解決硬碟 I/O 阻塞

# 引入微服務全域變數與模型常駐單例客戶端
from config import settings
from db.vector_client import get_vector_client
from core.rerank_client import rerank_client
from core.llm_client import llm_client
# from core.cache_manager import cache_manager
from services.prompts import FINQA_RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)

class QueryService:
  def _read_parent_json(self, path: Path) -> dict:
    """標準同步讀檔函式，供執行緒池調用"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

  async def answer_question(self, question: str, category: str) -> dict:
    """
    微服務: Advanced RAG 線上即時問答業務調度大腦
    """
    # # ====== Step 1: Redis 語意快取檢查 ======
    # cached_answer = cache_manager.check_semantic_cache(question, category, threshold=settings.REDIS_SEMANTIC_CACHE_THRESHOLD)
    # if cached_answer:
    #   cached_answer["cached"]=True
    #   return cached_answer

    # logger.info(f"🔍 [Step 1] Cache Missed. Starting Dense Retrieval for [{category}]")

    # ====== Step 2: Dense 向量檢索 ======
    search_filter = {"category": category}
    dense_top_docs = get_vector_client().db.similarity_search(
        query=question,
        k=settings.HYBRID_SEARCH_TOP_K,
        filter=search_filter
    )
    
    if not dense_top_docs:
        return {
            "answer": "抱歉，在目前的資料庫中找不到與您問題相關的參考文件。請先上傳參考文件後再行提問。",
            "sources": [],
            "cached": False
        }
    logger.info(f"🧲 [Step 2] Dense Vector Search completed. Retrieved {len(dense_top_docs)} candidate chunks.")

    # ====== Step 3: 子文檔去重 ======
    seen_child_ids = set()
    unique_child_docs = []
    for doc in dense_top_docs:
        c_id = doc.metadata.get("child_id")
        if c_id not in seen_child_ids:
            seen_child_ids.add(c_id)
            unique_child_docs.append(doc)
    logger.info(f"🔀 Merged & De-duplicated child chunks.")

    # ====== Step 4: 實體物理反查父文檔 (Parent Retrieval) ======
    parent_documents = []
    loaded_parent_files = {}  
    seen_parent_ids = set()

    for c_doc in unique_child_docs:   
      p_id = c_doc.metadata.get("parent_id")
      f_name = c_doc.metadata.get("file_name")
      
      if not p_id or p_id in seen_parent_ids:
          continue
      seen_parent_ids.add(p_id)
      
      pdf_stem = Path(f_name).stem
      parent_json_path = settings.PARENT_CHUNKS_DIR / category / f"{pdf_stem}_parents.json"
      
      if parent_json_path.exists():
          # 💡 優化 2：將原本會卡死 Event Loop 的同步讀檔，丟給執行緒池（to_thread）非同步處理！
          if str(parent_json_path) not in loaded_parent_files:
              loaded_parent_files[str(parent_json_path)] = await asyncio.to_thread(
                  self._read_parent_json, parent_json_path
              )
          
          p_data = loaded_parent_files[str(parent_json_path)].get(p_id)
          if p_data:
              from langchain_core.documents import Document
              parent_documents.append(Document(
                  page_content=p_data["page_content"],
                  metadata=p_data["metadata"]
              ))

    if not parent_documents:
        return {
            "answer": "抱歉，資料庫中無法尋獲對應的完整文檔架構。請重新上傳補充文件。",
            "sources": [],
            "cached": False
        }
    logger.info(f"📂 [Step 4] Parent Retrieval success. Loaded {len(parent_documents)} parent documents non-blockingly.")

    # ====== Step 5: Reranker 交叉深度重新評分 (動態讀取 Top K) ======
    reranked_top_docs = rerank_client.rerank(
        query=question,
        documents=parent_documents,
        top_k=settings.RERANK_OUT_TOP_K
    )
    logger.info(f"⚖️ [Step 5] CrossEncoder Reranking finished. Selected top core contexts.")

    # ====== Step 6: 建立問答 Prompt 與 LLM 生成 ======
    context_str = ""
    sources = []
    for idx, doc in enumerate(reranked_top_docs):
      src_file = doc.metadata.get("file_name", "未知文件")
      parent_id = doc.metadata.get("parent_id", "未知 ID")  # 撈出 rerank後文件對應的 parent ID
      
      # 取出層級最高的 Markdown 標題作為人類可讀的章節導航
      section_header = doc.metadata.get("Header1", doc.metadata.get("Header2", doc.metadata.get("Header3", "正文段落")))
      
      # 給 LLM 閱讀的脈絡維持前台序號
      context_str += f"[文件來源 {idx+1}]: {src_file} ({section_header})\n{doc.page_content}\n\n"
      
      # 增加引用資料方便回溯原始文件
      sources.append({
          "frontend_index": idx + 1,  # 供 Streamlit 前端渲染文件序號
          "file_name": src_file,
          "category": category,
          "parent_id": parent_id,     # 允許 LLM生成答案後人工反查 parent chunk JSON 
          "header": section_header
      })

    rag_prompt = FINQA_RAG_PROMPT_TEMPLATE.format(
        context_str=context_str,
        question=question
    )

    llm_answer = llm_client.generate(rag_prompt)
    logger.info(f"🤖 [Step 6] LLM Text Generation completed with full trace sources.")

    final_response = {
        "answer": llm_answer,
        "sources": sources,  
        "cached": False
    }      

    # ====== Step 7: 更新 Redis Semantic Cache ======
    # cache_manager.set_semantic_cache(question, category, final_response)
    
    return final_response

# 全域單例
query_service = QueryService()