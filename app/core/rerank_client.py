import logging
from config import settings
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankClient:
    def __init__(self):
        logger.info(f"Loading CrossEncoder Reranker model [{settings.RERANKER_MODEL_NAME}] on [{settings.RERANKER_DEVICE}]...")
        # 💡 關鍵優化：Colab 讀取 "cuda" 飛速重排，地端 Docker 讀取 "cpu" 釋放顯存
        self.model = CrossEncoder(
            model_name=settings.RERANKER_MODEL_NAME,
            device=settings.RERANKER_DEVICE
        )

    def rerank(self, query: str, documents: list, top_k: int = 3) -> list:
        """
        將 LangChain Document 陣列傳入進行深度交叉評分
        """
        if not documents:
            return []
            
        # 組裝成 CrossEncoder 要求的 [[query, text1], [query, text2], ...] 格式
        pairs = [[query, str(doc.page_content)] for doc in documents]
        
        # Reranker 重新打分
        scores = self.model.predict(pairs)
        
        # 將分數塞回 Document 的 metadata 中 ##這裡的idx要跟上面text1, text2 idx一樣
        for idx, score in enumerate(scores):
            documents[idx].metadata["rerank_score"] = float(score)
            
        # 依分數從高到低排序並切出前 K 個最優上下文
        sorted_docs = sorted(documents, key=lambda x: x.metadata["rerank_score"], reverse=True)
        return sorted_docs[:top_k]

# 全域單例：服務啟動時只會載入一次權重
rerank_client = RerankClient()