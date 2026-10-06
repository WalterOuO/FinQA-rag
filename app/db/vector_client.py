import logging

from config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import FastEmbedSparse
from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

class VectorDBClient:
  def __init__(self):
    logger.info("初始化 Embedding Model 與 Qdrant Client...")

    # Embedding model：Dense Vector
    self.embeddings = HuggingFaceEmbeddings(
      model_name=settings.EMBEDDING_MODEL_NAME,
      model_kwargs={"device": settings.RERANKER_DEVICE}
    )

    # Sparse Vector：BM25
    self.sparse_embeddings = FastEmbedSparse(
      model_name="Qdrant/bm25"
    )

    self._init_qdrant()

  def _init_qdrant(self):
    self.client = QdrantClient(
      path=str(settings.LOCAL_QDRANT_DB_DIR)
    )

    self.db = QdrantVectorStore(
      client=self.client,
      collection_name="pdfqa_collection",
      embedding=self.embeddings,
      sparse_embedding=self.sparse_embeddings,
      retrieval_mode=RetrievalMode.HYBRID,
      vector_name="dense",
      sparse_vector_name="sparse"
    )

  def save_documents(self, documents: list):
    """統一的 Hybrid Dense + Sparse 子文檔寫入接口"""
    if not documents:
      return

    self.db.add_documents(documents)

    logger.info(f"成功儲存 {len(documents)} child chunks 到 Qdrant 資料庫.")


vector_client = None


def get_vector_client():
  global vector_client

  if vector_client is None:
    # Lazy initialization：
    # 避免 Celery Fork 前初始化 CUDA，
    # 確保 Embedding Model 在 Worker 內才初始化。
    logger.info("於 Celery worker 內初始化 VectorDBClient...")
    vector_client = VectorDBClient()

  return vector_client