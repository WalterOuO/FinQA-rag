import logging
import chromadb
from config import settings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class VectorDBClient:
    def __init__(self):
      logger.info("Initializing global Embedding Model and Chroma Client...")
      self.embeddings = HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': settings.RERANKER_DEVICE}
      )
      self._init_chroma()

    def _init_chroma(self):
      # Colab 驗證或本地直接測試環境
      self.db = Chroma(
        persist_directory=str(settings.LOCAL_CHROMA_DB_DIR),
        embedding_function=self.embeddings,
        collection_name="pdfqa_collection"
      )

    def save_documents(self, documents: list):
      """統一的子文檔寫入接口"""
      if not documents:
        return
      self.db.add_documents(documents)
      logger.info(f"Successfully persisted {len(documents)} child chunks into ChromaDB.")

vector_client = None

def get_vector_client():
  global vector_client

  if vector_client is None:
    # Lazy Installization: 避免 Celery Fork 前就初始化 CUDA 造成重複初始化 CUDA 的錯誤
    logger.info("Initializing VectorDBClient inside Celery worker...")
    vector_client = VectorDBClient()

  return vector_client