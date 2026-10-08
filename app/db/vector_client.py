import logging

from fastembed import SparseTextEmbedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
from qdrant_client import QdrantClient, models
from config import settings

logger = logging.getLogger(__name__)

class FastEmbedBM25(SparseEmbeddings):
  def __init__(self):
    self.model = SparseTextEmbedding(
      model_name="Qdrant/bm25"
    )

  def embed_documents(self, texts: list[str]) -> list[SparseVector]:
    embeddings = self.model.embed(texts)
    return [
      SparseVector(
        indices=embedding.indices.tolist(),
        values=embedding.values.tolist()
      ) for embedding in embeddings
      ]

  def embed_query(self, text: str) -> SparseVector:
    embedding = next(self.model.embed([text]))
    return SparseVector(
      indices=embedding.indices.tolist(),
      values=embedding.values.tolist()
    )

class VectorDBClient:
  def __init__(self):
    logger.info("初始化 Embedding Model 與 Qdrant Client...")

    # Embedding model：Dense Vector
    self.embeddings = HuggingFaceEmbeddings(
      model_name=settings.EMBEDDING_MODEL_NAME,
      model_kwargs={"device": settings.RERANKER_DEVICE}
    )

    self.sparse_embeddings = FastEmbedBM25()

    self._init_qdrant()

  def _init_qdrant(self):
    self.client = QdrantClient(url=settings.QDRANT_URL)
    collection_name = "pdfqa_collection"

    if not self.client.collection_exists(collection_name):
      dense_dim = len(self.embeddings.embed_query("dimension check"))

      self.client.create_collection(
        collection_name=collection_name,
        vectors_config={
          "dense": models.VectorParams(
            size=dense_dim,
            distance=models.Distance.COSINE
          )
        },
        sparse_vectors_config={
          "sparse": models.SparseVectorParams()
        }
      )

      logger.info(f"建立 Qdrant collection: {collection_name}")

    self.db = QdrantVectorStore(
      client=self.client,
      collection_name=collection_name,
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

    logger.info(f"成功儲存 {len(documents)} chunks 到 Qdrant 資料庫.")


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