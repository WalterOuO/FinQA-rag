from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定義專案根目錄 (FinQA-rag-project/)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent


class Settings(BaseSettings):

  # 1. HuggingFace Token
  HF_TOKEN: str

  # 2. 硬體運作配置
  RERANKER_DEVICE: str = "cuda"

  # 3. 單一 PDF 檔案大小上傳限制 (MB)
  PDF_FILE_MAX_SIZE: int = 50

  # 4. Celery 處理幾個 Task 就釋放記憶體
  MAX_TASKS_NUMBER: int = 50


  # Evaluation 版本選擇
  EVALUATION_VERSION: str = "E0"

  # Qdrant Server 版本對應 Port
  QDRANT_PORTS = {
      "E0": 6333,
      "E1": 6334,
      "E2": 6335,
      "E3": 6336,
  }

  QDRANT_PORT: int = QDRANT_PORTS[EVALUATION_VERSION]
  QDRANT_URL: str = f"http://127.0.0.1:{QDRANT_PORT}"

  # 地端 Qdrant Storage 獨立儲存每個 Evaluation的資料
  QDRANT_STORAGE_DIR: Path = Path("/content/qdrant_db") / EVALUATION_VERSION


  # Chunking 參數設定
  CHILD_CHUNK_SIZE: int = 400
  CHILD_CHUNK_OVERLAP: int = 50

  # Retrieval 參數設定
  REDIS_SEMANTIC_SEARCH_ALGORITHM: str = "FLAT"
  REDIS_SEMANTIC_CACHE_THRESHOLD: float = 0.92
  HYBRID_SEARCH_TOP_K: int = 3
  RERANK_OUT_TOP_K: int = 3

  # 7. Microservice 連線端點
  REDIS_URL: str = "redis://localhost:6379/0"
  VLLM_URL: str = "http://localhost:8000"

  # 8. RAG 核心模型
  EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
  EMBEDDING_MODEL_DIMENSION: int = 512

  RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
  LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8"

  # 6. 持久化雲端儲存路徑
  STORAGE_DIR: Path = PROJECT_ROOT / "storage"
  MARKDOWN_STORE_DIR: Path = PROJECT_ROOT / "storage" / EVALUATION_VERSION / "markdown_store"
  PARENT_CHUNKS_DIR: Path = PROJECT_ROOT / "storage" / EVALUATION_VERSION / "parent_chunks"
  QDRANT_BACKUP_DIR: Path = PROJECT_ROOT / "storage" / EVALUATION_VERSION /"qdrant_db"


  # .env 讀取
  model_config = SettingsConfigDict(
      env_file=PROJECT_ROOT / ".env",
      env_file_encoding="utf-8",
      extra="ignore"
    )

  def init_directories(self):
    categories = ["military", "finance"]

    # PDF / Markdown / Parent Chunk 儲存目錄
    for cat in categories:
      sub_dirs = [
        self.PDF_STORE_DIR / cat,
        self.MARKDOWN_STORE_DIR / cat,
        self.PARENT_CHUNKS_DIR / cat
      ]

      for folder in sub_dirs:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".gitkeep").touch(exist_ok=True)

    # 建立目前 Evaluation Version 的地端 Qdrant Storage
    self.QDRANT_STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    # 建立目前 Evaluation Version 的雲端 Backup 儲存空間
    self.QDRANT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    (self.QDRANT_BACKUP_DIR / ".gitkeep").touch(exist_ok=True)


# 實例化全域唯一的 settings 物件
settings = Settings()
settings.init_directories()