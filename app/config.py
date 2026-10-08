import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# 定義專案根目錄 (FinQA-rag-project/)
APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

class Settings(BaseSettings):
    
    # 1. huggingface token
    HF_TOKEN: str

    # 2. 硬體運作配置
    RERANKER_DEVICE: str = "cuda"  # 本機時 docker-compose會改成"cpu"
    # OCR_USE_GPU: bool = True

    # 3. 單一 PDF 檔案大小限制 (MB)
    PDF_FILE_MAX_SIZE: int = 50  
    
    # 4. Celery 處理幾個 Task 就釋放記憶體
    MAX_TASKS_NUMBER: int = 50  # 本機設為 3

    # # PaddleOCR 切換 Server版 / 輕量版
    # DET_MODEL: str = "https://bcebos.com"    # 輕量版直接留空白
    # REC_MODEL: str = "https://bcebos.com"    # 輕量版直接留空白
    
    # 5. parameters
    # Ingestion
    CHILD_CHUNK_SIZE: int = 400
    CHILD_CHUNK_OVERLAP: int = 50
    # Query
    REDIS_SEMANTIC_SEARCH_ALGORITHM: str = "FLAT" # FLAT演算法（暴力搜尋）適合1萬筆以下規模數據，最準。HNSW適合規模大(記憶體開銷較大)95%準而已
    REDIS_SEMANTIC_CACHE_THRESHOLD: float = 0.92    # query與過往query相似度高於多少就直接給答案
    HYBRID_SEARCH_TOP_K: int = 3
    RERANK_OUT_TOP_K: int = 3

    # 6. 持久化本地儲存路徑
    STORAGE_DIR: Path = PROJECT_ROOT / "storage"
    PDF_STORE_DIR: Path = PROJECT_ROOT / "storage" / "pdf_store" 
    MARKDOWN_STORE_DIR: Path = PROJECT_ROOT / "storage" / "markdown_store"
    PARENT_CHUNKS_DIR: Path = PROJECT_ROOT / "storage" / "parent_chunks"
    
    # 在本地 Colab 運作 Qdrant 資料庫避免頻繁drive網路傳輸
    # LOCAL_QDRANT_DB_DIR: Path = PROJECT_ROOT / "storage" / "child_vector_db"
    QDRANT_URL: str = "http://127.0.0.1:6333"
    LOCAL_QDRANT_DB_DIR: Path = Path("/content/qdrant_db")

    # Google Drive：Qdrant 持久化備份
    QDRANT_BACKUP_DIR: Path = PROJECT_ROOT / "storage" / "child_vector_db"
    QDRANT_BACKUP_FILE: Path = QDRANT_BACKUP_DIR / "vector_db_backup.tar.gz"

    # 7. microservice連線端點
    REDIS_URL: str = "redis://localhost:6379/0"
    VLLM_URL: str = "http://localhost:8000"


    # 8. RAG 核心模型選用
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-small-zh-v1.5"
    EMBEDDING_MODEL_DIMENSION: int = 512  # 要隨著更換embedding模型而更改向量維度
    RERANKER_MODEL_NAME: str = "BAAI/bge-reranker-base"
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8" # 本機 "llama3.2-taiwan:1b"

    # 允許從 .env 檔案自動讀取
    model_config = SettingsConfigDict(
      env_file=PROJECT_ROOT / ".env", 
      env_file_encoding="utf-8",
      extra="ignore"
    )
    
    def init_directories(self):
      categories = ["military", "finance"]

      """確保所有實體儲存目錄在上線前皆已自動建立，並附帶 .gitkeep 以利 Git 追蹤"""
      for cat in categories:
        # 定義所有需要建立的子目錄路徑
        sub_dirs = [
          self.PDF_STORE_DIR / cat,
          self.MARKDOWN_STORE_DIR / cat,
          self.PARENT_CHUNKS_DIR / cat
        ]
        
        # 批次建立目錄並產生 .gitkeep
        for folder in sub_dirs:
          folder.mkdir(parents=True, exist_ok=True)
          (folder / ".gitkeep").touch(exist_ok=True)

      # 建立本地資料庫目錄與其 .gitkeep
      self.LOCAL_QDRANT_DB_DIR.mkdir(parents=True, exist_ok=True)
      self.QDRANT_BACKUP_DIR.mkdir(parents=True, exist_ok=True)
      (self.QDRANT_BACKUP_DIR / ".gitkeep").touch(exist_ok=True)

# 實例化全域唯一的 settings 物件
settings = Settings()
settings.init_directories()