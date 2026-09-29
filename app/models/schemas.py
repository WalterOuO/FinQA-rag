from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field

# === Ingestion ===
# 一般不設 UploadRequest, 因為FastAPI的 UploadFile的 Form-data格式與 pydantic JSON格式衝突

class DocumentCategory(str, Enum):
  INSURANCE = "insurance"
  FINANCE = "finance"

class UploadResponse(BaseModel):
  task_id: str = Field(..., description="Celery 非同步任務的全域唯一識別碼，可用於追蹤進度")
  status: str = Field(..., description="當前任務佇列狀態，如 QUEUED")
  message: str = Field(..., description="給予前端使用者的說明文字")
  filename: str = Field(..., description="原始上傳的檔案名稱")
  category: DocumentCategory = Field(..., description="指派的文件業務類別 (insurance/finance)")

# ==== Ask ====

class QueryRequest(BaseModel):
    question: str = Field(..., description="使用者輸入的即時金融或保險問題", min_length=2, max_length=500)
    category: DocumentCategory = Field(..., description="要指定進行語意檢索的業務類別限制 (insurance/finance)")


class SourceDocumentTrace(BaseModel):
    frontend_index: int = Field(..., description="前台查詢序號")
    file_name: str = Field(..., description="原始文件的完整檔案名稱")
    category: str = Field(..., description="文件業務類別")
    parent_id: str = Field(..., description="實體硬碟 JSON 中的大父文檔唯一 ID（用於工程反查與人工審閱）")
    page_num_list: list = Field(..., description="LLM生成的回答來自的原始文件頁數")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="由 LLM (vLLM/Ollama) 依據保證憑據所生成的繁體中文分析回答")
    sources: List[SourceDocumentTrace] = Field(..., description="完整的來源文件追溯矩陣，兼顧工程與人工審查能力")
    cached: bool = Field(..., description="標記此回答是否直接命中 Redis 語意快取")

# ==== Task Status ====

class TaskStatusResponse(BaseModel):
    task_id: str = Field(..., description="任務的全域唯一識別碼")
    file_name: str = Field(..., description="該任務所對應的原始檔案名稱")
    category: DocumentCategory = Field(..., description="該任務所對應的業務類別 (insurance/finance)")
    status: str = Field(..., description="任務目前的 Checkpoint 狀態 (PENDING/PROCESSING/SUCCESS/FAILURE)")
    current_stage: Optional[str] = Field(None, description="Ingestion 管線目前正在執行的具體內部步驟")
    result: Optional[Any] = Field(None, description="任務成功完成後的回傳數據")