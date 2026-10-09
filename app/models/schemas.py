from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field


# === Ingestion ===
# 一般不設 UploadRequest, 因為FastAPI的 UploadFile的 Form-data格式與 pydantic JSON格式衝突

class DocumentCategory(str, Enum):
  MILITARY = "military"
  FINANCE = "finance"


class UploadResponse(BaseModel):
  task_id: str = Field(..., description="Celery 非同步任務的全域唯一識別碼，可用於追蹤進度")
  status: str = Field(..., description="當前任務佇列狀態，如 QUEUED")
  message: str = Field(..., description="給予前端使用者的說明文字")
  filename: str = Field(..., description="原始上傳的檔案名稱")
  category: DocumentCategory = Field(..., description="指派的文件類別 (military/finance)")


# ==== Ask ====

class QueryRequest(BaseModel):
  question: str = Field(..., description="使用者輸入的即時問題", min_length=2, max_length=500)
  category: DocumentCategory = Field(..., description="指定進行語意檢索的文件類別限制 (military/finance)")


class SourceDocumentTrace(BaseModel):
  frontend_index: int = Field(..., description="前台查詢序號")
  file_name: str = Field(..., description="原始文件的完整檔案名稱")
  category: str = Field(..., description="文件業務類別")


class TokenUsage(BaseModel):
  prompt_tokens: int = Field(..., description="LLM 輸入 token 數")
  completion_tokens: int = Field(..., description="LLM 輸出 token 數")
  total_tokens: int = Field(..., description="LLM 總 token 數")


class InferencePerformance(BaseModel):
  total_latency_seconds: float = Field(..., description="LLM 完整推理延遲")
  ttft_seconds: Optional[float] = Field(None, description="Time to First Token")
  tpot_seconds: Optional[float] = Field(None, description="Time per Output Token")


class QueryResponse(BaseModel):
  answer: str = Field(..., description="由自架 vLLM 生成的繁體中文回答")
  sources: List[SourceDocumentTrace] = Field(..., description="來源文件追蹤資訊")
  context: str = Field(..., description="實際餵給 LLM 的 context_str")
  token_usage: TokenUsage = Field(..., description="本次 LLM 推理的 token 使用量")
  performance: InferencePerformance = Field(..., description="LLM inference performance metrics")


# ==== Task Status ====

class TaskStatusResponse(BaseModel):
  task_id: str = Field(..., description="任務的全域唯一識別碼")
  file_name: str = Field(..., description="該任務所對應的原始檔案名稱")
  category: DocumentCategory = Field(..., description="該任務所對應的文件類別 (military/finance)")
  status: str = Field(..., description="任務目前的 Checkpoint 狀態 (PENDING/PROCESSING/SUCCESS/FAILURE)")
  current_stage: Optional[str] = Field(None, description="Ingestion 管線目前正在執行的具體內部步驟")
  result: Optional[Any] = Field(None, description="任務成功完成後的回傳數據")