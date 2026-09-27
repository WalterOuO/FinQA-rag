# app/api/task_status.py
import logging
import redis
from fastapi import APIRouter, HTTPException, status
from celery.result import AsyncResult
from celery_app import celery_app
from config import settings
from models.schemas import TaskStatusResponse, DocumentCategory

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/task", tags=["Task Lifecycle"])

@router.get("/{task_id}", response_model=TaskStatusResponse, status_code=status.HTTP_200_OK)
async def get_task_ingestion_status(task_id: str):
    """
    任務生命週期追蹤接口：
    """
    try:
        # 連線 Redis 反查當初上傳時由 upload_service 寫入的 Hash 快取
        r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
        mapping_data = r.hgetall(f"task_file_mapping:{task_id}")
        
        # 提取快取欄位，若因為超時（超過24小時）失效則優雅降級兜底
        file_name = mapping_data.get("file_name", "未知檔案名稱(快取已過期)")
        category_str = mapping_data.get("category", "未知類別(快取已過期)")
        
        # 透過 AsyncResult 連線 Celery 的 db_backend 提取任務最新進度
        async_result = AsyncResult(task_id, app=celery_app)
        current_status = async_result.status
        
        # 提取我們在 pipeline_task 中用 self.update_state() 塞入的自訂 Checkpoint 訊息
        meta_info = async_result.info if isinstance(async_result.info, dict) else {}
        current_stage = meta_info.get("current_stage", "Waiting in Queue")
        
        # 組裝為符合 TaskStatusResponse 的封包
        response_data = {
            "task_id": task_id,
            "file_name": file_name,
            "category": DocumentCategory(category_str),  # 強制轉為 Pydantic 認可的 Enum 物件
            "status": current_status,
            "current_stage": current_stage,
            "result": None
        }
        
        # 當背景任務完全成功時
        if current_status == "SUCCESS":
            response_data["current_stage"] = "Ingestion_Pipeline_Completed"
            # 💡 此處的 async_result.result 就是 pipeline_task.py return 的那份成功字典！
            response_data["result"] = async_result.result
            
        # 當背景任務崩潰時
        elif current_status == "FAILURE":
            response_data["current_stage"] = "Ingestion_Pipeline_Crashed"
            response_data["result"] = {"error": str(async_result.info)}
            
        return response_data
        
    except Exception as e:
        logger.error(f"Failed to fetch Celery task status for {task_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"查詢背景任務進度時發生 Redis 狀態庫連線錯誤: {str(e)}"
        )
