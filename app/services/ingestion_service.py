import os
import logging
import shutil
import redis
from celery import uuid
from fastapi import UploadFile
from config import settings
from models.schemas import DocumentCategory
from tasks.pipeline_task import process_pdf_pipeline

logger = logging.getLogger(__name__)

class UploadService:
  async def handle_pdf_upload(self, file: UploadFile, category: DocumentCategory) -> dict:
    """
    處理 PDF 文件存檔，並派發 Celery Pipeline 非同步任務
    """
    # 💡 乾淨的檔名，並精準定位到 /storage/pdf_store/{insurance|finance}/ 檔案夾
    safe_filename = file.filename.replace(" ", "_")
    saved_file_path = settings.PDF_STORE_DIR / category.value / safe_filename
    
    logger.info(f"0️⃣ 正將上傳的檔案: {safe_filename} 儲存至雲端硬碟: {saved_file_path}")
    
    # 讀取二進位內容寫入硬碟, 防止檔案指標卡在檔案尾端產生空pdf，進而阻塞後續process_pdf任務
    content = await file.read()
    try:
      with open(saved_file_path, "wb") as f:
        f.write(content)
    finally:
      await file.close()
    # try:
    #   with open(saved_file_path, "wb") as buffer:
    #     buffer.write(content)
    #     buffer.flush()  # 強制將 Python 緩衝區寫入作業系統
    #     os.fsync(buffer.fileno())  # 強制將快取同步到磁碟
    # finally:
    #   await file.close()  # 釋放檔案控制權
        
    logger.info(f"1️⃣ 檔案落地儲存成功, 正在派發非同步任務到 Celery Queue...")
    
    # 建立 Redis 連線
    r = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    # 預先生成專屬的 Task ID
    task_id = uuid()
    mapping_key = f"task_file_mapping:{task_id}"
    
    # 用 HSET 將 filename 與 category 一起包進同一個 Hash 鍵中
    r.hset(mapping_key, mapping={"file_name": file.filename, "category": category.value})
    r.expire(mapping_key, 7200)  # 快取存活 2 小時


    # 發送非同步任務至 Redis Broker。傳入 category.value 確保 pipeline_task 的資料標記完全一致
    task = process_pdf_pipeline.apply_async(
      kwargs={
        "file_path": str(saved_file_path),
        "category": category.value,
        "file_name": file.filename
      },
      task_id=task_id
    )
    
    logger.info(f"2️⃣ 成功派發非同步任務. Task ID: {task.id}")
    
    return {
      "task_id": task_id,
      "status": "QUEUED",
      "message": "文件已加入非同步任務隊伍中等待排隊處理。",
      "filename": file.filename,
      "category": category
    }

# 全域單例
ingestion_service = UploadService()