import logging
from celery import Celery
from config import settings

logger = logging.getLogger(__name__)

# 初始化 Celery 物件
celery_app = Celery(
  "finqa_rag_worker",
  broker=settings.REDIS_URL,
  backend=settings.REDIS_URL,
  include=["tasks.pipeline_task"]
)

# ====== ⚙️ Celery 高級效能與穩定性配置 ======
celery_app.conf.update(
    # 1. 任務序列化格式與時區
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="Asia/Taipei",
    enable_utc=True,

    # 2. 任務結果存活時間 (TTL)
    # 設定 2 (3600*2) 小時後自動從 Redis 釋放，防止 Redis 記憶體暴增
    result_expires=7200,

    # 3. 失敗任務與連線重試機制
    broker_connection_retry_on_startup=True,
    task_reject_on_worker_lost=True,    # 若 Worker 突然崩潰 (如 OOM)，將任務放回佇列重新執行
    task_acks_late=True,          # 任務完全執行成功後才發送確認信號 (ACK)

    # 4. 預取機制優化 (Prefetch Limits)
    # 每個 Worker 提早把 K 個任務從 Redis 拿過來塞在自己的預備隊列
    # (-c 2) 時可以避免 worker任務分配不均
    # 單執行緒 (-c 1)時，任務能一個一個 PDF 解析排隊，讓 VRAM 不爆炸。
    worker_prefetch_multiplier=1,

    # 5. 防止記憶體洩漏 (Memory Leak)
    # 雖然在 OCR寫了 empty_cache()，但 Paddle 在 Linux 仍可能有 C++ 層面的記憶體殘留。
    # 設定 Worker 每執行完 3-50 個任務就自動重啟並釋放所有系統 RAM，完全不影響線上問答。
    worker_max_tasks_per_child=settings.MAX_TASKS_NUMBER,
)

logger.info("Celery asynchronous infrastructure initialized successfully.")