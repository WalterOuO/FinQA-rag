import os
import sys
import onnxruntime as ort
print("[Celery ORT] Python:", sys.executable, flush=True)
print("[Celery ORT] Version:", ort.__version__, flush=True)
print("[Celery ORT] Package:", ort.__file__, flush=True)
print("[Celery ORT] Build:", ort.get_build_info(), flush=True)
print("[Celery ORT] Providers:", ort.get_available_providers(), flush=True)
try:
    devices = ort.get_ep_devices()
    for device in devices:
      print("[Celery] EP device:", device, flush=True)
except Exception as e:
    print("[Celery] get_ep_devices unavailable:", repr(e), flush=True)
print("[Celery ORT] LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"), flush=True)
ort.set_default_logger_severity(3)

import os
import json
import logging
from pathlib import Path

import redis
from docling.chunking import HybridChunker
from langchain_core.documents import Document

from config import settings
from celery_app import celery_app
from db.vector_client import get_vector_client
from utils.docling_helper import get_docling_parser

logger = logging.getLogger(__name__)


class IngestionTaskDLQ(celery_app.Task):
  def on_failure(self, exc, task_id, args, kwargs, einfo):
    logger.critical(
        f"🚨 [DLQ Triggered] Task {task_id} failed permanently! "
        f"Error: {str(exc)}"
    )

    file_name = kwargs.get("file_name", "Unknown_File")
    category = kwargs.get("category", "Unknown_Category")
    file_path = kwargs.get("file_path", "Unknown_Path")

    r = redis.Redis.from_url(settings.REDIS_URL)
    dlq_report = {
      "file_name": file_name,
      "category": category,
      "error_msg": str(exc),
      "local_path": file_path,
      "status": "FAILED",
    }
    r.hset(
      "rag:dlq:failed_tasks",
      task_id,
      json.dumps(dlq_report, ensure_ascii=False),
    )


@celery_app.task(
  name="tasks.pipeline_task.process_pdf_pipeline",
  bind=True,
  base=IngestionTaskDLQ,
)
def process_pdf_pipeline(self, file_path: str, category: str, file_name: str):
  logger.info(f"🚀 開始非同步處理文件 [{category}]: {file_name}")
  pdf_path = Path(file_path)

  # ====== 1. PDF Parsing: Docling + OCR ======
  self.update_state(state="PROCESSING", meta={"current_stage": "DOCLING_&_OCR"})

  try:
    docling_parser = get_docling_parser()
    docling_document = docling_parser.convert_pdf(pdf_path)

    full_markdown = docling_document.export_to_markdown(
      page_break_placeholder="<!-- PAGE_BREAK -->",
      image_mode="placeholder",
    )

  except Exception as e:
    logger.error(f"❌ Docling PDF 解析出錯: {e}", exc_info=True)
    raise

  md_file_path = settings.MARKDOWN_STORE_DIR / category / f"{pdf_path.stem}.md"
  md_file_path.parent.mkdir(parents=True, exist_ok=True)

  with open(md_file_path, "w", encoding="utf-8") as f:
    f.write(full_markdown)

  logger.info(f"✅ Ingestion Step 1. PDF Docling Parsing 完成，Markdown 已存入: {md_file_path}")

  # ====== 2. Chunking: Docling HybridChunker ======
  self.update_state(state="PROCESSING",  meta={"current_stage": "Docling HybridChunking"})

  # 使用與 embedding 模型相符的 tokenizer
  chunker = HybridChunker(
    tokenizer=settings.EMBEDDING_MODEL_NAME,
    merge_peers=True)

  child_documents = []
  chunks_store = {}

  for c_idx, chunk in enumerate(chunker.chunk(dl_doc=docling_document)):
    chunk_text = chunker.contextualize(chunk).strip()

    if not chunk_text:
      continue

    # 取得頁碼資訊
    page_numbers = sorted({
      prov.page_no
      for doc_item in chunk.meta.doc_items
      for prov in doc_item.prov
      if getattr(prov, "page_no", None) is not None
    })

    child_id = f"{pdf_path.stem}_C_{c_idx}"
    c_metadata = {
      "chunk_id": child_id,
      "category": category,
      "file_name": file_name,
      "page_numbers": page_numbers,
      "headings": chunk.meta.headings or [],
    }

    chunks_store[child_id] = {
      "page_content": chunk_text,
      "metadata": c_metadata,
    }

    child_documents.append(Document(page_content=chunk_text, metadata=c_metadata))

  chunks_json_path = (settings.PARENT_CHUNKS_DIR / category / f"{pdf_path.stem}_chunks.json")
  chunks_json_path.parent.mkdir(parents=True, exist_ok=True)

  with open(chunks_json_path, "w", encoding="utf-8") as f:
    json.dump(chunks_store, f, ensure_ascii=False, indent=2)

  logger.info(f"✅ Ingestion Step 2. HybridChunker 完成，共 {len(child_documents)} 個 chunks。紀錄已存入 {chunks_json_path}")

  # ====== 3. Embedding + Qdrant ======
  self.update_state(state="PROCESSING", meta={"current_stage": "Vectorizing_&_Qdrant_Storing"},)

  if child_documents:
    get_vector_client().save_documents(child_documents)

  logger.info(f"✅ Ingestion Step 3. 文件 {file_name} 處理完成，共寫入 {len(child_documents)} 個 chunks。")

  if pdf_path.exists():
    os.remove(pdf_path)

  return {
    "status": "success",
    "file_name": file_name,
    "chunks_count": len(child_documents),
  }
