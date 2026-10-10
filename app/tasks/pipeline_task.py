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


import numpy as np
from pathlib import Path
model = Path("/usr/local/lib/python3.13/dist-packages/rapidocr/models/PP-OCRv6_det_small.onnx")
session = ort.InferenceSession(str(model), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
inp = session.get_inputs()[0]
x = np.zeros([1, 3, 64, 64], dtype=np.float32)
try:
    y = session.run(None, {inp.name: x})
    print("[Celery CUDA Test] Inference succeeded", flush=True)
    print("[Celery CUDA Test] Providers:", session.get_providers(), flush=True)
    session = None
except Exception as e:
    print("[Celery CUDA Test] Inference failed:", repr(e), flush=True)
    session = None

from collections import Counter
_original_session = ort.InferenceSession
ort_sessions = []

def debug_session(*args, **kwargs):
    options = kwargs.get("sess_options")
    if options is None:
        options = ort.SessionOptions()
        kwargs["sess_options"] = options

    options.enable_profiling = True
    session = _original_session(*args, **kwargs)
    ort_sessions.append(session)

    print("[ORT Session] Providers:", session.get_providers(), flush=True)
    return session
ort.InferenceSession = debug_session
ort.set_default_logger_severity(3)


import re
import json
import logging
from pathlib import Path
import redis

# 引入第三方套件與 LangChain 基礎物件
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入自訂之微服務功能
from config import settings
from celery_app import celery_app
from db.vector_client import get_vector_client
from utils.docling_helper import get_docling_parser

logger = logging.getLogger(__name__)

class IngestionTaskDLQ(celery_app.Task):
  def on_failure(self, exc, task_id, args, kwargs, einfo):
    """
    當整個 PDF Ingestion 崩潰失敗時，此函數會自動被觸發 (等同進入死信隊列Dead Letter Queue)
    """
    logger.critical(f"🚨 [DLQ Triggered] Task {task_id} failed permanently! Error: {str(exc)}")
    
    # 從參數中抓出檔案資訊
    file_name = kwargs.get("file_name", "Unknown_File")
    category = kwargs.get("category", "Unknown_Category")
    file_path = kwargs.get("file_path", "Unknown_Path")
    
    # 將失敗的任務資訊丟入 Redis 的獨立死信名冊中 (rag:dlq:failed_tasks)，方便日後人工審查
    r = redis.Redis.from_url(settings.REDIS_URL)
    
    dlq_report = {
      "file_name": file_name,
      "category": category,
      "error_msg": str(exc),
      "local_path": file_path,
      "status": "FAILED"
    }
    r.hset("rag:dlq:failed_tasks", task_id, json.dumps(dlq_report, ensure_ascii=False))


CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=settings.CHILD_CHUNK_SIZE, chunk_overlap=settings.CHILD_CHUNK_OVERLAP)

@celery_app.task(name="tasks.pipeline_task.process_pdf_pipeline", bind=True, base=IngestionTaskDLQ)
def process_pdf_pipeline(self, file_path: str, category: str, file_name: str):
    """
    非同步處理PDF pipeline：
    1. PDF Parsing: pdfplumber 處理文字/表格
    2. RecursiveTextSplitter Chunking: 單純最普通 chunking 方法
    3. Embedding: 存入 ChromaDB
    """
    logger.info(f"🚀 開始非同步處理文件 [{category}]: {file_name}")

    # ====== 1. PDF Parsing: Docling + OCR ======
    self.update_state(state="PROCESSING", meta={"current_stage": "DOCLING_&_OCR"})
    pdf_path = Path(file_path)
    
    try:
      docling_parser = get_docling_parser()
      full_markdown = docling_parser.parse_pdf(pdf_path)

    except Exception as e:
      logger.error(f"❌ Docling PDF 解析出錯: {str(e)}")
      raise e
    
    # 將每個PDF的 Markdown 存入對應的業務子資料夾中 (Docker Bind Mount)
    md_file_path = settings.MARKDOWN_STORE_DIR / category / f"{pdf_path.stem}.md"
    with open(md_file_path, "w", encoding="utf-8") as f:
      f.write(full_markdown)
        
    logger.info(f"✅ Ingestion Step 1. PDF Parsing 完成，{file_name}產生的 markdown 已存入: {md_file_path} 。")

    # for i, session in enumerate(ort_sessions):
    #   try:
    #     profile_path = session.end_profiling()

    #     with open(profile_path, "r", encoding="utf-8") as f:
    #         events = json.load(f)

    #     nodes = [
    #         event for event in events
    #         if event.get("cat") == "Node"
    #         and "provider" in event.get("args", {})
    #     ]

    #     counts = Counter(event["args"]["provider"] for event in nodes)

    #     print(f"[ORT Profile {i}] File: {profile_path}", flush=True)
    #     print(f"[ORT Profile {i}] Node counts: {dict(counts)}", flush=True)

    #   except Exception as e:
    #     print(f"[ORT Profile {i}] Failed: {e!r}", flush=True)
  
    # logger.info(f"✅ Ingestion Step 1.5. 成功印出 ORT Profile。")

    # ====== 2. Chunking ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Chunking"})

    child_documents = []
    chunks_store = {}

    # Recursive 切成 Chunk
    sub_chunks = CHILD_SPLITTER.split_text(full_markdown)
    for c_idx, c_doc in enumerate(sub_chunks):
      child_id = f"{pdf_path.stem}_C_{c_idx}"
      c_metadata = {
        "chunk_id": child_id,
        "category": category,
        "file_name": file_name
      }   
      chunks_store[child_id] = {
        "page_content": c_doc,
        "metadata": c_metadata
      }

      child_documents.append(Document(page_content=c_doc, metadata=c_metadata))

    # 將 Chunk 存成 JSON 檔方便日後核對
    chunks_json_path = settings.PARENT_CHUNKS_DIR / category / f"{pdf_path.stem}_chunks.json"
    with open(chunks_json_path, "w", encoding="utf-8") as f:
      json.dump(chunks_store, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Ingestion Step 2. Chunking 完成！共生成 {len(child_documents)} 個 chunks。Chunks 已存入 {chunks_json_path}")
     
    # ====== 3: Embedding 轉向量並寫入 ChromaDB ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Vectorizing_&_Chroma_Storing"})
    get_vector_client().save_documents(child_documents)
        
    logger.info(f"✅ Ingestion Step 3. 文件 {file_name} 處理完成！已將 {len(child_documents)}個 chunks 寫入向量資料庫。")
    
    # 清除暫存的 PDF path，節省硬碟空間
    if pdf_path.exists():
      os.remove(pdf_path)
        
    return {"status": "success", "file_name": file_name, "chunks_count": len(child_documents)}
