import os
import re
import json
import logging
from pathlib import Path
import redis

# 引入核心第三方套件与 LangChain 基礎物件
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 引入自訂之微服務神經中樞與解耦組件
from config import settings
from celery_app import celery_app
from db.vector_client import get_vector_client
# from utils.ocr_helper import ocr_helper

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
    
    self.update_state(state="PROCESSING", meta={"current_stage": "OCR_&_Content_Parsing"})

    pdf_path = Path(file_path)
    markdown_content = []

    # ====== 1. PDF Parsing ======
    try:
      with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
          page_num = page.page_number
          markdown_content.append(f"\n<!-- PAGE_START_{page_num} -->\n")
          
          # Text
          text = page.extract_text()
          if text:
            markdown_content.append("\n<!-- TEXT_START -->\n")
            markdown_content.append(text)
            markdown_content.append("\n<!-- TEXT_END -->\n")
          
          # Table
          tables = page.extract_tables()
          for table in tables:
            if not table:
              continue

            markdown_content.append("\n<!-- TABLE_START -->\n")
            
            for row in table:
              if row:
                # 把表格row內每一格整理成文字
                row_text = " | ".join(
                  str(cell or "").replace("\n", "").strip()
                  for cell in row
                )
                markdown_content.append(row_text + "\n")

            markdown_content.append("\n<!-- TABLE_END -->\n")

          markdown_content.append(f"\n<!-- PAGE_END_{page_num} -->\n")
                
    except Exception as e:
      logger.error(f"❌ PDF 解析出錯: {str(e)}")
      raise e
            
    # 將所有的 markdown 串成起來
    full_markdown = "".join(markdown_content)
    
    # 將每個PDF的 Markdown 存入對應的業務子資料夾中 (Docker Bind Mount)
    md_file_path = settings.MARKDOWN_STORE_DIR / category / f"{pdf_path.stem}.md"
    with open(md_file_path, "w", encoding="utf-8") as f:
      f.write(full_markdown)
        
    logger.info(f"✅ Ingestion Step 1. PDF Parsing 完成，{file_name}產生的 markdown 已存入: {md_file_path} 。")

    # ====== 2. Chunking ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Chunking"})

    child_documents = []
    
    # Recursive 切成 Chunk
    sub_chunks = CHILD_SPLITTER.split_text(full_markdown)
    for c_idx, c_doc in enumerate(sub_chunks):
      c_metadata = {
        "chunk_id": f"{pdf_path.stem}_C_{c_idx}",
        "category": category,
        "file_name": file_name
      }    
      child_documents.append(Document(page_content=c_doc, metadata=c_metadata))

    logger.info(f"✅ Ingestion Step 2. Chunking 完成！共生成 {len(child_documents)} 個 chunks。")
    
    # ====== 3: Embedding 轉向量並寫入 ChromaDB ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Vectorizing_&_Chroma_Storing"})
    get_vector_client().save_documents(child_documents)
        
    logger.info(f"✅ Ingestion Step 3. 文件 {file_name} 處理完成！已將 chunks 寫入向量資料庫。")
    
    # 清除暫存的 PDF path，節省硬碟空間
    if pdf_path.exists():
      os.remove(pdf_path)
        
    return {"status": "success", "file_name": file_name, "child_chunks_count": len(child_documents)}
