import os
import json
import logging
from pathlib import Path
import redis

# 引入核心第三方套件与 LangChain 基礎物件
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

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


HEADERS_TO_SPLIT = [("#", "Header1"), ("##", "Header2"), ("###", "Header3")]
# strip_headers=False 來阻止MarkdownHeaderTextSplitter砍掉 #標題
MD_SPLITTER = MarkdownHeaderTextSplitter(headers_to_split_on=HEADERS_TO_SPLIT, strip_headers=False)
CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=settings.CHILD_CHUNK_SIZE, chunk_overlap=settings.CHILD_CHUNK_OVERLAP)

@celery_app.task(name="tasks.pipeline_task.process_pdf_pipeline", bind=True, base=IngestionTaskDLQ)
def process_pdf_pipeline(self, file_path: str, category: str, file_name: str):
    """
    非同步處理PDF pipeline：
    1. Hybrid Parsing: pdfplumber 處理文字/表格 + PaddleOCR 處理圖片
    2. Parent-Child Chunking: 雙層結構切片，並將父文檔落盤快取
    3. Embedding & Store: 自動存入獨立的 ChromaDB, 父文檔
    若無法處理具備Dead Letter Queue功能
    """
    logger.info(f"🚀 開始非同步處理文件 [{category}]: {file_name}")
    
    self.update_state(state="PROCESSING", meta={"current_stage": "OCR_&_Content_Parsing"})

    pdf_path = Path(file_path)
    markdown_content = []
    
    # ====== 1. Hybrid PDF Parsing (順序拼接) ======
    try:
      with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
          page_num = page.page_number
          markdown_content.append(f"\n<!-- PAGE_START_{page_num} -->\n")
          
          # 1. 檢查是否為圖片型 PDF
          # text = page.extract_text()
          # if not text or len(text.strip()) < 40:
          #   logger.info(f"Page {page_num} seems to be scanned/image. Triggering OCR...")
          #   ocr_text = ocr_helper.extract_text_from_page(page)
          #   # 提示：PaddleOCR 純文字很難辨識出 # 標題，
          #   # 若需要段落，可在 ocr_text 上做規則微調，或依靠大段落分塊
          #   markdown_content.append(ocr_text)
          #   markdown_content.append(f"\n<!-- PAGE_END_{page_num} -->\n")
          #   continue # 這頁就只走 OCR 路線，直接跳過後面步驟
          
          # 2. 基於字型大小區分段落架構 (針對文字型 PDF)
          # 撈出此頁所有字元的詳細資料，計算出最常出現的字體大小（通常是正文）
          chars = page.chars
          if chars:
            sizes = [round(c["size"], 1) for c in chars if "size" in c]
            body_size = max(set(sizes), key=sizes.count) if sizes else 10.0
          else:
            body_size = 10.0

          # 使用 extract_words 拿到帶有字體大小與座標的單字，重建行結構
          words = page.extract_words(keep_blank_chars=False)
          
          # 將單字依照靠上的座標(top)分組，重新組合出每一行
          lines = {}
          for w in words:
            # 容許上下 2 像素的誤差視為同一行
            top_bucket = round(w["top"] / 2) * 2
            lines.setdefault(top_bucket, []).append(w)
          
          # 按垂直位置從上到下排序處理每一行
          for top in sorted(lines.keys()):
            line_words = sorted(lines[top], key=lambda x: x["x0"])
            line_text = "".join([w["text"] for w in line_words]).strip()
            
            if not line_text:
              continue
            
            # 取這一行所有字元的最大字體大小
            line_max_size = max([w.get("size", body_size) for w in line_words])
            
            # 根據字體大小動態打上 Markdown 標題標籤！
            if line_max_size >= body_size + 6:
                markdown_content.append(f"\n# {line_text}\n")     # 一級標題
            elif line_max_size >= body_size + 3:
                markdown_content.append(f"\n## {line_text}\n")     # 二級標題
            elif line_max_size >= body_size + 1.5:
                markdown_content.append(f"\n### {line_text}\n")     # 三級標題
            else:
                markdown_content.append(line_text + "\n")        # 一般正文

          # 3. 擷取表格並轉換成標準 Markdown Grid 語法
          tables = page.extract_tables()
          for table in tables:
            if not table or not table[0]:
              continue
            
            # 取出第一列作為 Header
            header_row = table[0]
            markdown_content.append("\n| " + " | ".join([str(cell or '').replace('\n', ' ') for cell in header_row]) + " |")
            markdown_content.append("| " + " | ".join(["---" for _ in header_row]) + " |")
            
            # 寫入其餘資料列
            for row in table[1:]:
                markdown_content.append("| " + " | ".join([str(cell or '').replace('\n', ' ') for cell in row]) + " |")
            markdown_content.append("\n")
                  
          markdown_content.append(f"\n<!-- PAGE_END_{page_num} -->\n")
                
    except Exception as e:
        logger.error(f"❌ PDF 解析出錯: {str(e)}")
        raise e
            
    # 將所有順序拼接的 markdown 串接成大字串
    full_markdown = "".join(markdown_content)
    
    # 將每個PDF的 Markdown 存入對應的業務子目錄中 (Docker Bind Mount)
    md_file_path = settings.MARKDOWN_STORE_DIR / category / f"{pdf_path.stem}.md"
    with open(md_file_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)
        
    # ====== 2. Parent-Child Chunking ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Parent_Child_Chunking"})
    # A. 依據 Markdown 標題切出「大父文件」
    parent_docs = MD_SPLITTER.split_text(full_markdown)
    parent_store = {}
    child_documents = []
    
    for p_idx, p_doc in enumerate(parent_docs):
        parent_id = f"{pdf_path.stem}_P_{p_idx}"
        
        # 更新父文件的 Metadata (除了#/## header資訊以外，加入id,類別,檔名)
        p_metadata = p_doc.metadata.copy()
        p_metadata.update({"parent_id": parent_id, "category": category, "file_name": file_name})
        
        # 父文件寫入store，用於 RAG 問答階段時的「反查」
        parent_store[parent_id] = {
            "page_content": p_doc.page_content,
            "metadata": p_metadata
        }
        
        # B. 將父文件進一步切成細碎子文件
        sub_chunks = CHILD_SPLITTER.split_text(p_doc.page_content)
        for c_idx, sub_chunk in enumerate(sub_chunks):
            # 子文件 Metadata 必須挾帶父文檔 ID、#/##Header、類別等標記(知道子從哪個父來的)
            c_metadata = p_doc.metadata.copy()
            c_metadata.update({
                "parent_id": parent_id,
                "child_id": f"{parent_id}_C_{c_idx}",
                "category": category,
                "file_name": file_name
            })
            
            child_documents.append(Document(page_content=sub_chunk, metadata=c_metadata))
            
    # 將父文件 JSON 存入資料夾 (Docker Bind Mount), 供 RAG backend/celery worker讀取
    parent_json_path = settings.PARENT_CHUNKS_DIR / category / f"{pdf_path.stem}_parents.json"
    with open(parent_json_path, "w", encoding="utf-8") as f:
        json.dump(parent_store, f, ensure_ascii=False, indent=2)

    # ====== Step 3: Embedding 轉向量並透過全域單例寫入 ChromaDB ======
    self.update_state(state="PROCESSING", meta={"current_stage": "Vectorizing_&_Chroma_Storing"})
    get_vector_client().save_documents(child_documents)
        
    logger.info(f"✅ 文件 {file_name} 處理完成！共生成 {len(parent_docs)} 個父文檔，{len(child_documents)} 個子文檔並已成功寫入向量資料庫。")
    
    # 清除暫存的 PDF path，節省硬碟空間
    if pdf_path.exists():
        os.remove(pdf_path)
        
    return {"status": "success", "file_name": file_name, "child_chunks_count": len(child_documents)}
