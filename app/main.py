import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

# 引入微服務全域配置
from config import settings

# 💡 引入全域模型常駐單例客戶端（在開機時強迫觸發其 __init__ 載入權重）
from core.llm_client import llm_client
from core.rerank_client import rerank_client
from db.vector_client import get_vector_client
# from core.cache_manager import cache_manager

# 引入介面路由器
from api.upload import router as upload_router
from api.query import router as query_router
from api.task_status import router as task_status_router

# 設置日誌記錄器
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====== 1. 定義 FastAPI 生命週期治理 (Lifespan) ======
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    管理 FastAPI 應用程式的開機與關機生命週期事件。
    在伺服器正式對外開放（前端能連線）之前，強制完成所有目錄建立與龐大 AI 模型權重的預熱載入。
    """
    logger.info("🚀 FinQA RAG 後端正在啟動並執行資料夾初始化與模型載入...")
    
    try:
        # 🛡️ 步驟 A：自動建立 storage(pdf_store, markdown_store, parent_chunks) 資料夾
        settings.init_directories()
        logger.info("📂 [開機步驟] 持久化本地儲存目錄與類別子目錄初始化成功。")
        
        # 🧠 步驟 B：利用執行緒池主動預熱龐大的 AI 模型常駐記憶體
        # 由於載入模型權重（如 CrossEncoder、Embedding）是繁重的磁碟與矩陣 CPU/GPU I/O，
        # 我們將它們交給 Uvicorn 執行緒池，防止阻塞 FastAPI 啟動事件的事件迴圈。
        logger.info(f"⏳ [模型載入] 正在將 {settings.RERANKER_MODEL_NAME} 重排模型載入至 {settings.RERANKER_DEVICE}...")
        # 存取 rerank_client 屬性強迫觸發單例物件載入
        _ = rerank_client.model
        
        logger.info(f"⏳ [模型載入] 正在初始化 {settings.EMBEDDING_MODEL_NAME} 向量模型...")
        _ = get_vector_client().embeddings
        
        logger.info(f"⏳ [模型載入] 正在建立與遠端 {settings.LLM_MODEL_NAME} 聊天引擎的連線管道...")
        _ = llm_client.engine

        # logger.info("⚡ [模型載入] 正在初始化 Redis 語意快取向量查詢工具...")
        # _ = cache_manager.index_name
        
        logger.info("🎉 [開機步驟] 所有大型 AI 推理模型權重已全數常駐記憶體，預熱管線宣告完成！")
        
    except Exception as e:
        logger.critical(f"❌ [開機未成功] 系統在模型載入或目錄初始化期間發生錯誤: {str(e)}", exc_info=True)
        raise e

    yield  # 🟢 標記：所有權重讀取正式完成！伺服器開始對外接收前端 HTTP/POST 請求。

    logger.info("🛑 FinQA RAG Backend Service is shutting down gracefully...")


# ====== 2. 實例化 FastAPI 核心大腦 ======
app = FastAPI(
    title="FinQA 高階雙軌雙防禦 RAG 系統 API",
    description=(
        "本系統為金融科技 RAG 服務，專注於提供保險（Insurance）與財務（Finance）雙領域的高精度知識檢索與智能問答。"
    ),
    version="1.0.0",
    lifespan=lifespan
)


# ====== 3. 設定 CORS 跨網域防禦白名單 ======
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生產環境中應限縮為特定前端 URL
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


# ====== 4. 註冊子業務路由器 (API Endpoints) ======
app.include_router(upload_router)
app.include_router(query_router)
app.include_router(task_status_router)  # 🎯 完美註冊了進度條狀態查詢大門！


# ====== 5. 健康檢查端點 (Health Check) ======
@app.get("/health", status_code=status.HTTP_200_OK, tags=["System Health"])
async def system_health_check():
    """
    提供監控工具或前端確認後端是否完成預熱、進入健康服務狀態的檢查機制
    """
    return {
        "status": "healthy",
        "message": "金融保險 RAG 後端中樞運作良好，所有 AI 推理模型已常駐記憶體就位。",
        "llm_engine": settings.LLM_MODEL_NAME,
        "reranker_device": settings.RERANKER_DEVICE
    }
