import logging
from fastapi import APIRouter, HTTPException, status
from models.schemas import QueryRequest, QueryResponse
from services.query_service import query_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/query", tags=["RAG Query"])

@router.post("", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def ask_rag_question(request: QueryRequest):
    """
    RAG 即時問答核心接口：
    1. 接收使用者問題與類別限制（經由 Pydantic 嚴格限制為 insurance/finance）
    2. 調度語意快取檢查與 RAG 反查 + Rerank 檢索流
    3. 回傳具備高度物理追溯力的分析答案
    """
    logger.info(f"📥 Received RAG Question. Category: [{request.category.value}], Query: '{request.question}'")
        
    try:
      # 呼叫 Service 服務層，直接傳入問題文字與枚舉字串值
      result = await query_service.answer_question(
          question=request.question,
          category=request.category.value
      )
      
      return result
        
    except Exception as e:
      logger.error(f"❌ Failed to process RAG question API: {str(e)}", exc_info=True)
      raise HTTPException(
          status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
          detail=f"問答檢索管線執行期間發生內部非同步錯誤: {str(e)}"
      )
