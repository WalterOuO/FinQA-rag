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
  1. 接收使用者問題與類別限制（經由 Pydantic 嚴格限制為 military/finance）
  2. 調度 RAG 檢索與生成流程
  3. 回傳答案、來源、Token Usage 與 Inference Performance
  """
  logger.info(f"📥 Received RAG Question. Category: [{request.category.value}], Query: '{request.question}'")

  try:
    return await query_service.answer_question(
      question=request.question,
      category=request.category.value
    )

  except Exception as e:
    logger.error(f"❌ Failed to process RAG question API: {str(e)}", exc_info=True)
    raise HTTPException(
      status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
      detail=f"問答檢索管線執行期間發生內部錯誤: {str(e)}"
    )