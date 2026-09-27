import logging
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from models.schemas import DocumentCategory, UploadResponse
from services.upload_service import upload_service
from config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/upload", tags=["Upload"])

@router.post("", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_pdf_document(
    file: UploadFile = File(..., description="要上傳的保險或財務 PDF 檔案"),
    category: DocumentCategory = Form(..., description="文件類別，限定：insurance 或 finance")
):
    """
    接收 PDF 文件接口：
    1. 透過 Enum 驗證業務類別 (insurance/finance)
    2. 驗證副檔名必須為 pdf
    3. 落地保存檔案並派發任務給 Celery 後台 Pipeline
    """
    if not file.filename.lower().endswith(".pdf"):
      raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="不支援的檔案格式。僅接收 PDF 檔案。"
      )
    
    # 限制 PDF file 檔案大小 < MB_SIZE MB
    MB_SIZE = settings.PDF_FILE_MAX_SIZE
    MAX_FILE_SIZE = MB_SIZE*1024*1024
    if file.size > MAX_FILE_SIZE:
      raise HTTPException(
        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
        detail=f"檔案過大，單一 PDF 大小限制在 {MB_SIZE}MB以內。"
      )

    logger.info(f"Received upload request. File: {file.filename}, Category: {category.value}")

    try:
      # 呼叫上傳服務層處理文件的解析與派發，直接傳入 pdf, Enum 物件
      result = await upload_service.handle_pdf_upload(file, category)
      return result
    except Exception as e:
      logger.error(f"Failed to process upload API request: {str(e)}")
      raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=f"Upload API 非同步任務派發錯誤: {str(e)}"
        )

