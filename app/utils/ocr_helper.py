import logging
import json
import numpy as np
from paddleocr import PaddleOCR

logger = logging.getLogger(__name__)


class OCRHelper:
    def __init__(self):
      self.ocr_engine = None

    def _get_ocr_engine(self):
      """
      Lazy initialization：
      第一次真正需要 OCR 時才建立 PaddleOCR，
      後續頁面重用同一個 engine。
      """
      if self.ocr_engine is None:
        logger.info("Initializing PaddleOCR 3.x...")

        self.ocr_engine = PaddleOCR(
            lang="ch",
            device="gpu"
        )

        logger.info("PaddleOCR initialized successfully.")

      return self.ocr_engine

    def extract_text_from_page(self, page) -> str:
        """
        將 pdfplumber 的頁面渲染為圖片並執行 PaddleOCR。
        PaddleOCR engine 在第一次使用時初始化，
        後續頁面重用同一個 engine。
        """
        logger.info(f"Triggering PaddleOCR for page {page.page_number}")

        try:
          # 第一次呼叫時才初始化 PaddleOCR
          ocr_engine = self._get_ocr_engine()

          # 將 PDF 頁面渲染成 150 DPI 的圖片
          pix = page.to_image(resolution=150)
          pil_img = pix.original
          img_array = np.array(pil_img)

          # PaddleOCR 3.x 推論
          ocr_result = ocr_engine.predict(img_array)

          # 解析 PaddleOCR 3.x 結果
          lines = []

          for result in ocr_result:
            data = result.json

            if isinstance(data, str):
              data = json.loads(data)

            ocr_data = data.get("res", data)
            texts = ocr_data.get("rec_texts", [])

            for text in texts:
              if text:
                lines.append(text)

          if lines:
              return "\n".join(lines) + "\n"
          return ""

        except Exception as e:
          logger.exception(
            f"Error during OCR Processing on page "
            f"{page.page_number}: {str(e)}"
          )
          return ""

# 導出 Helper 實例
ocr_helper = OCRHelper()

# import gc
# import logging
# import numpy as np
# import torch
# from paddleocr import PaddleOCR
# from config import settings

# logger = logging.getLogger(__name__)

# class OCRHelper:
#     @staticmethod
#     def extract_text_from_page(page) -> str:
#         """
#         將 pdfplumber 的頁面渲染為圖片並執行 PaddleOCR 辨識，
#         完成後自動且強制清空 VRAM 顯存與系統記憶體。
#         """
#         logger.info(f"Triggering PaddleOCR v4 for page {page.page_number}")
        
#         # 💡 每次動態宣告：配合 Celery 的單執行緒限制 (-c 1) 且用完立刪，確保 VRAM 不會發生 OOM
#         ocr_engine = PaddleOCR(
#             ocr_version="PP-OCRv4",
#             det_model_dir=settings.DET_MODEL,
#             rec_model_dir=settings.REC_MODEL,
#             use_angle_cls=True,
#             lang="ch",
#             use_gpu=settings.OCR_USE_GPU,
#             ir_optim=True,
#             enable_mkldnn=True,
#             show_log=False
#         )
        
#         try:
#             # 將 PDF 頁面渲染成 150 DPI 的圖片並轉為 Numpy 陣列
#             pix = page.to_image(resolution=150)
#             pil_img = pix.original
#             img_array = np.array(pil_img)
            
#             # 執行 PaddleOCR 推論
#             ocr_result = ocr_engine.ocr(img_array, cls=True)
            
#             # 解析並拼接辨識出的文字
#             if ocr_result and ocr_result[0]:
#                 lines = [line[1][0] for line in ocr_result[0] if line and len(line) > 1]
#                 return "\n".join(lines) + "\n"
#             return ""
            
#         except Exception as e:
#             logger.error(f"Error during OCR Processing on page {page.page_number}: {str(e)}")
#             return ""
            
#         finally:
#             # 💡 用完就刪除OCR，不佔用VRAM
#             del ocr_engine
#             gc.collect()
#             if torch.cuda.is_available():
#                 torch.cuda.empty_cache()

# # 導出 Helper 實例
# ocr_helper = OCRHelper()