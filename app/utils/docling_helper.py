import logging
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import (
  AcceleratorDevice,
  AcceleratorOptions
)
from docling.datamodel.pipeline_options import (
  PdfPipelineOptions,
  RapidOcrOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption

logger = logging.getLogger(__name__)


class DoclingParser:
  def __init__(self):
    logger.info("初始化 Docling PDF Parser...")

    pipeline_options = PdfPipelineOptions(
      accelerator_options=AcceleratorOptions(
        device=AcceleratorDevice.CUDA
      ),
      do_ocr=True,
      do_table_structure=True,
      ocr_options=RapidOcrOptions(
        backend="torch",
        lang=["chinese_cht"],
        force_full_page_ocr=False
      )
    )

    self.converter = DocumentConverter(
      allowed_formats=[InputFormat.PDF],
      format_options={
        InputFormat.PDF: PdfFormatOption(
          pipeline_options=pipeline_options
        )
      }
    )

    logger.info("Docling PDF Parser 初始化完成 (CUDA + RapidOCR Torch + Table Structure)")

  def parse_pdf(self, pdf_path: str | Path) -> str:
    pdf_path = Path(pdf_path)

    logger.info(f"開始使用 Docling 解析 PDF: {pdf_path.name}")

    try:
      result = self.converter.convert(pdf_path)
      document = result.document

      markdown = document.export_to_markdown(
        page_break_placeholder="<!-- PAGE_BREAK -->",
        image_mode="placeholder"
      )

      logger.info(f"Docling PDF 解析完成: {pdf_path.name}, pages={document.num_pages()}")

      return markdown

    except Exception as e:
      logger.error(f"❌ Docling PDF 解析失敗: {pdf_path.name}: {str(e)}", exc_info=True)
      raise

# Lazy initialization
docling_parser = None

def get_docling_parser() -> DoclingParser:
  global docling_parser

  if docling_parser is None:
    logger.info("Lazy initialization: 建立 DoclingParser...")
    docling_parser = DoclingParser()

  return docling_parser