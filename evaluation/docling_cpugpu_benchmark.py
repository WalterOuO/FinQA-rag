import time
import statistics
import onnxruntime as ort
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.document_converter import DocumentConverter, PdfFormatOption

ort.set_default_logger_severity(3)
pdf_path = Path("/content/drive/MyDrive/FinQA-rag/docs/military/114年老年犯罪概況.pdf")
assert pdf_path.is_file()

def create_converter(ocr_use_cuda: bool):
  rapidocr_params = {
      "EngineConfig.onnxruntime.use_cuda": ocr_use_cuda,
      "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
      "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "DEFAULT",
  }
  pipeline_options = PdfPipelineOptions(
      accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CUDA),
      do_ocr=True,
      do_table_structure=True,
      ocr_options=RapidOcrOptions(backend="onnxruntime", lang=["chinese_cht"], force_full_page_ocr=False, rapidocr_params=rapidocr_params),
  )
  return DocumentConverter(
      allowed_formats=[InputFormat.PDF],
      format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)},
  )

def benchmark(name: str, ocr_use_cuda: bool, repeats: int = 3):
  print(f"\n===== {name} =====")
  init_start = time.perf_counter()
  converter = create_converter(ocr_use_cuda)
  init_seconds = time.perf_counter() - init_start
  print(f"Converter initialization: {init_seconds:.2f}s")

  # 第一次當作 warm-up，不納入平均
  result = converter.convert(pdf_path)
  warmup_chars = len(result.document.export_to_markdown())
  print(f"Warm-up Markdown chars: {warmup_chars}")

  timings = []
  for i in range(repeats):
      start = time.perf_counter()
      result = converter.convert(pdf_path)
      elapsed = time.perf_counter() - start
      markdown = result.document.export_to_markdown()
      timings.append(elapsed)
      print(f"Run {i + 1}: {elapsed:.2f}s | Markdown chars: {len(markdown)}")

  return {
      "mode": name,
      "init_seconds": round(init_seconds, 2),
      "runs_seconds": [round(t, 2) for t in timings],
      "median_seconds": round(statistics.median(timings), 2),
      "markdown_chars": len(markdown),
  }

gpu_result = benchmark("OCR ONNX CUDA DEFAULT", True)
cpu_result = benchmark("OCR ONNX CPU", False)

print("\n===== Comparison =====")
print("GPU:", gpu_result)
print("CPU:", cpu_result)

gpu_time = gpu_result["median_seconds"]
cpu_time = cpu_result["median_seconds"]
print(f"GPU time: {gpu_time}")
print(f"CPU time: {cpu_time}")
print(f"\nCPU/GPU time ratio: {cpu_time / gpu_time:.2f}x")
print(f"GPU time reduction: {(cpu_time - gpu_time) / cpu_time * 100:.1f}%")