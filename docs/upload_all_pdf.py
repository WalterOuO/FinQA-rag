import requests
from pathlib import Path

API_URL = "http://127.0.0.1:8001/upload"

BASE_DIR = Path("/content/drive/MyDrive/FinQA-rag/docs")

CATEGORIES = {
  "finance": BASE_DIR / "finance",
  "military": BASE_DIR / "military"
}


def upload_pdf(pdf_path: Path, category: str):
  with open(pdf_path, "rb") as f:
    files = {"file": (pdf_path.name, f, "application/pdf")}
    data = {"category": category}

    response = requests.post(
      API_URL,
      files=files,
      data=data,
      timeout=300
    )

  response.raise_for_status()

  return response.json()


def main():
  total = 0
  success = 0
  failed = 0

  print("=" * 80)
  print("Starting batch PDF ingestion")
  print("=" * 80)

  for category, directory in CATEGORIES.items():
    pdf_files = sorted(directory.glob("*.pdf"))

    print()
    print(f"[{category.upper()}] Found {len(pdf_files)} PDF files")

    for pdf_path in pdf_files:
      total += 1

      print(f"[{total}] Uploading: {pdf_path.name}")

      try:
        result = upload_pdf(pdf_path, category)
        success += 1

        print(
          f"    ✓ Uploaded | "
          f"Task ID: {result.get('task_id')} | "
          f"Status: {result.get('status')}"
        )

      except Exception as e:
        failed += 1
        print(f"    ✗ Failed: {e}")

  print()
  print("=" * 80)
  print("BATCH INGESTION SUBMITTED")
  print("=" * 80)
  print(f"Total   : {total}")
  print(f"Success : {success}")
  print(f"Failed  : {failed}")


if __name__ == "__main__":
  main()