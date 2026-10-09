import json
import time
import requests
import argparse
from pathlib import Path

API_URL = "http://127.0.0.1:8001/query"

DATASET_FILE = Path("evaluation/evaluation_dataset.json")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--experiment",
    required=True,
    choices=["E0", "E1", "E2", "E3"]
  )
  args = parser.parse_args()
  
  output_file = Path(f"evaluation/evaluation_answers/{args.experiment}_answers.json")


  with open(DATASET_FILE, "r", encoding="utf-8") as f:
    evaluation_dataset = json.load(f)

  results = []

  for item in evaluation_dataset:
    question_id = item["id"]
    question = item["quetsion"]
    category = item["category"]
    ground_truth = item["ground_truth"]
    source_file = item["source_file"]

    print("=" * 80)
    print(f"[Question {question_id}]")
    print(f"Question : {question}")
    print(f"Category : {category}")

    payload = {
      "question": question,
      "category": category
    }

    try:
      start_time = time.perf_counter()
      response = requests.post(API_URL, json=payload, timeout=300)
      request_latency = time.perf_counter() - start_time

      response.raise_for_status()
      data = response.json()

      # ====== Token Usage Validation ======
      token_usage = data.get("token_usage")

      if not token_usage:
        raise ValueError("API response missing 'token_usage'.")

      required_tokens = [
        "prompt_tokens",
        "completion_tokens",
        "total_tokens"
      ]

      missing_tokens = [
        key for key in required_tokens
        if key not in token_usage or token_usage[key] is None
      ]

      if missing_tokens:
        raise ValueError(f"API response missing token fields: {missing_tokens}")

      prompt_tokens = int(token_usage["prompt_tokens"])
      completion_tokens = int(token_usage["completion_tokens"])
      total_tokens = int(token_usage["total_tokens"])

      # ====== Inference Performance Validation ======
      performance = data.get("performance")

      if not performance:
        raise ValueError("API response missing 'performance'.")

      if performance.get("total_latency_seconds") is None:
        raise ValueError("API response missing 'performance.total_latency_seconds'.")

    except Exception as e:
      print(f"[ERROR] RAG request failed: {e}")

      results.append({
        "id": question_id,
        "quetsion": question,
        "category": category,
        "ground_truth": ground_truth,
        "source_file": source_file,
        "error": str(e)
      })

      continue

    generated_answer = data.get("answer", "")
    sources = data.get("sources", [])
    context = data.get("context", "")

    result = {
      "id": question_id,
      "quetsion": question,
      "category": category,
      "ground_truth": ground_truth,
      "source_file": source_file,
      "generated_answer": generated_answer,
      "retrieved_sources": sources,
      "context": context,

      # API request 整體延遲
      "request_latency_seconds": round(request_latency, 4),

      # LLM inference performance
      "performance": {
        "total_latency_seconds": performance["total_latency_seconds"],
        "ttft_seconds": performance.get("ttft_seconds"),
        "tpot_seconds": performance.get("tpot_seconds")
      },
      # LLM token usage
      "token_usage": {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens
      }
    }

    results.append(result)

    print(f"Request Latency  : {request_latency:.4f}s")
    print(f"LLM Latency    : {performance['total_latency_seconds']:.4f}s")
    print(f"TTFT        : {performance.get('ttft_seconds')}")
    print(f"TPOT        : {performance.get('tpot_seconds')}")
    print(f"Prompt Tokens   : {prompt_tokens}")
    print(f"Completion Tokens : {completion_tokens}")
    print(f"Total Tokens   : {total_tokens}")
    print(f"Answer       : {generated_answer}")

  output_file.parent.mkdir(parents=True, exist_ok=True)

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

  print("\n" + "=" * 80)
  print("RAG GENERATION COMPLETED")
  print("=" * 80)
  print(f"Experiment      : {args.experiment}")
  print(f"Total Questions : {len(results)}")
  print(f"Output          : {output_file}")


if __name__ == "__main__":
  main()