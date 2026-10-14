import json
import argparse
from pathlib import Path
from collections import defaultdict
from langchain_openai import ChatOpenAI
from evaluate_utils import calculate_reciprocal_rank, calculate_hit_at_k, parse_judge_response

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/content/drive/MyDrive/FinQA-rag")
EVALUATION_DIR = BASE_DIR / "evaluation"
PROMPT_FILE = EVALUATION_DIR / "judge_prompt.txt"

VLLM_URL = "http://127.0.0.1:8000"
JUDGE_MODEL = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a8"
TOP_K = 3


# ============================================================
# LLM Judge
# ============================================================

def load_judge_prompt():
  with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    return f.read()


def judge_answer(client, prompt_template, question, ground_truth, generated_answer, context):
  prompt = prompt_template.format(
    question=question,
    ground_truth=ground_truth,
    generated_answer=generated_answer,
    context=context
  )
  response = client.invoke(prompt)
  return parse_judge_response(response.content)


# ============================================================
# Metrics Summary
# ============================================================

def calculate_summary(results):
  count = len(results)

  def average(key):
    values = [item[key] for item in results if item.get(key) is not None]
    return round(sum(values) / len(values), 4) if values else None

  def average_token(key):
    values = [
      item["token_usage"][key]
      for item in results
      if item.get("token_usage") and item["token_usage"].get(key) is not None
    ]
    return round(sum(values) / len(values), 2) if values else None

  return {
    "total_questions": count,
    "MRR": average("reciprocal_rank"),
    f"Hit@{TOP_K}": average(f"hit_at_{TOP_K}"),
    "average_answer_correctness": average("answer_correctness"),
    "average_faithfulness": average("faithfulness"),
    "average_request_latency_seconds": average("request_latency_seconds"),
    "average_inference_latency_seconds": average("inference_latency_seconds"),
    "average_token_usage": {
      "prompt_tokens": average_token("prompt_tokens"),
      "completion_tokens": average_token("completion_tokens"),
      "total_tokens": average_token("total_tokens")
    }
  }


# ============================================================
# Main
# ============================================================

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--experiment",
    required=True,
    choices=["E0", "E1", "E2", "E3"]
  )
  args = parser.parse_args()

  input_file = EVALUATION_DIR / "evaluation_answers" / f"{args.experiment}_answers.json"

  with open(input_file, "r", encoding="utf-8") as f:
    evaluation_answers = json.load(f)

  prompt_template = load_judge_prompt()

  # Judge vLLM must be started externally
  judge_llm = ChatOpenAI(
    openai_api_base=f"{VLLM_URL}/v1",
    model=JUDGE_MODEL,
    temperature=0,
    max_tokens=300,
    openai_api_key="vllm-free-token-bypass"
  )

  # Group evaluated results by category
  category_results = defaultdict(list)

  print("=" * 80)
  print("Starting Evaluation with existing Judge vLLM...")
  print("=" * 80)

  for item in evaluation_answers:
    if "error" in item:
      print(f"[{item['id']}] Skipped because generation failed: {item['error']}")
      continue

    question_id = item["id"]
    question = item["quetsion"]
    category = item["category"]
    ground_truth = item["ground_truth"]
    ground_truth_source = item["ground_truth_source"]
    generated_answer = item["generated_answer"]
    retrieved_sources = item["retrieved_sources"]
    context = item["context"]

    request_latency = item["request_latency_seconds"]
    performance = item.get("performance")

    if not performance or performance.get("total_latency_seconds") is None:
      raise ValueError(f"Question {question_id}: missing inference performance.")

    inference_latency = performance["total_latency_seconds"]
    token_usage = item.get("token_usage")

    if not token_usage:
      raise ValueError(f"Question {question_id}: missing token usage.")

    # Retrieval metrics
    reciprocal_rank = calculate_reciprocal_rank(retrieved_sources, ground_truth_source)
    hit_at_k = calculate_hit_at_k(retrieved_sources, ground_truth_source, TOP_K)

    # LLM Judge
    judge_result = judge_answer(
      judge_llm, prompt_template, question,
      ground_truth, generated_answer, context
    )

    answer_correctness = judge_result["answer_correctness"]
    faithfulness = judge_result["faithfulness"]

    result = {
      "id": question_id,
      "quetsion": question,
      "category": category,
      "ground_truth": ground_truth,
      "ground_truth_source": ground_truth_source,
      "generated_answer": generated_answer,
      "retrieved_sources": retrieved_sources,
      "reciprocal_rank": round(reciprocal_rank, 4),
      f"hit_at_{TOP_K}": hit_at_k,
      "answer_correctness": answer_correctness,
      "faithfulness": faithfulness,
      "judge_reason": judge_result.get("reason", ""),
      "request_latency_seconds": round(request_latency, 4),
      "inference_latency_seconds": inference_latency,
      "performance": {
        "total_latency_seconds": performance.get("total_latency_seconds"),
        "ttft_seconds": performance.get("ttft_seconds"),
        "tpot_seconds": performance.get("tpot_seconds")
      },
      "token_usage": {
        "prompt_tokens": token_usage["prompt_tokens"],
        "completion_tokens": token_usage["completion_tokens"],
        "total_tokens": token_usage["total_tokens"]
      }
    }

    category_results[category].append(result)

    print(
      f"[{question_id}] Category={category} | "
      f"MRR={reciprocal_rank:.4f} | Hit@{TOP_K}={hit_at_k} | "
      f"Correctness={answer_correctness}/5 | Faithfulness={faithfulness}/5"
    )

  # ========================================================
  # Build category-level output
  # ========================================================

  output = {}

  for category, results in category_results.items():
    output[category] = {
      "summary": calculate_summary(results),
      "results": results
    }

  output_file = EVALUATION_DIR / "evaluation_results" / f"{args.experiment}_results.json"
  output_file.parent.mkdir(parents=True, exist_ok=True)

  with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

  print()
  print("=" * 80)
  print("EVALUATION COMPLETED")
  print("=" * 80)
  print(f"Experiment: {args.experiment}")
  print(json.dumps(
    {category: data["summary"] for category, data in output.items()},
    ensure_ascii=False,
    indent=2
  ))
  print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
  main()