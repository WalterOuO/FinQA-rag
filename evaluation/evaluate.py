import json
from pathlib import Path
from langchain_openai import ChatOpenAI
from evaluate_utils import calculate_reciprocal_rank, calculate_hit_at_k, parse_judge_response

# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path("/content/drive/MyDrive/FinQA-rag")
EVALUATION_DIR = BASE_DIR / "evaluation"

INPUT_FILE = EVALUATION_DIR / "evaluation_answers.json"
OUTPUT_FILE = EVALUATION_DIR / "evaluation_results.json"
PROMPT_FILE = EVALUATION_DIR / "judge_prompt.txt"

VLLM_URL = "http://127.0.0.1:8000"
JUDGE_MODEL = "Llama-3.1-8B-Instruct-GPTQ-Int8"

TOP_K = 3


# ============================================================
# LLM Judge
# ============================================================

def load_judge_prompt():
  with open(PROMPT_FILE, "r", encoding="utf-8") as f:
    return f.read()


def judge_answer(
    client,
    prompt_template,
    question,
    ground_truth,
    generated_answer,
    context
  ):
  prompt = prompt_template.format(
    question=question,
    ground_truth=ground_truth,
    generated_answer=generated_answer,
    context=context
  )

  response = client.invoke(prompt)

  return parse_judge_response(response.content)


# ============================================================
# Main
# ============================================================

def main():
  with open(INPUT_FILE, "r", encoding="utf-8") as f:
    evaluation_answers = json.load(f)

  prompt_template = load_judge_prompt()

  # Judge vLLM 必須由外部先行啟動
  judge_llm = ChatOpenAI(
    openai_api_base=f"{VLLM_URL}/v1",
    model=JUDGE_MODEL,
    temperature=0,
    max_tokens=300,
    openai_api_key="vllm-free-token-bypass"
  )

  results = []

  reciprocal_ranks = []
  hit_results = []
  answer_correctness_scores = []
  faithfulness_scores = []
  request_latencies = []
  inference_latencies = []

  print("=" * 80)
  print("Starting Evaluation with existing Judge vLLM...")
  print("=" * 80)
  print()

  for item in evaluation_answers:
    if "error" in item:
      print(
        f"[{item['id']}] Skipped because generation failed: "
        f"{item['error']}"
      )
      continue

    question_id = item["id"]
    question = item["quetsion"]
    category = item["category"]
    ground_truth = item["ground_truth"]
    source_file = item["source_file"]
    generated_answer = item["generated_answer"]
    sources = item["retrieved_sources"]
    context = item["context"]

    request_latency = item["request_latency_seconds"]

    performance = item.get("performance")

    if not performance:
      raise ValueError(f"Question {question_id}: missing 'performance'.")

    inference_latency = performance.get("total_latency_seconds")
    if inference_latency is None:
      raise ValueError(f"Question {question_id}: missing 'performance.total_latency_seconds'.")

    token_usage = item.get("token_usage")

    if not token_usage:
      raise ValueError(f"Question {question_id}: missing 'token_usage'.")

    # ========================================================
    # Retrieval
    # ========================================================

    reciprocal_rank = calculate_reciprocal_rank(sources, source_file)

    hit_at_k = calculate_hit_at_k(sources, source_file, TOP_K)

    # ========================================================
    # LLM Judge
    # ========================================================

    judge_result = judge_answer(
      judge_llm,
      prompt_template,
      question,
      ground_truth,
      generated_answer,
      context
    )

    answer_correctness = judge_result["answer_correctness"]
    faithfulness = judge_result["faithfulness"]
    judge_reason = judge_result.get("reason", "")

    # ========================================================
    # Collect Metrics
    # ========================================================

    reciprocal_ranks.append(reciprocal_rank)
    hit_results.append(hit_at_k)
    answer_correctness_scores.append(answer_correctness)
    faithfulness_scores.append(faithfulness)
    request_latencies.append(request_latency)
    inference_latencies.append(inference_latency)

    # ========================================================
    # Store Result
    # ========================================================

    results.append({
      "id": question_id,
      "quetsion": question,
      "category": category,
      "ground_truth": ground_truth,
      "source_file": source_file,
      "generated_answer": generated_answer,
      "retrieved_sources": sources,

      "reciprocal_rank": round(reciprocal_rank, 4),

      f"hit_at_{TOP_K}": hit_at_k,

      "answer_correctness": answer_correctness,
      "faithfulness": faithfulness,
      "judge_reason": judge_reason,

      "request_latency_seconds": round(request_latency, 4),

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
    })

    print(
      f"[{question_id}] "
      f"MRR={reciprocal_rank:.4f} | "
      f"Hit@{TOP_K}={hit_at_k} | "
      f"Correctness={answer_correctness}/5 | "
      f"Faithfulness={faithfulness}/5 | "
      f"Latency={inference_latency:.4f}s"
    )

  # ==========================================================
  # Summary
  # ==========================================================

  count = len(results)

  summary = {
    "total_questions": count,
    "MRR": round(sum(reciprocal_ranks) / count, 4) if count else 0,
    f"Hit@{TOP_K}": round(sum(hit_results) / count, 4) if count else 0,
    "average_answer_correctness": round(
      sum(answer_correctness_scores) / len(answer_correctness_scores), 4) if answer_correctness_scores else None,
    "average_faithfulness": round(
      sum(faithfulness_scores) / len(faithfulness_scores), 4) if faithfulness_scores else None,
    "average_request_latency_seconds": round(
      sum(request_latencies) / len(request_latencies), 4) if request_latencies else None,
    "average_inference_latency_seconds": round(
      sum(inference_latencies) / len(inference_latencies),4) if inference_latencies else None
    }

  output = {
    "summary": summary,
    "results": results
  }

  # ==========================================================
  # Save
  # ==========================================================

  OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

  with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

  print()
  print("=" * 80)
  print("EVALUATION COMPLETED")
  print("=" * 80)
  print(json.dumps(summary, ensure_ascii=False, indent=2))
  print()
  print(f"Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
  main()