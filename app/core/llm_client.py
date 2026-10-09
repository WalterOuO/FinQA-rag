import logging
import time
from typing import Iterator, Optional

from config import settings
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class LLMClient:
  def __init__(self):
    logger.info(
      f"Initializing LLM Engine: {settings.LLM_MODEL_NAME}"
    )
    self._init_engine()

  def _init_engine(self):
    self.engine = ChatOpenAI(
      openai_api_base=f"{settings.VLLM_URL}/v1",
      model=settings.LLM_MODEL_NAME,
      temperature=0.2,
      max_tokens=1024,
      openai_api_key="vllm-free-token-bypass"
    )

    logger.info("🔥 Self-hosted vLLM Engine connected via OpenAI-compatible API.")

  def generate(self, prompt: str) -> dict:
    """
    Non-streaming inference.

    Returns:
      {
        "content": str,
        "token_usage": {...},
        "performance": {...}
      }
    """
    start_time = time.perf_counter()

    try:
      response = self.engine.invoke(prompt)

      end_time = time.perf_counter()

      content = (response.content if hasattr(response, "content") else str(response))

      token_usage = self._extract_token_usage(response)

      performance = self._build_performance_metrics(
        start_time=start_time,
        first_token_time=None,
        end_time=end_time,
        token_usage=token_usage
      )

      return {
        "content": content,
        "token_usage": token_usage,
        "performance": performance
      }

    except Exception as e:
      logger.error(f"LLM inference failed: {str(e)}", exc_info=True)
      raise RuntimeError(f"LLM inference failed: {str(e)}") from e

  def stream(self, prompt: str) -> Iterator[dict]:
    """
    Streaming inference.

    Each yielded item:
      {
        "token": str,
        "is_first_token": bool,
        "timestamp": float
      }

    最後一個 chunk 會包含完整 metrics。
    """
    start_time = time.perf_counter()
    first_token_time: Optional[float] = None
    full_content = []
    token_usage = {}

    try:
      for chunk in self.engine.stream(prompt):
        timestamp = time.perf_counter()

        token = (chunk.content if hasattr(chunk, "content") else str(chunk))

        if token and first_token_time is None:
          first_token_time = timestamp

        if token:
          full_content.append(token)

        yield {
          "token": token,
          "is_first_token": (first_token_time is not None and 
                    timestamp == first_token_time),
          "timestamp": timestamp
        }

      end_time = time.perf_counter()

      performance = self._build_performance_metrics(
        start_time=start_time,
        first_token_time=first_token_time,
        end_time=end_time,
        token_usage=token_usage
      )

      yield {
        "token": "",
        "is_first_token": False,
        "timestamp": end_time,
        "final": True,
        "content": "".join(full_content),
        "token_usage": token_usage,
        "performance": performance
      }

    except Exception as e:
      logger.error(f"LLM streaming inference failed: {str(e)}", exc_info=True)
      raise RuntimeError(f"LLM streaming inference failed: {str(e)}") from e

  def _extract_token_usage(self, response) -> dict:
    usage = getattr(response, "usage_metadata", None) or {}

    if not usage:
      metadata = getattr(response, "response_metadata", {}) or {}
      usage = metadata.get("token_usage", {}) or {}

    prompt_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
    completion_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    total_tokens = usage.get("total_tokens")

    if (
      prompt_tokens is None or 
      completion_tokens is None or 
      total_tokens is None
    ):
      raise RuntimeError("LLM response does not contain complete token usage.")

    return {
      "prompt_tokens": int(prompt_tokens),
      "completion_tokens": int(completion_tokens),
      "total_tokens": int(total_tokens)
    }

  def _build_performance_metrics(
    self,
    start_time: float,
    first_token_time: Optional[float],
    end_time: float,
    token_usage: dict
  ) -> dict:
    total_latency = end_time - start_time

    ttft = None
    tpot = None

    if first_token_time is not None:
      ttft = first_token_time - start_time

      completion_tokens = token_usage.get("completion_tokens", 0)

      if completion_tokens > 1:
        generation_time = end_time - first_token_time
        tpot = generation_time / (completion_tokens - 1)

    return {
      "total_latency_seconds": round(total_latency, 6),
      "ttft_seconds": (round(ttft, 6) if ttft is not None else None),
      "tpot_seconds": (round(tpot, 6) if tpot is not None else None)
    }


llm_client = LLMClient()