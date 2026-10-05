import logging
from pathlib import Path
import json
import asyncio
from langchain_core.documents import Document

# 引入微服務全域變數與模型常駐單例客戶端
from config import settings
from db.vector_client import get_vector_client
from core.rerank_client import rerank_client
from core.llm_client import llm_client
# from core.cache_manager import cache_manager
from services.prompts import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


class QueryService:
  def _read_parent_json(self, path: Path) -> dict:
    """標準同步讀檔函式，供執行緒池調用"""
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)

  async def answer_question(self, question: str, category: str) -> dict:
    """
    微服務: Advanced RAG 線上即時問答業務調度大腦
    """
    # # ====== Step 1: Redis 語意快取檢查 ======
    # cached_answer = cache_manager.check_semantic_cache(question, category, threshold=settings.REDIS_SEMANTIC_CACHE_THRESHOLD)
    # if cached_answer:
    #   cached_answer["cached"]=True
    #   return cached_answer

    # logger.info(f"🔍 [Step 1] Cache Missed. Starting Dense Retrieval for [{category}]")

    # ====== Step 2: Dense 向量檢索 ======
    search_filter = {"category": category}
    dense_top_docs = get_vector_client().db.similarity_search(
      query=question,
      k=settings.HYBRID_SEARCH_TOP_K,
      filter=search_filter
    )

    if not dense_top_docs:
      return {
        "answer": "抱歉，在目前的資料庫中找不到與您問題相關的參考文件。請先上傳參考文件後再行提問。",
        "sources": [],
        "context": "",
        "token_usage": None,
        "performance": None
      }

    logger.info(f"🧲 [Step 2] Dense Vector Search completed. Retrieved {len(dense_top_docs)} candidate chunks.")

    # # ====== Step 3: 子文檔去重 & 蒐集 page_num ======
    # parent_page_map = {}
    # for doc in dense_top_docs:
    #   p_id = doc.metadata.get("parent_id")
    #   page_num = doc.metadata.get("page_num")
    #   if not p_id:
    #     continue

    #   if p_id not in parent_page_map:
    #     parent_page_map[p_id] = {
    #       "file_name": doc.metadata.get("file_name"),
    #       "page_num": set()
    #     }
    #   if page_num is not None:
    #     parent_page_map[p_id]["page_num"].add(page_num)

    # unique_parent_docs = []
    # for p_id, info in parent_page_map.items():
    #   unique_parent_docs.append({
    #     "parent_id": p_id,
    #     "file_name": info["file_name"],
    #     "page_num": sorted(info["page_num"])
    #   })
    # logger.info(f"🔀 [Step 3] De-duplicated {len(dense_top_docs)} child chunks into {len(unique_parent_docs)} unique parents.")

    # # ====== Step 4: 回查父文檔 (Parent Retrieval) ======
    # parent_documents = []
    # loaded_parent_files = {}

    # for p_doc in unique_parent_docs:
    #   p_id = p_doc["parent_id"]
    #   f_name = p_doc["file_name"]
    #   page_num_list = p_doc["page_num"]

    #   pdf_stem = Path(f_name).stem
    #   parent_json_path = settings.PARENT_CHUNKS_DIR / category / f"{pdf_stem}_parents.json"

    #   # 💡 2：將原本會卡死 Event Loop 的同步讀檔，丟給執行緒池（to_thread）非同步處理！
    #   if parent_json_path.exists():
    #     if str(parent_json_path) not in loaded_parent_files:
    #       loaded_parent_files[str(parent_json_path)] = await asyncio.to_thread(
    #         self._read_parent_json, parent_json_path
    #       )

    #     p_data = loaded_parent_files[str(parent_json_path)].get(p_id)
    #     if p_data:
    #       p_metadata = p_data["metadata"].copy()
    #       p_metadata["page_num_list"] = page_num_list

    #       parent_documents.append(Document(
    #         page_content=p_data["page_content"],
    #         metadata=p_metadata
    #       ))

    # if not parent_documents:
    #   return {
    #     "answer": "抱歉，資料庫中無法尋獲對應的完整文檔架構。請重新上傳補充文件。",
    #     "sources": [],
    #     "context": "",
    #     "token_usage": None,
    #     "performance": None
    #   }
    # logger.info(f"📂 [Step 4] Parent Retrieval success. Loaded {len(parent_documents)} parent documents non-blockingly.")
    # logger.info(f"🔎 Parent metadata sample: {parent_documents[0].metadata}")

    # # ====== Step 5: Reranker 交叉深度重新評分 (動態讀取 Top K) ======
    # top_k_num = settings.RERANK_OUT_TOP_K
    # reranked_top_docs = rerank_client.rerank(
    #     query=question,
    #     documents=parent_documents,
    #     top_k=top_k_num
    # )
    # logger.info(f"⚖️ [Step 5] CrossEncoder Reranking finished. Selected top {top_k_num} contexts.")

    # ====== Step 6: 建立問答 Prompt 與 LLM 生成 ======
    context_str = ""
    sources = []

    for idx, doc in enumerate(dense_top_docs):
      src_file = doc.metadata.get("file_name", "未知文件")

      context_str += (
        f"===== CONTEXT {idx + 1} =====\n"
        f"來源文件: {src_file}\n"
        f"文件內容: \n{doc.page_content}\n"
        f"===== END CONTEXT {idx + 1} =====\n\n"
      )

      sources.append({
        "frontend_index": idx + 1,
        "file_name": src_file,
        "category": category
      })

    rag_prompt = RAG_PROMPT_TEMPLATE.format(
      context_str=context_str,
      question=question
    )

    llm_result = llm_client.generate(rag_prompt)

    token_usage = llm_result["token_usage"]
    performance = llm_result["performance"]

    logger.info(
      f"🤖 [Step 6] LLM Text Generation completed. "
      f"Token usage: prompt={token_usage['prompt_tokens']}, "
      f"completion={token_usage['completion_tokens']}, "
      f"total={token_usage['total_tokens']}, "
      f"latency={performance['total_latency_seconds']:.4f}s"
    )

    final_response = {
      "answer": llm_result["content"],
      "sources": sources,
      "context": context_str,
      "token_usage": token_usage,
      "performance": performance
    }

    return final_response


# 全域單例
query_service = QueryService()