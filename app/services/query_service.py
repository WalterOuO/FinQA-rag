import logging
from pathlib import Path
import json
import asyncio

from langchain_core.documents import Document
from qdrant_client import models

from config import settings
from db.vector_client import get_vector_client
from core.rerank_client import rerank_client
from core.llm_client import llm_client
from services.prompts import RAG_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# ============================================================
# Hybrid Search Fusion Strategy
# ============================================================

# Qdrant RRF 使用 RRF 預設的 rank constant = 60
# (根據 dense, parse 各自排名重新計算總分)
# k=60 時，混合檢索的 Recall 與 Acc 達到了最完美的平衡點
fusion_strategy = models.FusionQuery(
  fusion=models.Fusion.RRF
)

# 混合策略 DBSF (根據分佈投影到0-1區間再加權相加): 
# fusion_strategy = models.FusionQuery(
#   fusion=models.Fusion.DBSF
# )


class QueryService:
  def _read_parent_json(self, path: Path) -> dict:
    """標準同步讀檔函式，供執行緒池調用"""
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f)

  def _build_category_filter(self, category: str) -> models.Filter:
    """建立 Qdrant metadata category filter"""
    return models.Filter(
      must=[
        models.FieldCondition(
          key="metadata.category",
          match=models.MatchValue(value=category)
        )
      ]
    )

  async def answer_question(self, question: str, category: str) -> dict:
    """
    微服務: Advanced RAG 線上即時問答業務調度大腦
    """

    # ====== Step 2: Hybrid Dense + Sparse 向量檢索 ======
    search_filter = self._build_category_filter(category)

    hybrid_top_docs = get_vector_client().db.similarity_search(
      query=question,
      k=settings.HYBRID_SEARCH_TOP_K,
      filter=search_filter,
      hybrid_fusion=fusion_strategy
    )

    if not hybrid_top_docs:
      return {
        "answer": "抱歉，在目前的資料庫中找不到與您問題相關的參考文件。請先上傳參考文件後再行提問。",
        "sources": [],
        "context": "",
        "token_usage": None,
        "performance": None
      }

    logger.info(f"🔎 [Step 2] 完成 Hybrid Search, 取得 {len(hybrid_top_docs)} 個候選 chunks.")

    # # ====== Step 3: 子文檔去重 & 蒐集 page_num ======
    # parent_page_map = {}
    # for doc in hybrid_top_docs:
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
    # logger.info(
    #   f"🔀 [Step 3] De-duplicated {len(hybrid_top_docs)} "
    #   f"child chunks into {len(unique_parent_docs)} unique parents."
    # )

    # # ====== Step 4: 回查父文檔 (Parent Retrieval) ======
    # parent_documents = []
    # loaded_parent_files = {}

    # for p_doc in unique_parent_docs:
    #   p_id = p_doc["parent_id"]
    #   f_name = p_doc["file_name"]
    #   page_num_list = p_doc["page_num"]

    #   pdf_stem = Path(f_name).stem
    #   parent_json_path = (
    #     settings.PARENT_CHUNKS_DIR
    #     / category
    #     / f"{pdf_stem}_parents.json"
    #   )

    #   # 將同步讀檔丟給執行緒池，避免阻塞 Event Loop
    #   if parent_json_path.exists():
    #     if str(parent_json_path) not in loaded_parent_files:
    #       loaded_parent_files[str(parent_json_path)] = (
    #         await asyncio.to_thread(
    #           self._read_parent_json,
    #           parent_json_path
    #         )
    #       )

    #     p_data = loaded_parent_files[
    #       str(parent_json_path)
    #     ].get(p_id)

    #     if p_data:
    #       p_metadata = p_data["metadata"].copy()
    #       p_metadata["page_num_list"] = page_num_list

    #       parent_documents.append(
    #         Document(
    #           page_content=p_data["page_content"],
    #           metadata=p_metadata
    #         )
    #       )

    # if not parent_documents:
    #   return {
    #     "answer": "抱歉，資料庫中無法尋獲對應的完整文檔架構。請重新上傳補充文件。",
    #     "sources": [],
    #     "context": "",
    #     "token_usage": None,
    #     "performance": None
    #   }

    # logger.info(
    #   f"📂 [Step 4] 成功回查父文檔. "
    #   f"Loaded {len(parent_documents)} parent documents non-blockingly."
    # )

    # # ====== Step 5: Reranker 交叉深度重新評分 ======
    # top_k_num = settings.RERANK_OUT_TOP_K

    # reranked_top_docs = rerank_client.rerank(
    #   query=question,
    #   documents=parent_documents,
    #   top_k=top_k_num
    # )

    # logger.info(
    #   f"⚖️ [Step 5] 完成 CrossEncoder Reranking. "
    #   f"選取 top {top_k_num} 文件."
    # )

    # ====== Step 6: 建立問答 Prompt 與 LLM 生成 ======
    context_str = ""
    sources = []

    for idx, doc in enumerate(hybrid_top_docs):
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
      f"🤖 [Step 6] LLM 已生成回答. "
      f"Token usage: prompt={token_usage['prompt_tokens']}, "
      f"completion={token_usage['completion_tokens']}, "
      f"total={token_usage['total_tokens']}, "
      f"latency={performance['total_latency_seconds']:.4f}s"
    )

    return {
      "answer": llm_result["content"],
      "sources": sources,
      "context": context_str,
      "token_usage": token_usage,
      "performance": performance
    }


# 全域單例
query_service = QueryService()