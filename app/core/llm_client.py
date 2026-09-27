import logging
from config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self):
        logger.info(f"Initializing global LLM Chat Engine: {settings.LLM_MODEL_NAME}")
        self._init_engine()

    def _init_engine(self):
        """根據連線端點自動辨識要串接 vLLM (OpenAI 協議) 還是地端 Ollama"""
        if "ollama" in settings.OLLAMA_URL:
            # 💻 地端本機 Docker 環境：使用最新官方特化的 ChatOllama 封裝對話模型
            from langchain_ollama import ChatOllama
            self.engine = ChatOllama(
                base_url=settings.OLLAMA_URL,
                model=settings.LLM_MODEL_NAME,
                temperature=0.2
            )
            logger.info("🟢 Local Ollama Chat Engine container bridge linked.")
        else:
            # ☁️ 雲端 Colab 環境：使用業界最常用的 ChatOpenAI 串接自託管 vLLM 引擎（100% 免費本地運算）
            from langchain_openai import ChatOpenAI
            self.engine = ChatOpenAI(
                # vLLM 預設的 OpenAI 相容端點路徑（例如：http://localhost:8000/v1）
                openai_api_base=f"{settings.VLLM_URL}/v1", 
                model=settings.LLM_MODEL_NAME,
                temperature=0.2,
                max_tokens=1024,  # 升級為 8B 模型後，建議將最大生成長度放大至 1024，確保財報長分析不被切斷
                openai_api_key="vllm-free-token-bypass"  # 💡 隨意填寫即可，vLLM 底層會自動跳過驗證
            )
            logger.info(f"🔥 Self-hosted vLLM Engine linked via OpenAI compatible protocol.")

    def generate(self, prompt: str) -> str:
        """全域統一的 Chat LLM 生成接口，完美對齊最新 LangChain 訊息回傳格式"""
        try:
            # 呼叫 invoke 會回傳 ChatMessage 物件 (如 AIMessage)
            response = self.engine.invoke(prompt)
            
            # ChatOpenAI 與 ChatOllama 的回傳文字皆存放在 .content 屬性中
            if hasattr(response, "content"):
                return response.content
            return str(response)
            
        except Exception as e:
            logger.error(f"Error during LLM Chat Inference: {str(e)}", exc_info=True)
            return f"❌ 系統後端 LLM 引擎生成失敗: {str(e)}"

# 全域單例：服務啟動時只會定義一次並常駐記憶體
llm_client = LLMClient()
