from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── OpenAI（唯一 LLM 提供商）─────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # ── ChromaDB（本地持久化，无需额外服务）──────────────────
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection: str = "stock_filings"

    # ── Agent 参数 ─────────────────────────────────────────────
    rag_top_k: int = 5

    # ── LangSmith（可选，用于 Agent 执行轨迹可视化）──────────
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "stock-advisor"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    # ── Redis（LangGraph Checkpointer，可选）────────────────────
    redis_url: str = ""

    # ── PostgreSQL（历史报告持久化，可选）──────────────────────
    database_url: str = ""

    # ── CORS ──────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
