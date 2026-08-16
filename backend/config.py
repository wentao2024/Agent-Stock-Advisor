from functools import lru_cache
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ── OpenAI (sole LLM provider) ────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o"

    # ── LangSmith (optional, for agent trace visualization) ───
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "stock-advisor"
    langchain_endpoint: str = "https://api.smith.langchain.com"

    
    # ── Redis (LangGraph Checkpointer, optional) ──────────────
    redis_url: str = ""
    

    # ── PostgreSQL (historical report persistence, optional) ──
    database_url: str = ""

    
    # ── CORS ──────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
