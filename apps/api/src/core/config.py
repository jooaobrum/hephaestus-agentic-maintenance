from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text

_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    CO_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    CM_QDRANT_COLLECTION: str = "cm_interventions_hybrid"
    PROC_QDRANT_COLLECTION: str = "procedures_hybrid"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    KEYWORD_MODEL: str = "bm25"
    GENERATION_MODEL: str = "gpt-4.1-mini"
    RERANKING_MODEL: str = "rerank-v4.0-pro"
    EVALUATION_MODEL: str = "gpt-5.4-mini"
    DATASET_NAME: str = "rag-evaluation-dataset"
    TOP_N: int = 20
    TOP_K: int = 10
    PROMPTS_PATH: Path = (
        Path(__file__).parent.parent / "agents" / "prompts" / "retrieval_generation.yml"
    )
    PG_URL: str = (
        "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
    )
    model_config = SettingsConfigDict(
        env_file=(_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )


config = Settings()
