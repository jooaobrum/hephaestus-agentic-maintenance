from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION: str = "cm_interventions"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    GENERATION_MODEL: str = "gpt-5.4-nano"
    EVALUATION_MODEL: str = "gpt-5.4-mini"
    DATASET_NAME: str = "rag-evaluation-dataset"

    model_config = SettingsConfigDict(
        env_file=(_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )


config = Settings()
