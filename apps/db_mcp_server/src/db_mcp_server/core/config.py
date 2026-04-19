from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import create_engine, text

_ROOT = Path(__file__).resolve().parents[4]


class Settings(BaseSettings):
    OPENAI_API_KEY: str
    PG_URL: str = (
        "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
    )
    model_config = SettingsConfigDict(
        env_file=(_ROOT / ".env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )


settings = Settings()
