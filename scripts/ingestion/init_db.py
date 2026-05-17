"""Run init_db.sql against Postgres to create the maintenance schema and tables.

Usage:
    uv run python scripts/ingestion/init_db.py
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_URL = (
    os.getenv("POSTGRES_URL")
    or os.getenv("postgres_url")
    or os.getenv("DATABASE_URL")
    or "postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
)
SQL_FILE = Path(__file__).resolve().parent / "init_db.sql"


def main() -> None:
    engine = create_engine(DB_URL)
    print("Running init_db.sql ...")
    with engine.connect() as conn:
        conn.execute(text(SQL_FILE.read_text()))
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
