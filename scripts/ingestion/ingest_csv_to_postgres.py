"""Ingest a CSV file into a PostgreSQL table.

Usage:
    uv run python scripts/ingestion/ingest_csv_to_postgres.py \\
        --csv data/interventions.csv \\
        --table interventions \\
        --schema maintenance \\
        [--chunksize 50000] \\
        [--if-exists replace|append]
"""

import argparse
import os
from pathlib import Path
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DB_URL = (
    os.getenv("POSTGRES_URL")
    or os.getenv("postgres_url")
    or os.getenv("DATABASE_URL")
    or "postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
)



def load_csv_to_table(csv_path: Path, table: str, schema: str, chunksize: int, if_exists: str) -> None:
    engine = create_engine(DB_URL)
    reader = pd.read_csv(csv_path, chunksize=chunksize)
    total = 0
    for i, chunk in enumerate(reader):
        mode = if_exists if i == 0 else "append"
        chunk.to_sql(table, engine, schema=schema, if_exists=mode, index=False)
        total += len(chunk)
        print(f"  {total:,} rows loaded ...")
    print(f"Done — {total:,} rows into {schema}.{table}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest CSV into a Postgres table.")
    parser.add_argument("--csv", required=True, type=Path, help="Path to the CSV file.")
    parser.add_argument("--table", required=True, help="Target table name.")
    parser.add_argument("--schema", default="maintenance", help="Target schema (default: maintenance).")
    parser.add_argument("--chunksize", type=int, default=50_000, help="Rows per chunk (default: 50000).")
    parser.add_argument(
        "--if-exists",
        choices=["replace", "append", "fail"],
        default="replace",
        help="Behaviour when table exists (default: replace).",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        raise FileNotFoundError(f"CSV not found: {args.csv}")

    print(f"Ingesting {args.csv} → {args.schema}.{args.table} ...")
    load_csv_to_table(args.csv, args.table, args.schema, args.chunksize, args.if_exists)


if __name__ == "__main__":
    main()
