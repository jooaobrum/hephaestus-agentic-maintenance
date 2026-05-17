"""Hephaestus Data Engineering Pipeline Orchestrator.

Provides clean, simple pipelines called as direct Python functions:
  1. Interventions Pipeline: Ensuring database creation, conditional CSV loading,
     Qdrant ingestion, and known issues clustering graph synthesis.
  2. Procedures Pipeline: OCR extraction of PDFs, chunk enrichment, and Qdrant ingestion.

Usage:
    uv run python scripts/orchestrate_pipeline.py [interventions|procedures|all] [--force-db-load]
"""

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configuration URLs from environment variables
QDRANT_URL = os.getenv("QDRANT_URL") or os.getenv("qdrant_url") or "http://localhost:6333"
POSTGRES_URL = (
    os.getenv("POSTGRES_URL")
    or os.getenv("postgres_url")
    or os.getenv("DATABASE_URL")
    or "postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
)

# Add project root to sys.path to enable clean imports of scripts
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


from scripts.ingestion.init_db import main as init_db
from scripts.ingestion.ingest_csv_to_postgres import load_csv_to_table
from scripts.ingestion.ingest_interventions_to_qdrant import ingest_interventions
from scripts.ingestion.build_known_issues_graph import build_known_issues
from scripts.ingestion.parse_procedures_pdf import parse_procedures
from scripts.ingestion.extract_procedures import extract_procedures
from scripts.ingestion.ingest_procedures import ingest_procedures


def check_table_has_data(table_name: str, schema: str = "maintenance") -> bool:
    """Check if a table exists in PostgreSQL and contains rows."""
    from sqlalchemy import create_engine, inspect, text
    try:
        engine = create_engine(POSTGRES_URL)
        inspector = inspect(engine)
        if not inspector.has_table(table_name, schema=schema):
            return False
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {schema}.{table_name}"))
            count = result.scalar()
            return count > 0
    except Exception as e:
        print(f"Database connection or table check failed: {e}")
        return False



def run_interventions_pipeline(csv_path: str = "data/interventions.csv", force_db_load: bool = False) -> None:
    """Runs the Interventions & Fault Clustering Pipeline."""
    print("\n==================================================")
    print("🚀 RUNNING INTERVENTIONS PIPELINE")
    print("==================================================")
    
    # 1. Ensure DB exists and is initialized
    print("\n[Step 1/4] Ensuring database schema and tables exist...")
    init_db()
    
    # 2. Ingest CSV to Postgres if it doesn't already have data
    print("\n[Step 2/4] Checking database load state...")
    has_data = check_table_has_data("interventions")
    if not has_data or force_db_load:
        if force_db_load:
            print("Force loading interventions to Postgres...")
        else:
            print("Interventions table is empty. Loading CSV to Postgres...")
        
        load_csv_to_table(
            csv_path=Path(csv_path),
            table="interventions",
            schema="maintenance",
            chunksize=50000,
            if_exists="replace"
        )
    else:
        print("Interventions table already has data. Skipping CSV load.")
        
    # 3. Ingest CM Interventions to Qdrant (dense + sparse hybrid)
    print("\n[Step 3/4] Ingesting interventions to Qdrant...")
    ingest_interventions(
        csv_path=Path(csv_path),
        collection="cm_interventions_hybrid",
        qdrant_url=QDRANT_URL,
        batch_size=100
    )
    
    # 4. Cluster Interventions and Build Known Issues Graph
    print("\n[Step 4/4] Building known issues graph in Qdrant...")
    build_known_issues(
        source_collection="cm_interventions_hybrid",
        output_collection="known_issues",
        qdrant_url=QDRANT_URL,
        min_cluster_size=5,
        top_k_reps=8
    )
    
    print("\n✨ Interventions Pipeline completed successfully!")


def run_procedures_pipeline(input_dir: str = "data/procedures", 
                            output_dir: str = "data/procedures_extracted",
                            chunks_csv: str = "data/procedure_chunks.csv") -> None:
    """Runs the Troubleshooting Procedures & OCR Ingestion Pipeline."""
    print("\n==================================================")
    print("🚀 RUNNING PROCEDURES PIPELINE")
    print("==================================================")
    
    # 1. Parse procedure PDFs using Mistral OCR (skips already processed)
    print("\n[Step 1/3] Parsing troubleshooting PDFs...")
    parse_procedures(
        input_dir=Path(input_dir),
        output_dir=Path(output_dir)
    )
    
    # 2. Extract semantic chunks and enrich with LLM context
    print("\n[Step 2/3] Extracting and enriching chunks to CSV...")
    extract_procedures(
        json_dir=Path(output_dir),
        output_csv=Path(chunks_csv)
    )
    
    # 3. Ingest enriched chunks to Qdrant hybrid vectors
    print("\n[Step 3/3] Ingesting procedure chunks to Qdrant...")
    ingest_procedures(
        chunks_csv=Path(chunks_csv),
        collection="procedures_hybrid",
        qdrant_url=QDRANT_URL,
        batch_size=100
    )
    
    print("\n✨ Procedures Pipeline completed successfully!")


def main() -> None:
    parser = argparse.ArgumentParser(description="Hephaestus Data Engineering Pipeline Orchestrator")
    parser.add_argument(
        "pipeline",
        choices=["interventions", "procedures", "all"],
        help="Pipeline to run: 'interventions', 'procedures', or 'all' to run both."
    )
    parser.add_argument(
        "--force-db-load",
        action="store_true",
        help="Force reload interventions.csv to Postgres even if table has data."
    )
    args = parser.parse_args()
    
    if args.pipeline == "interventions":
        run_interventions_pipeline(force_db_load=args.force_db_load)
    elif args.pipeline == "procedures":
        run_procedures_pipeline()
    elif args.pipeline == "all":
        run_interventions_pipeline(force_db_load=args.force_db_load)
        run_procedures_pipeline()


if __name__ == "__main__":
    main()
