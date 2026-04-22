"""Load sensor CSVs into the Postgres maintenance schema."""

import sys
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

DB_URL = "postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SQL_FILE = Path(__file__).resolve().parent / "init_sensor_db.sql"


def main():
    engine = create_engine(DB_URL)

    # 1. Create schema + tables
    print("Creating schema and tables ...")
    with engine.connect() as conn:
        conn.execute(text(SQL_FILE.read_text()))
        conn.commit()
    print("  Done.")

    # 2. Load sensor_catalog (77 rows)
    df_cat = pd.read_csv(DATA_DIR / "sensor_catalog.csv")
    print(f"Loading sensor_catalog ({len(df_cat)} rows) ...")
    df_cat.to_sql(
        "sensor_catalog", engine, schema="maintenance",
        if_exists="replace", index=False,
    )
    print("  Done.")

    # 3. Load sensor_readings (~800k rows, chunked)
    print("Loading sensor_readings ...")
    chunks = pd.read_csv(DATA_DIR / "sensor_readings.csv", chunksize=50_000)
    total = 0
    for i, chunk in enumerate(chunks):
        mode = "replace" if i == 0 else "append"
        chunk.to_sql(
            "sensor_readings", engine, schema="maintenance",
            if_exists=mode, index=False,
        )
        total += len(chunk)
        print(f"  {total:,} rows loaded ...")
    print(f"  Done ({total:,} total).")

    # 4. Load remaining_life (128 rows)
    df_rl = pd.read_csv(DATA_DIR / "remaining_life.csv")
    print(f"Loading remaining_life ({len(df_rl)} rows) ...")
    df_rl.to_sql(
        "remaining_life", engine, schema="maintenance",
        if_exists="replace", index=False,
    )
    print("  Done.")

    # 5. Load interventions
    df_int = pd.read_csv(DATA_DIR / "interventions.csv")
    print(f"Loading interventions ({len(df_int)} rows) ...")
    df_int.to_sql(
        "interventions", engine, schema="maintenance",
        if_exists="replace", index=False,
    )
    print("  Done.")

    # 6. Recreate indexes (replace drops them)
    print("Recreating indexes ...")
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_readings_machine_ts
                ON maintenance.sensor_readings (machine, timestamp);
            CREATE INDEX IF NOT EXISTS idx_readings_tag_ts
                ON maintenance.sensor_readings (tag, timestamp);
            CREATE INDEX IF NOT EXISTS idx_remaining_life_machine
                ON maintenance.remaining_life (machine);
            CREATE INDEX IF NOT EXISTS idx_interventions_machine_date
                ON maintenance.interventions (machine, date_start);
            CREATE INDEX IF NOT EXISTS idx_interventions_type
                ON maintenance.interventions (intervention_type);
        """))
        conn.commit()
    print("  Done.")

    print("\nAll tables loaded successfully.")


if __name__ == "__main__":
    main()
