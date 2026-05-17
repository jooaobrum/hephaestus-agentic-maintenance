"""Ingest corrective-maintenance interventions CSV into a Qdrant hybrid collection.

Reads interventions.csv, filters CM rows, builds text summaries, embeds with
text-embedding-3-small (dense) + BM25 (sparse), and upserts into Qdrant.

Usage:
    uv run python scripts/ingestion/ingest_interventions_to_qdrant.py \\
        [--csv data/interventions.csv] \\
        [--collection cm_interventions_hybrid] \\
        [--qdrant-url http://localhost:6333] \\
        [--batch-size 100]
"""

import argparse
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    Modifier,
    PointStruct,
    SparseVectorParams,
    VectorParams,
)

EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536


def validate_fault_code(fault_code) -> bool:
    if not isinstance(fault_code, str):
        return False
    parts = fault_code.split("-")
    return len(parts) == 2 and len(parts[0]) == 1 and len(parts[1]) == 3


def build_summary(row: dict) -> str:
    parts = []
    if row.get("is_valid_fault_code"):
        parts.append(f"[FAULT_CODE] {row['fault_code']}")
    if pd.notna(row.get("related_intervention")):
        parts.append(f"[RELATED_INTERVENTION] {row['related_intervention']}")
    parts.append(f"[EVENT] {row['events']}")
    parts.append(f"[COMMENTS] {row['comments']}")
    return "\n".join(parts).strip()


def embed_batch(texts: list[str], client: OpenAI, batch_size: int) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        resp = client.embeddings.create(
            input=texts[i : i + batch_size], model=EMBEDDING_MODEL
        )
        embeddings.extend(e.embedding for e in resp.data)
    return embeddings


def ensure_collection(qdrant: QdrantClient, collection: str) -> None:
    try:
        qdrant.get_collection(collection)
        print(f"Collection '{collection}' already exists — upserting into it.")
    except Exception:
        qdrant.create_collection(
            collection_name=collection,
            vectors_config={
                EMBEDDING_MODEL: VectorParams(
                    size=VECTOR_SIZE, distance=Distance.COSINE
                )
            },
            sparse_vectors_config={"bm25": SparseVectorParams(modifier=Modifier.IDF)},
        )
        print(f"Created collection '{collection}'.")


def upsert_points(
    qdrant: QdrantClient,
    collection: str,
    records: list[dict],
    embeddings: list[list[float]],
    batch_size: int,
) -> None:
    now = datetime.utcnow().isoformat()
    points = []
    for record, dense in zip(records, embeddings):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    EMBEDDING_MODEL: dense,
                    "bm25": Document(text=record["summary"], model="qdrant/bm25"),
                },
                payload={
                    "id": record["id"],
                    "date_start": record["date_start"],
                    "machine": record["machine"],
                    "duration_min": record["duration_min"],
                    "summary": record["summary"],
                    "embedding_model": EMBEDDING_MODEL,
                    "created_at": now,
                },
            )
        )

    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=collection, points=points[i : i + batch_size])
        print(f"  Upserted {min(i + batch_size, len(points))}/{len(points)} points ...")


def ingest_interventions(
    csv_path: Path, collection: str, qdrant_url: str, batch_size: int = 100
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df = df[df["intervention_type"] == "CM"].copy()
    df["is_valid_fault_code"] = df["fault_code"].apply(validate_fault_code)
    df["summary"] = df.apply(build_summary, axis=1)
    records = df[["id", "date_start", "machine", "duration_min", "summary"]].to_dict(
        orient="records"
    )
    print(f"Loaded {len(records)} CM interventions from {csv_path}.")

    oai = OpenAI()
    qdrant = QdrantClient(url=qdrant_url)

    ensure_collection(qdrant, collection)

    print("Embedding ...")
    embeddings = embed_batch([r["summary"] for r in records], oai, batch_size)

    print("Upserting ...")
    upsert_points(qdrant, collection, records, embeddings, batch_size)

    print(f"\nDone — {len(records)} points in '{collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest CM interventions into Qdrant hybrid collection."
    )
    parser.add_argument("--csv", type=Path, default=Path("data/interventions.csv"))
    parser.add_argument("--collection", default="cm_interventions_hybrid")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    ingest_interventions(args.csv, args.collection, args.qdrant_url, args.batch_size)


if __name__ == "__main__":
    main()
