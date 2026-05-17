"""Ingest procedure chunks CSV into Qdrant hybrid collection.

Reads procedure_chunks.csv (output of extract_procedures.py), embeds
each chunk's context+text, and upserts into procedures_hybrid.

Usage:
    uv run python scripts/ingestion/ingest_procedures.py \\
        [--chunks-csv data/procedure_chunks.csv] \\
        [--collection procedures_hybrid] \\
        [--qdrant-url http://localhost:6333] \\
        [--batch-size 100]
"""

import argparse
import json
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


def embed_batch(texts: list[str], oai: OpenAI, batch_size: int) -> list[list[float]]:
    embeddings = []
    for i in range(0, len(texts), batch_size):
        resp = oai.embeddings.create(
            input=texts[i : i + batch_size], model=EMBEDDING_MODEL
        )
        embeddings.extend(e.embedding for e in resp.data)
    return embeddings


def upsert_chunks(
    qdrant: QdrantClient,
    collection: str,
    df: pd.DataFrame,
    embeddings: list[list[float]],
    batch_size: int,
) -> None:
    now = datetime.utcnow().isoformat()
    points = []

    for idx, row in df.iterrows():
        context_text = str(row["context"]) + "\n\n" + str(row["text"])
        page_numbers = (
            json.loads(row["page_numbers"])
            if isinstance(row["page_numbers"], str)
            else row["page_numbers"]
        )
        image_paths = (
            json.loads(row["image_paths"])
            if isinstance(row["image_paths"], str)
            else row["image_paths"]
        )

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    EMBEDDING_MODEL: embeddings[idx],
                    "bm25": Document(text=context_text, model="qdrant/bm25"),
                },
                payload={
                    "chunk_number": int(row["chunk_number"]),
                    "file_name": str(row["file_name"]),
                    "section_title": str(row["section_title"]),
                    "context": str(row["context"]),
                    "text": str(row["text"]),
                    "page_numbers": page_numbers,
                    "image_paths": image_paths,
                    "contains_table": bool(row["contains_table"]),
                    "contains_image": bool(row["contains_image"]),
                    "prev_chunk": int(row["prev_chunk"])
                    if pd.notna(row["prev_chunk"])
                    else None,
                    "next_chunk": int(row["next_chunk"])
                    if pd.notna(row["next_chunk"])
                    else None,
                    "embedding_model": EMBEDDING_MODEL,
                    "created_at": now,
                },
            )
        )

    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=collection, points=points[i : i + batch_size])
        print(f"  Upserted {min(i + batch_size, len(points))}/{len(points)} points ...")


def ingest_procedures(
    chunks_csv: Path, collection: str, qdrant_url: str, batch_size: int = 100
) -> None:
    if not chunks_csv.exists():
        raise FileNotFoundError(f"CSV not found: {chunks_csv}")

    df = pd.read_csv(chunks_csv)
    print(f"Loaded {len(df)} chunks from {chunks_csv}.")

    oai = OpenAI()
    qdrant = QdrantClient(url=qdrant_url)

    ensure_collection(qdrant, collection)

    print("Embedding ...")
    context_texts = [
        str(row["context"]) + "\n\n" + str(row["text"]) for _, row in df.iterrows()
    ]
    embeddings = embed_batch(context_texts, oai, batch_size)

    print("Upserting ...")
    upsert_chunks(qdrant, collection, df, embeddings, batch_size)

    print(f"\nDone — {len(df)} chunks in '{collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest procedure chunks CSV into Qdrant."
    )
    parser.add_argument(
        "--chunks-csv", type=Path, default=Path("data/procedure_chunks.csv")
    )
    parser.add_argument("--collection", default="procedures_hybrid")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    ingest_procedures(
        args.chunks_csv, args.collection, args.qdrant_url, args.batch_size
    )


if __name__ == "__main__":
    main()
