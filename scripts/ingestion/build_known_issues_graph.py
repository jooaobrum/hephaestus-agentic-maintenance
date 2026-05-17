"""Build the known_issues Qdrant collection from cm_interventions_hybrid.

Pipeline:
  1. Fetch all embeddings + payloads from cm_interventions_hybrid
  2. UMAP 10-D reduction + HDBSCAN clustering
  3. Select cluster representatives (medoid + top members by probability)
  4. Synthesize a KnownIssue per cluster via LLM structured output
  5. Recreate known_issues collection and upsert all issues

Usage:
    uv run python scripts/ingestion/build_known_issues_graph.py \\
        [--source-collection cm_interventions_hybrid] \\
        [--output-collection known_issues] \\
        [--qdrant-url http://localhost:6333] \\
        [--min-cluster-size 5] \\
        [--top-k-reps 8]
"""

import argparse
import uuid
from datetime import datetime

import numpy as np
import pandas as pd
import hdbscan
import umap
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_SIZE = 1536


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class RootCauseAction(BaseModel):
    root_cause: str = Field(
        description="Normalized root cause name (specific, avoid generalist terms)."
    )
    actions: list[str] = Field(
        description="Ordered list of corrective actions that resolved this root cause."
    )


class KnownIssue(BaseModel):
    symptom_name: str = Field(
        description=(
            "Normalized failure phenomenon name. "
            "NEVER include machine IDs or machine names — those belong in affected_machines. "
            "Good: 'Coil Cooling Flow Fault'. Bad: 'Coil Cooling Flow Fault – IH-300'. "
            "Calibrate specificity to cluster size: be more precise for small clusters."
        )
    )
    description: str = Field(
        description="2-4 sentence description of the issue pattern."
    )
    root_causes: list[RootCauseAction] = Field(
        description="Main root causes observed in this cluster, each with the actions that resolved them."
    )
    affected_machines: list[str] = Field(
        description="Normalized machine IDs impacted (e.g., ['CNC-500', 'CNC-750'])."
    )
    affected_machine_families: list[str] = Field(
        description="Machine type/family names (e.g., ['CNC Machining Center'])."
    )
    representative_intervention_ids: list[str] = Field(
        description="INT-IDs of the most representative interventions for this issue."
    )


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def scroll_all_points(
    qdrant: QdrantClient, collection: str
) -> tuple[list, np.ndarray, list[dict]]:
    ids, vectors, payloads = [], [], []
    offset = None
    while True:
        records, next_offset = qdrant.scroll(
            collection_name=collection,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=[EMBEDDING_MODEL],
        )
        for r in records:
            vec = r.vector
            if isinstance(vec, dict):
                vec = vec.get(EMBEDDING_MODEL)
            if vec is None:
                continue
            ids.append(r.id)
            vectors.append(vec)
            payloads.append(r.payload)
        if next_offset is None:
            break
        offset = next_offset
    return ids, np.array(vectors, dtype=np.float32), payloads


# ---------------------------------------------------------------------------
# Cluster
# ---------------------------------------------------------------------------


def cluster_embeddings(
    embeddings: np.ndarray, min_cluster_size: int, min_samples: int
) -> np.ndarray:
    reduced = umap.UMAP(
        n_components=10,
        n_neighbors=15,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    ).fit_transform(embeddings)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method="eom",
        metric="euclidean",
        prediction_data=True,
    )
    clusterer.fit(reduced)
    return clusterer.labels_, clusterer.probabilities_


def get_representatives(
    df_cluster: pd.DataFrame, cluster_embeddings: np.ndarray, top_k: int
) -> pd.DataFrame:
    centroid = cluster_embeddings.mean(axis=0)
    dists = np.linalg.norm(cluster_embeddings - centroid, axis=1)
    medoid_idx = int(np.argmin(dists))

    rest = df_cluster.drop(index=df_cluster.index[medoid_idx])
    rest = rest.sort_values("cluster_prob", ascending=False).head(top_k - 1)
    return pd.concat([df_cluster.iloc[[medoid_idx]], rest], ignore_index=True)


# ---------------------------------------------------------------------------
# Synthesize
# ---------------------------------------------------------------------------

SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior maintenance engineer synthesizing a knowledge base of known equipment issues.
Given a set of corrective-maintenance intervention records belonging to the same semantic cluster,
produce a single structured KnownIssue entry.

Rules:
- symptom_name must describe the failure phenomenon only — never include machine IDs, machine names,
  or location suffixes. Machine scope belongs exclusively in affected_machines and affected_machine_families.
- symptom_name must be specific and equipment-class-scoped; avoid generic labels.
  Scale specificity to cluster size: a 5-record cluster warrants more precision than a 30-record one.
- List ALL root causes observed across the cluster, not just the most common one.
- Actions must be practical and ordered (first step first).
- Machines and families must be normalized (consistent casing, standard IDs).
- representative_intervention_ids should list only the most representative INT-IDs provided.
"""


def synthesize_known_issue(
    reps: pd.DataFrame, cluster_size: int, oai: OpenAI
) -> KnownIssue:
    lines = ["--- Representative Interventions ---"]
    for _, row in reps.iterrows():
        lines.append(
            f"\nID: {row.get('id', 'N/A')}\n"
            f"Machine: {row.get('machine', 'N/A')}\n"
            f"Summary:\n{row.get('summary', '')}"
        )
    user_msg = f"Cluster size (total interventions): {cluster_size}\n" + "\n".join(
        lines
    )

    resp = oai.beta.chat.completions.parse(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": SYNTHESIS_SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        response_format=KnownIssue,
    )
    return resp.choices[0].message.parsed


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def recreate_collection(qdrant: QdrantClient, name: str) -> None:
    if qdrant.collection_exists(name):
        qdrant.delete_collection(name)
        print(f"Dropped existing '{name}' collection.")
    qdrant.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBEDDING_SIZE, distance=Distance.COSINE),
    )
    print(f"Created '{name}' collection.")


def upsert_known_issues(
    qdrant: QdrantClient, issues: dict[int, KnownIssue], collection: str, oai: OpenAI
) -> None:
    points = []
    now = datetime.utcnow().isoformat()
    for cid, issue in issues.items():
        text = f"{issue.symptom_name}\n{issue.description}"
        vec = oai.embeddings.create(input=text, model=EMBEDDING_MODEL).data[0].embedding
        payload = issue.model_dump()
        payload["version"] = 1
        payload["created_at"] = now
        payload["updated_at"] = now
        points.append(PointStruct(id=str(uuid.uuid4()), vector=vec, payload=payload))

    qdrant.upsert(collection_name=collection, points=points)
    print(f"Upserted {len(points)} known issues into '{collection}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_known_issues(
    source_collection: str = "cm_interventions_hybrid",
    output_collection: str = "known_issues",
    qdrant_url: str = "http://localhost:6333",
    min_cluster_size: int = 5,
    top_k_reps: int = 8,
) -> None:
    oai = OpenAI()
    qdrant = QdrantClient(url=qdrant_url)

    print(f"Fetching points from '{source_collection}' ...")
    ids, embeddings, payloads = scroll_all_points(qdrant, source_collection)
    print(f"Fetched {len(ids)} points (embedding dim={embeddings.shape[1]}).")

    df = pd.DataFrame(payloads)

    print("Clustering ...")
    labels, probs = cluster_embeddings(embeddings, min_cluster_size, min_samples=3)
    df["cluster"] = labels
    df["cluster_prob"] = probs

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"Clusters found: {n_clusters}  |  Noise points: {n_noise}/{len(labels)}")

    cluster_ids = sorted(c for c in set(labels) if c != -1)
    cluster_reps: dict[int, pd.DataFrame] = {}
    for cid in cluster_ids:
        mask = (df["cluster"] == cid).values
        cluster_reps[cid] = get_representatives(
            df[mask].reset_index(drop=True),
            embeddings[mask],
            top_k_reps,
        )

    print("\nSynthesizing known issues ...")
    known_issues: dict[int, KnownIssue] = {}
    for cid, reps in cluster_reps.items():
        cluster_size = int((df["cluster"] == cid).sum())
        print(f"  Cluster {cid} ({cluster_size} interventions) ...")
        issue = synthesize_known_issue(reps, cluster_size, oai)
        known_issues[cid] = issue
        print(f"    → {issue.symptom_name}")

    print(f"\nSynthesized {len(known_issues)} known issues.")

    recreate_collection(qdrant, output_collection)
    upsert_known_issues(qdrant, known_issues, output_collection, oai)

    print(f"\nDone — {len(known_issues)} known issues in '{output_collection}'.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build known_issues Qdrant collection from CM interventions."
    )
    parser.add_argument("--source-collection", default="cm_interventions_hybrid")
    parser.add_argument("--output-collection", default="known_issues")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--top-k-reps", type=int, default=8)
    args = parser.parse_args()

    build_known_issues(
        source_collection=args.source_collection,
        output_collection=args.output_collection,
        qdrant_url=args.qdrant_url,
        min_cluster_size=args.min_cluster_size,
        top_k_reps=args.top_k_reps,
    )


if __name__ == "__main__":
    main()
