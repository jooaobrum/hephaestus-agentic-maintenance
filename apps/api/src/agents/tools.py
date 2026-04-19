import cohere
from openai import OpenAI
from qdrant_client import QdrantClient, models
from langchain_core.tools import tool
from langsmith import traceable

from core.config import config
from sqlalchemy import create_engine, text
import pandas as pd

pg_engine = create_engine(config.PG_URL)
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
cohere_client = cohere.Client(api_key=config.CO_API_KEY)
qdrant_client = QdrantClient(url=config.QDRANT_URL)


@traceable(name="embed_query", run_type="embedding")
def _embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=config.EMBEDDING_MODEL)
    return response.data[0].embedding


@traceable(name="cm_data_retrieval", run_type="retriever")
def _retrieve_cm(query: str, top_k: int = 5) -> list[dict]:
    query_vector = _embed_text(query)
    search_results = qdrant_client.query_points(
        collection_name=config.CM_QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector, using=config.EMBEDDING_MODEL, limit=top_k // 2
            ),
            models.Prefetch(
                query=models.Document(
                    text=query, model="qdrant/" + config.KEYWORD_MODEL
                ),
                using=config.KEYWORD_MODEL,
                limit=top_k // 2,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1, 1])),
        limit=top_k,
    ).points
    return [
        {"id": point.id, "payload": point.payload, "score": point.score}
        for point in search_results
    ]


@traceable(name="proc_data_retrieval", run_type="retriever")
def _retrieve_procedures(query: str, top_k: int = 5) -> list[dict]:
    query_vector = _embed_text(query)
    search_results = qdrant_client.query_points(
        collection_name=config.PROC_QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector, using=config.EMBEDDING_MODEL, limit=top_k // 2
            ),
            models.Prefetch(
                query=models.Document(
                    text=query, model="qdrant/" + config.KEYWORD_MODEL
                ),
                using=config.KEYWORD_MODEL,
                limit=top_k // 2,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1, 1])),
        limit=top_k,
    ).points
    return [
        {"id": point.id, "payload": point.payload, "score": point.score}
        for point in search_results
    ]


@traceable(name="format_cm_context", run_type="prompt")
def _format_cm_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        context += (
            f"ID: {payload.get('id', 'N/A')}\n"
            f"Machine: {payload.get('machine', 'N/A')}\n"
            f"Date: {payload.get('date_start', 'N/A')}\n"
            f"Summary: {payload.get('summary', 'N/A')}\n" + "-" * 40 + "\n"
        )
    return context


@traceable(name="format_proc_context", run_type="prompt")
def _format_proc_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        context += (
            f"File: {payload.get('file_name', 'N/A')}\n"
            f"Section: {payload.get('section_title', 'N/A')}\n"
            f"Context: {payload.get('context', 'N/A')}\n"
            f"Text: {payload.get('text', 'N/A')}\n" + "-" * 40 + "\n"
        )
    return context


# @traceable(name="reranking", run_type="retriever")
# def rerank_results(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
#     if not results:
#         return []
#     contexts = [result["payload"]["summary"] for result in results]
#     response = cohere_client.rerank(
#         model=config.RERANKING_MODEL, query=query, documents=contexts, top_n=top_k
#     )
#     return [results[res.index] for res in response.results]


@traceable(name="sensor_catalog_retrieval", run_type="retriever")
def get_sensor_catalog(machine: str) -> str:
    """Return the sensor catalog for a given machine (tags, thresholds, units, fault correlations)."""
    query = text("""
        SELECT sensor_id, tag, sensor_name, unit,
               nominal_value, warn_lo, warn_hi, crit_lo, crit_hi,
               fault_correlation, active
        FROM maintenance.sensor_catalog
        WHERE machine = :machine
        ORDER BY tag
    """)
    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"machine": machine})
    if df.empty:
        return f"No sensors found for machine '{machine}'."
    return df.to_markdown(index=False)


@traceable(name="sensor_readings_retrieval", run_type="retriever")
def get_sensor_readings(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
) -> str:
    """Return sensor readings for a machine within a time window.

    Args:
        machine: Machine ID (e.g. 'HX-200').
        start_date: Start of window (ISO format, e.g. '2024-12-01').
        end_date: End of window (ISO format, e.g. '2024-12-18').
        tag: Optional sensor tag to filter (e.g. 'TS-101'). If omitted, returns all sensors.
    """
    if tag:
        query = text("""
            SELECT timestamp, tag, sensor_name, value, unit, status, warn_lo, warn_hi
            FROM maintenance.sensor_readings
            WHERE machine = :machine
              AND tag = :tag
              AND timestamp >= :start_date
              AND timestamp <= :end_date
            ORDER BY timestamp
        """)
        params = {
            "machine": machine,
            "tag": tag,
            "start_date": start_date,
            "end_date": end_date,
        }
    else:
        query = text("""
            SELECT timestamp, tag, sensor_name, value, unit, status, warn_lo, warn_hi
            FROM maintenance.sensor_readings
            WHERE machine = :machine
              AND timestamp >= :start_date
              AND timestamp <= :end_date
            ORDER BY tag, timestamp
        """)
        params = {"machine": machine, "start_date": start_date, "end_date": end_date}

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)
    if df.empty:
        return (
            f"No readings found for machine '{machine}' between {start_date} and {end_date}"
            + (f", tag '{tag}'" if tag else "")
            + "."
        )
    return df.to_markdown(index=False)


@traceable(name="remaining_life_retrieval", run_type="retriever")
def get_remaining_life(machine: str) -> str:
    """Return remaining useful life (RUL) for all components of a machine.

    Shows condition (EXCELLENT/GOOD/MONITOR/END_OF_LIFE/OVERDUE), remaining hours/pct,
    replacement cost, and next inspection date.
    """
    query = text("""
        SELECT component_id, component_name, condition,
               current_hours, remaining_hours, remaining_pct,
               unit_cost_eur, last_inspection, next_inspection, notes
        FROM maintenance.remaining_life
        WHERE machine = :machine
        ORDER BY remaining_pct ASC
    """)
    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"machine": machine})
    if df.empty:
        return f"No component life data found for machine '{machine}'."
    return df.to_markdown(index=False)


@tool
def get_formatted_cm_context(query: str, top_n: int = 10) -> str:
    """Retrieve past corrective maintenance intervention records for a query.

    Searches the maintenance history database using hybrid retrieval (dense + BM25).
    Use this for questions about past failures, root causes, repair actions, and intervention history.

    Args:
        query: The search query string describing the issue or machine.
        top_n: Number of results to retrieve. Defaults to 10.

    Returns:
        A formatted string with intervention records showing ID, machine, date, and summary.
    """
    results = _retrieve_cm(query, top_k=top_n)
    return _format_cm_context(results)


@tool
def get_formatted_procedure_context(query: str, top_n: int = 10) -> str:
    """Retrieve troubleshooting procedure documentation for a query.

    Searches the procedures knowledge base (extracted from machine troubleshooting PDFs).
    Use this for questions about diagnostic steps, fault codes, PPE, emergency procedures, and standard repair procedures.

    Args:
        query: The search query string describing the fault or procedure needed.
        top_n: Number of results to retrieve. Defaults to 10.

    Returns:
        A formatted string with procedure chunks showing file, section, context, and text.
    """
    results = _retrieve_procedures(query, top_k=top_n)
    return _format_proc_context(results)


@tool
def get_sensor_catalog_tool(machine: str) -> str:
    """Return the sensor catalog for a given machine.

    Lists all sensors installed on the machine with their tags, thresholds, units, and correlated fault codes.
    Use this to understand what sensors are available and what their normal/warning/critical ranges are.

    Args:
        machine: Machine ID (e.g. 'HX-200', 'CB-200').

    Returns:
        A markdown table with sensor details.
    """
    return get_sensor_catalog(machine)


@tool
def get_sensor_readings_tool(
    machine: str, start_date: str, end_date: str, tag: str | None = None
) -> str:
    """Return sensor readings for a machine within a time window.

    Use this to check recent sensor values, trends, and whether any readings are in warning/critical status.

    Args:
        machine: Machine ID (e.g. 'HX-200').
        start_date: Start of window (ISO format, e.g. '2024-12-01').
        end_date: End of window (ISO format, e.g. '2024-12-18').
        tag: Optional sensor tag to filter (e.g. 'HX-200-TS-101'). If omitted, returns all sensors.

    Returns:
        A markdown table with timestamped readings.
    """
    return get_sensor_readings(machine, start_date, end_date, tag)


@tool
def get_remaining_life_tool(machine: str) -> str:
    """Return remaining useful life (RUL) for all components of a machine.

    Shows condition (EXCELLENT/GOOD/MONITOR/END_OF_LIFE/OVERDUE), remaining hours/percentage,
    replacement cost, and next inspection date. Use this to assess component health and plan replacements.

    Args:
        machine: Machine ID (e.g. 'HX-200').

    Returns:
        A markdown table with component life data.
    """
    return get_remaining_life(machine)
