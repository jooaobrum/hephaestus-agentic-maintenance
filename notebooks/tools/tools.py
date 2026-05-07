from langchain_core.tools import tool
from datetime import datetime
from qdrant_client import models

from openai import OpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text

PG_URL = (
    "postgresql+psycopg://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
)
pg_engine = create_engine(PG_URL)

# Clients (reuse existing)
OPENAI_CLIENT = OpenAI()
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)

# Models & Collections
CM_COLLECTION = "cm_interventions_hybrid"
PROC_COLLECTION = "procedures_hybrid"
EMBEDDING_MODEL = "text-embedding-3-small"
KEYWORD_MODEL = "bm25"


def _build_proc_filter(
    file_name: str | None = None,
    contains_table: bool | None = None,
):
    conditions = []

    if file_name:
        conditions.append(
            models.FieldCondition(
                key="file_name",
                match=models.MatchText(text=file_name),
            )
        )

    if contains_table is not None:
        conditions.append(
            models.FieldCondition(
                key="contains_table",
                match=models.MatchValue(value=contains_table),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def _build_filters(
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
):
    conditions = []

    # Exact machine
    if machine:
        conditions.append(
            models.FieldCondition(
                key="machine",
                match=models.MatchValue(value=machine),
            )
        )

    # Prefix/group machine (e.g. all CB*)
    if machine_prefix:
        conditions.append(
            models.FieldCondition(
                key="machine",
                match=models.MatchText(text=machine_prefix),
            )
        )

    # Date range
    if date_start or date_end:
        range_kwargs = {}

        if date_start:
            range_kwargs["gte"] = date_start

        if date_end:
            range_kwargs["lte"] = date_end

        conditions.append(
            models.FieldCondition(
                key="date_start",
                range=models.DatetimeRange(**range_kwargs),
            )
        )

    if not conditions:
        return None

    return models.Filter(must=conditions)


def get_sensor_timeline(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str,
) -> str:
    """
    Return chronological sensor readings for a single tag with per-reading trend arrows (↑↓→)
    and a summary of the first threshold breach and max rate of change.
    Use this to understand how a specific sensor degraded over time.
    Dates: ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS). Tag: e.g. 'TEMP_01'.
    """
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

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return f"No readings found for {tag} on {machine} between {start_date} and {end_date}."

    df["value_prev"] = df["value"].shift(1)
    df["delta"] = df["value"] - df["value_prev"]
    df["trend"] = df["delta"].apply(lambda x: "↑" if x > 0 else ("↓" if x < 0 else "→"))

    def mark_anomaly(row):
        if row["status"] in ["WARNING", "CRITICAL"]:
            return f"⚠️ {row['status']}"
        return row["status"]

    df["status_marked"] = df.apply(mark_anomaly, axis=1)

    anomalies = df[df["status"].isin(["WARNING", "CRITICAL"])]
    summary = ""
    if not anomalies.empty:
        first_anomaly = anomalies.iloc[0]
        summary += (
            f"\nFirst threshold breach: {first_anomaly['timestamp']} "
            f"(value={first_anomaly['value']}, status={first_anomaly['status']})\n"
        )
        if len(df) > 1:
            max_delta = df["delta"].max()
            min_delta = df["delta"].min()
            summary += f"**Trend:** max increase {max_delta:.2f}/reading, max decrease {min_delta:.2f}/reading\n"

    display_df = df[
        [
            "timestamp",
            "tag",
            "sensor_name",
            "value",
            "unit",
            "trend",
            "status_marked",
            "warn_lo",
            "warn_hi",
        ]
    ].copy()
    display_df.columns = [
        "Timestamp",
        "Tag",
        "Sensor",
        "Value",
        "Unit",
        "Trend",
        "Status",
        "Warn Low",
        "Warn High",
    ]

    return f"**Sensor Timeline for {tag}:**\n{summary}\n{display_df.to_markdown(index=False)}"


def get_threshold_events(
    machine: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:
    """
    Return all WARNING/CRITICAL sensor readings for a machine in a time window.
    Includes breach direction (ABOVE warn_hi / BELOW warn_lo) and a summary count by severity.
    Use this as a fast triage to identify which sensors are in alarm before diving deeper.
    Timestamps: ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
    """
    query = text("""
        SELECT timestamp, tag, sensor_name, value, unit, status, warn_lo, warn_hi
        FROM maintenance.sensor_readings
        WHERE machine = :machine
          AND timestamp >= :timestamp_start
          AND timestamp <= :timestamp_end
          AND status IN ('WARNING', 'CRITICAL')
        ORDER BY timestamp DESC
    """)
    params = {
        "machine": machine,
        "timestamp_start": timestamp_start,
        "timestamp_end": timestamp_end,
    }

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return f"No threshold breaches found for {machine} between {timestamp_start} and {timestamp_end}."

    def classify_breach(row):
        if row["status"] == "CRITICAL":
            if row["value"] < row["warn_lo"]:
                return f"BELOW warn_lo ({row['warn_lo']})"
            else:
                return f"ABOVE warn_hi ({row['warn_hi']})"
        elif row["status"] == "WARNING":
            if row["value"] < row["warn_lo"]:
                return f"BELOW warn_lo ({row['warn_lo']})"
            else:
                return f"ABOVE warn_hi ({row['warn_hi']})"
        return ""

    df["breach_type"] = df.apply(classify_breach, axis=1)

    critical_count = len(df[df["status"] == "CRITICAL"])
    warning_count = len(df[df["status"] == "WARNING"])
    unique_tags = df["tag"].nunique()
    summary = f"**Summary:** {critical_count} CRITICAL events, {warning_count} WARNING events across {unique_tags} unique sensors\n"

    display_df = df[
        ["timestamp", "tag", "sensor_name", "value", "unit", "breach_type"]
    ].copy()
    display_df.columns = ["Timestamp", "Tag", "Sensor", "Value", "Unit", "Breach Type"]

    return f"{summary}\n{display_df.to_markdown(index=False)}"


def get_sensor_catalog(machine: str) -> str:
    """
    Return the full sensor catalog for a machine: tag names, units, nominal value,
    warn_lo/warn_hi thresholds, fault_correlation labels, and active status.
    Call this first to discover which tags exist before querying readings or timelines.
    """
    query = text("""
        SELECT sensor_id, tag, sensor_name, unit,
               nominal_value, warn_lo, warn_hi,
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


def get_sensor_readings(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
) -> str:
    """
    Return raw sensor readings (value, unit, status, thresholds) for a machine in a time window.
    Optionally filter to a single tag; omit tag to get all sensors sorted by tag then time.
    Use get_sensor_catalog first to find valid tag names.
    Dates: ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS).
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


def get_remaining_life(machine: str) -> str:
    """
    Return remaining useful life (RUL) for all components of a machine, sorted by most critical first.
    Includes condition label, current hours, remaining hours, remaining %, replacement cost,
    last/next inspection dates, and notes. Use this to assess wear-driven failure risk.
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


from datetime import datetime, timedelta
import json
import pandas as pd
from sqlalchemy import text
from qdrant_client import models
from openai import OpenAI

openai_client_local = OpenAI()


def _embed_text(text: str) -> list[float]:
    response = openai_client_local.embeddings.create(input=text, model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _expand_chunk_window(results: list[dict]) -> list[dict]:
    expanded = []
    seen = set()

    for result in results:
        payload = result["payload"]
        file_name = payload["file_name"]

        chunk_numbers = [
            payload.get("prev_chunk"),
            payload.get("chunk_number"),
            payload.get("next_chunk"),
        ]

        chunk_numbers = [int(c) for c in chunk_numbers if c is not None]

        # Fetch neighboring chunks
        neighbor_results = QDRANT_CLIENT.scroll(
            collection_name=PROC_COLLECTION,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="file_name",
                        match=models.MatchValue(value=file_name),
                    ),
                    models.FieldCondition(
                        key="chunk_number",
                        match=models.MatchAny(any=chunk_numbers),
                    ),
                ]
            ),
            limit=len(chunk_numbers),
            with_payload=True,
        )[0]

        for point in neighbor_results:
            if point.id in seen:
                continue
            seen.add(point.id)
            expanded.append(
                {
                    "id": point.id,
                    "payload": point.payload,
                    "score": getattr(point, "score", 0),
                }
            )

    # Sort by document order
    expanded.sort(
        key=lambda x: (
            x["payload"]["file_name"],
            x["payload"]["chunk_number"],
        )
    )
    return expanded


def _retrieve_cm(
    query: str,
    top_k: int = 5,
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
):
    query_vector = _embed_text(query)

    query_filter = _build_filters(
        machine=machine,
        machine_prefix=machine_prefix,
        date_start=date_start,
        date_end=date_end,
    )

    search_results = QDRANT_CLIENT.query_points(
        collection_name=CM_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=EMBEDDING_MODEL,
                limit=top_k * 2,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="qdrant/" + KEYWORD_MODEL,
                ),
                using=KEYWORD_MODEL,
                limit=top_k * 2,
                filter=query_filter,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1, 1])),
        limit=top_k,
    ).points

    return [
        {
            "id": point.id,
            "payload": point.payload,
            "score": point.score,
        }
        for point in search_results
    ]


def _retrieve_procedures(
    query: str,
    top_k: int = 5,
    file_name: str | None = None,
    contains_table: bool | None = None,
    expand_window: bool = True,
):
    query_vector = _embed_text(query)

    query_filter = _build_proc_filter(
        file_name=file_name,
        contains_table=contains_table,
    )

    search_results = QDRANT_CLIENT.query_points(
        collection_name=PROC_COLLECTION,
        prefetch=[
            models.Prefetch(
                query=query_vector,
                using=EMBEDDING_MODEL,
                limit=top_k * 2,
                filter=query_filter,
            ),
            models.Prefetch(
                query=models.Document(
                    text=query,
                    model="qdrant/" + KEYWORD_MODEL,
                ),
                using=KEYWORD_MODEL,
                limit=top_k * 2,
                filter=query_filter,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[0.7, 1.3])),
        limit=top_k,
        with_payload=True,
    ).points

    results = [
        {
            "id": point.id,
            "payload": point.payload,
            "score": point.score,
        }
        for point in search_results
    ]

    if expand_window:
        results = _expand_chunk_window(results)
    return results


def _format_cm_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        intervention_id = payload.get('id', 'N/A')
        context += (
            f"Source: INT-{intervention_id}\n"
            f"Machine: {payload.get('machine', 'N/A')}\n"
            f"Date: {payload.get('date_start', 'N/A')}\n"
            f"Summary: {payload.get('summary', 'N/A')}\n" + "-" * 40 + "\n"
        )
    return context


def _format_proc_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        file_name = payload.get('file_name', 'N/A')
        page_number = payload.get('page_number', payload.get('page', 'N/A'))
        chunk_number = payload.get('chunk_number', 'N/A')
        source_ref = f"PROC_REF:{file_name}:{page_number}:chunk#{chunk_number}"

        context += (
            f"Source: {source_ref}\n"
            f"File: {file_name}\n"
            f"Section: {payload.get('section_title', 'N/A')}\n"
            f"Context: {payload.get('context', 'N/A')}\n"
            f"Text: {payload.get('text', 'N/A')}\n" + "-" * 40 + "\n"
        )
    return context


@tool
def get_formatted_cm_context(
    query: str,
    top_k: int = 5,
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
) -> str:
    """
    Retrieve historical corrective maintenance (CM) interventions using hybrid search (dense + BM25).
    and pre-filtering.

    Use when: investigating past failures with known or broad symptom context. Query is always about an issue, symptom.

    Supports filtering by:
    - machine (exact ID)
    - machine_prefix (e.g. 'CB*')
    - date range (date_start, date_end)

    Returns: formatted list of matching interventions (ID, machine, date, summary).
    """
    results = _retrieve_cm(
        query=query,
        top_k=top_k,
        machine=machine,
        machine_prefix=machine_prefix,
        date_start=date_start,
        date_end=date_end,
    )
    return _format_cm_context(results)


@tool
def get_recent_formatted_cm_context(
    query: str,
    machine: str,
    top_k: int = 5,
    days_span: int = 30,
    date_end: str | None = None,
) -> str:
    """
    Retrieve recent CM interventions within a rolling time window (default: last 30 days).

    Use when: analyzing current or recent machine issues. DEFAULT choice for CM history (covers 95% of cases).
    Use BEFORE: get_long_formatted_cm_context (only if recent returns no results + chronic suspected).
    Use AFTER: get_intervention_detail(intervention_id) to expand relevant cases.

    Behavior:
    - date_start is derived from date_end - days_span
    - date_end defaults to today if not provided

    Returns: formatted recent intervention summaries (ID, machine, date, summary).
    """
    if not date_end:
        date_end = datetime.now().strftime("%Y-%m-%d")

    try:
        ref_date = datetime.fromisoformat(date_end)
    except ValueError:
        ref_date = datetime.strptime(date_end, "%Y-%m-%d")

    start_date = (ref_date - timedelta(days=days_span)).strftime("%Y-%m-%d")

    results = _retrieve_cm(
        query=query,
        top_k=top_k,
        machine=machine,
        date_start=start_date,
        date_end=date_end,
    )
    return (
        f"**Recent Interventions (Last {days_span} days until {date_end}):**\n"
        + _format_cm_context(results)
    )


@tool
def get_long_formatted_cm_context(
    query: str,
    machine: str | None = None,
    machine_prefix: str | None = None,
    top_k: int = 10,
) -> str:
    """
    Retrieve full-history CM interventions with no time filtering.

    Use ONLY when: get_recent_formatted_cm_context found nothing + you suspect chronic/recurring issue.
    Supports fleet-wide queries: machine_prefix (e.g., "CB*") to find pattern across machine family.

    Returns: formatted historical intervention summaries (oldest to newest).
    """
    results = _retrieve_cm(
        query=query,
        top_k=top_k,
        machine=machine,
        machine_prefix=machine_prefix,
    )
    return "**Historical Interventions (Full History):**\n" + _format_cm_context(
        results
    )


@tool
def get_formatted_procedure_context(
    query: str,
    top_k: int = 5,
    file_name: str | None = None,
    contains_table: bool | None = None,
    expand_window: bool = True,
) -> str:
    """
    Retrieve troubleshooting procedure documentation using hybrid search (dense + BM25).

    Use when: need diagnostic guidance, fault codes, decision tables, or step-by-step instructions.
    Use AFTER initial hypothesis generation (from graph or symptoms) to ground in procedures.
    Use BEFORE sensor deep-dives if procedure provides specific tag/location guidance.

    Optional filters:
    - file_name: restrict to a specific document
    - contains_table: prioritize diagnostic tables
    - expand_window: include neighboring chunks for continuity

    Returns: formatted procedure sections with context and structured steps.
    """
    results = _retrieve_procedures(
        query=query,
        top_k=top_k,
        file_name=file_name,
        contains_table=contains_table,
        expand_window=expand_window,
    )
    return _format_proc_context(results)


@tool
def check_machine_exists(machine: str) -> str:
    """
    Validate whether a machine ID exists in the system.

    Use FIRST whenever a machine ID is provided in user input.
    Blocks all downstream tools until machine is validated.

    Checks:
    - maintenance interventions database
    - sensor catalog (fallback)

    Returns:
    - confirmation if machine exists with metadata (intervention count, date range)
    - or error message if not found (user must provide valid ID)
    """
    query = text("""
        SELECT machine, COUNT(*) as intervention_count, 
               MIN(date_start) as first_intervention, MAX(date_start) as last_intervention
        FROM maintenance.interventions
        WHERE machine = :machine
        GROUP BY machine
    """)
    with pg_engine.connect() as conn:
        result = conn.execute(query, {"machine": machine}).fetchone()

    if result:
        return (
            f"✓ Machine '{machine}' exists in database.\n"
            f"- Interventions recorded: {result[1]}\n"
            f"- First intervention: {result[2]}\n"
            f"- Last intervention: {result[3]}"
        )

    query_sensors = text("""
        SELECT DISTINCT machine FROM maintenance.sensor_catalog 
        WHERE machine = :machine LIMIT 1
    """)
    with pg_engine.connect() as conn:
        sensor_result = conn.execute(query_sensors, {"machine": machine}).fetchone()

    if sensor_result:
        return f"✓ Machine '{machine}' found in sensor catalog (no interventions yet)."

    return f"✗ Machine '{machine}' not found in database. Please verify the machine ID."


@tool
def list_available_machines() -> str:
    """
    List all machines with recorded maintenance interventions.

    Use when: user does not specify a machine or needs valid machine IDs.

    Returns: alphabetically sorted list of machine IDs.
    """
    query = text("""
        SELECT DISTINCT machine FROM maintenance.interventions ORDER BY machine
    """)
    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return "No machines found in database."

    machines = sorted(df["machine"].unique().tolist())
    return "**Available Machines:**\n" + "\n".join(f"- {m}" for m in machines)


@tool
def get_current_date() -> str:
    """
    Return the current system date (YYYY-MM-DD).

    Use for: anchoring relative date calculations before downstream queries.
    """
    return datetime.now().strftime("%Y-%m-%d")


@tool
def calculate_date_window(reference_date: str, days_back: int) -> str:
    """
    Compute a relative date window from a reference date.

    Use when: building explicit time ranges for sensor or maintenance queries.

    Inputs:
    - reference_date: YYYY-MM-DD
    - days_back: number of days to look back

    Returns: JSON with start_date, end_date, label, days_span.
    """
    ref_date = datetime.fromisoformat(reference_date)
    start_date = ref_date - timedelta(days=days_back)
    end_date = ref_date

    labels = {
        1: "yesterday to today",
        7: "last 7 days",
        14: "last 2 weeks",
        30: "last month",
    }

    result = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "label": labels.get(days_back, f"last {days_back} days"),
        "days_span": days_back,
    }

    return json.dumps(result)


@tool
def get_sensor_catalog_tool(machine: str) -> str:
    """
    Retrieve sensor catalog metadata for a machine.

    Use FIRST in any sensor analysis to discover valid tags and thresholds.
    Use BEFORE: get_threshold_events_tool, get_sensor_readings_tool, get_sensor_timeline_tool.

    Includes:
    - tag names
    - units
    - thresholds (warn_lo / warn_hi)
    - fault correlation
    - active status

    Returns: structured sensor catalog.
    """
    return get_sensor_catalog(machine)


@tool
def get_sensor_readings_tool(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
) -> str:
    """
    Retrieve raw sensor readings for a machine in a time window.

    Use when: analyzing sensor values or trends.

    Optional:
    - tag: limit to a single sensor

    Returns: time-series sensor data.
    """
    return get_sensor_readings(machine, start_date, end_date, tag)


@tool
def get_remaining_life_tool(machine: str) -> str:
    """
    Retrieve remaining useful life (RUL) estimates for machine components.

    Use when: assessing wear-related contribution to failures.

    Returns: component-level RUL and condition metrics.
    """
    return get_remaining_life(machine)


@tool
def get_sensor_timeline_tool(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str,
) -> str:
    """
    Retrieve detailed time-series for a single sensor tag with trend analysis.

    Use AFTER get_sensor_anomaly_summary identifies the worst sensors to investigate.
    Use to understand degradation pattern before deep-diving.

    Includes:
    - trend direction (↑ ↓ →)
    - first threshold breach timestamp
    - rate-of-change (max delta per reading)

    Returns: chronological sensor evolution with anomaly markers.
    """
    return get_sensor_timeline(machine, start_date, end_date, tag)


@tool
def get_threshold_events_tool(
    machine: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:
    """
    Retrieve all sensor threshold breaches (WARNING / CRITICAL) in a time window.

    Use AFTER get_sensor_catalog_tool to identify abnormal signals.
    Use BEFORE: get_sensor_anomaly_summary, get_sensor_timeline_tool.

    Returns:
    - affected sensors
    - severity levels
    - breach direction (above/below thresholds)
    - chronological list (newest first)
    """
    return get_threshold_events(machine, timestamp_start, timestamp_end)


@tool
def query_known_issues_graph(query: str, machine: str | None = None) -> str:
    """
    Search Knowledge Graph for known failure modes and patterns.

    Use EARLY in diagnosis to generate initial hypotheses before sensor tool work.
    Use FIRST, in parallel with intake, to seed confidence scores.
    This is a consolidated knowledge base of previous failures and their root causes.

    Optional:
    - machine: prioritize machine-specific known issues (boosts confidence if matched)

    Returns:
    - symptom match
    - description
    - ranked root causes with recommended actions
    """
    query_vector = _embed_text(query)

    filter_cond = None
    if machine:
        from qdrant_client import models

        filter_cond = models.Filter(
            should=[
                models.FieldCondition(
                    key="payload.affected_machines",
                    match=models.MatchValue(value=machine),
                ),
            ]
        )

    try:
        search_result = QDRANT_CLIENT.query_points(
            collection_name="known_issues",
            query=query_vector,
            using="text-embedding-3-small",
            query_filter=filter_cond,
            limit=3,
        ).points
    except Exception as e:
        return f"Error querying Knowledge Graph: {e}"

    if not search_result:
        search_result = QDRANT_CLIENT.query_points(
            collection_name="known_issues",
            query=query_vector,
            using="text-embedding-3-small",
            limit=3,
        ).points

    if not search_result:
        return "No known issues found in Knowledge Graph for this symptom."

    output = []
    for hit in search_result:
        p = hit.payload
        rep_ids = p.get("representative_intervention_ids", [])
        source_ref = f"GRAPH: {', '.join(rep_ids)}" if rep_ids else "GRAPH: (no interventions)"

        res = f"Source: {source_ref}\n"
        res += f"Symptom: {p.get('symptom_name', 'N/A')}\n"
        res += f"Description: {p.get('description', 'N/A')}\n"
        res += "Potential Root Causes & Actions:\n"
        for rc in p.get("root_causes", []):
            res += f"- {rc.get('root_cause')}: {', '.join(rc.get('actions', []))}\n"

        machines = p.get("affected_machines", [])
        if machines:
            res += f"Fleet impact: {len(machines)} machine(s) affected — {', '.join(machines)}\n"

        output.append(res)

    return "\n---\n".join(output)


@tool
def get_sensor_anomaly_summary(
    machine: str,
    start_date: str,
    end_date: str,
    top_n: int = 5,
) -> str:
    """
    Aggregate threshold events by sensor/component, showing first breach + max severity.

    Use AFTER get_threshold_events_tool to drill into which sensors to investigate next.
    Groups WARNING/CRITICAL events, shows most critical sensors first.

    Inputs:
    - machine: machine ID
    - start_date, end_date: ISO date range
    - top_n: number of worst sensors to show (default 5)

    Returns: summary table of top sensors by severity, with first breach timestamp.
    """
    query = text("""
        SELECT
            tag,
            sensor_name,
            COUNT(*) as event_count,
            MAX(CASE WHEN status = 'CRITICAL' THEN 1 ELSE 0 END) as has_critical,
            MIN(timestamp) as first_breach,
            MAX(value) as max_value,
            MIN(value) as min_value,
            ROUND(AVG(value)::numeric, 2) as avg_value,
            warn_lo,
            warn_hi
        FROM maintenance.sensor_readings
        WHERE machine = :machine
          AND timestamp >= :start_date
          AND timestamp <= :end_date
          AND status IN ('WARNING', 'CRITICAL')
        GROUP BY tag, sensor_name, warn_lo, warn_hi
        ORDER BY has_critical DESC, event_count DESC
        LIMIT :top_n
    """)
    params = {
        "machine": machine,
        "start_date": start_date,
        "end_date": end_date,
        "top_n": top_n,
    }

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return f"No threshold anomalies found for {machine} between {start_date} and {end_date}."

    display_df = df[
        ["tag", "sensor_name", "has_critical", "event_count", "first_breach", "avg_value", "warn_lo", "warn_hi"]
    ].copy()
    display_df.columns = ["Tag", "Sensor", "Critical?", "Events", "First Breach", "Avg Value", "Warn Low", "Warn High"]
    display_df["Critical?"] = display_df["Critical?"].map({1: "Yes", 0: "No"})

    return f"**Sensor Anomaly Summary (Top {top_n}):**\n{display_df.to_markdown(index=False)}"


@tool
def get_intervention_detail(intervention_id: str) -> str:
    """
    Retrieve full details of a specific corrective maintenance intervention.

    Use AFTER finding a relevant intervention in CM search results to expand the summary
    into actionable details: parts replaced, duration, outcomes, technician notes.

    Inputs:
    - intervention_id: ID from CM search result

    Returns: complete intervention record with all metadata.
    """
    query = text("""
        SELECT
            id,
            machine,
            date_start,
            date_end,
            duration_hours,
            summary,
            description,
            root_cause,
            parts_replaced,
            labor_hours,
            downtime_hours,
            cost_eur,
            technician_name,
            tags
        FROM maintenance.interventions
        WHERE id = :intervention_id
    """)

    with pg_engine.connect() as conn:
        result = conn.execute(query, {"intervention_id": intervention_id}).fetchone()

    if not result:
        return f"Intervention '{intervention_id}' not found."

    columns = [
        "ID", "Machine", "Start Date", "End Date", "Duration (hrs)",
        "Summary", "Description", "Root Cause", "Parts Replaced",
        "Labor (hrs)", "Downtime (hrs)", "Cost (€)", "Technician", "Tags"
    ]

    output = f"**Intervention Detail: {intervention_id}**\n\n"
    for col, val in zip(columns, result):
        if val is not None:
            output += f"**{col}:** {val}\n"

    return output


@tool
def get_fleet_impact_for_symptom(symptom_query: str) -> str:
    """
    Search the Knowledge Graph for known issue patterns matching the symptom,
    then return which machines across the fleet have been affected by each pattern.
    Use this to determine if a failure is isolated to one machine or is a fleet-wide recurrence.
    Returns: symptom name, description, affected machines list, machine families, and representative intervention IDs.
    """
    query_vector = _embed_text(symptom_query)
    try:
        search_result = QDRANT_CLIENT.query_points(
            collection_name="known_issues",
            query=query_vector,
            using="text-embedding-3-small",
            limit=3,
        ).points
    except Exception as e:
        return f"Error querying Knowledge Graph: {e}"

    if not search_result:
        return "No known fleet-wide patterns found for this symptom."

    output = []
    for hit in search_result:
        p = hit.payload
        machines = p.get("affected_machines", [])
        families = p.get("affected_machine_families", [])
        rep_ids = p.get("representative_intervention_ids", [])
        block = (
            f"Pattern: {p.get('symptom_name', 'N/A')}\n"
            f"Description: {p.get('description', 'N/A')}\n"
            f"Affected machines ({len(machines)}): {', '.join(machines) if machines else 'none recorded'}\n"
            f"Machine families: {', '.join(families) if families else 'N/A'}\n"
            f"Representative interventions: {', '.join(rep_ids) if rep_ids else 'none'}"
        )
        output.append(block)

    return "\n---\n".join(output)
