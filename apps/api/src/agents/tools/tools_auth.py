# -*- coding: utf-8 -*-
from langchain_core.tools import tool
from datetime import datetime, timedelta
from qdrant_client import models
import json
import uuid

from openai import OpenAI
from qdrant_client import QdrantClient
from sqlalchemy import create_engine, text
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
import pandas as pd
from agents.core.auth import authorize_tool
from langgraph.prebuilt.tool_node import ToolRuntime
from core.config import config

PG_URL = config.PG_URL.replace("postgresql://", "postgresql+psycopg://")
pg_engine = create_engine(PG_URL)

OPENAI_CLIENT = OpenAI()
QDRANT_CLIENT = QdrantClient(url=config.QDRANT_URL)

# Models & Collections
CM_COLLECTION = "cm_interventions_hybrid"
PROC_COLLECTION = "procedures_hybrid"
EMBEDDING_MODEL = "text-embedding-3-small"
KEYWORD_MODEL = "bm25"
GENERATION_MODEL = "gpt-5.4-nano"


# =========================================================
# PYDANTIC MODELS FOR LLM-STRUCTURED OUTPUT
# =========================================================


class SummaryResponse(BaseModel):
    overview: str


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
            "NEVER include machine IDs or machine names -- those belong in affected_machines. "
            "Good: 'Coil Cooling Flow Fault'. Bad: 'Coil Cooling Flow Fault - IH-300'. "
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


_llm = ChatOpenAI(model=GENERATION_MODEL)
_llm_overview = _llm.with_structured_output(SummaryResponse)
_llm_known_issue = _llm.with_structured_output(KnownIssue)


# =========================================================
# DB TABLE INITIALIZATION
# =========================================================


def _initialize_known_case_templates_table():
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS maintenance.known_case_templates (
            template_id UUID PRIMARY KEY,
            symptom_name VARCHAR(255) NOT NULL,
            description TEXT,
            root_causes JSONB,
            affected_machines TEXT[],
            affected_machine_families TEXT[],
            representative_intervention_ids TEXT[],
            created_at TIMESTAMP,
            created_by_agent VARCHAR(100),
            validation_status VARCHAR(50),
            validation_issues TEXT[],
            validation_date TIMESTAMP,
            created_at_idx TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        with pg_engine.connect() as conn:
            conn.execute(create_table_query)
            conn.commit()
    except Exception as e:
        print(f"Table initialization error (may already exist): {e}")


def _initialize_confirmed_rca_cases_table():
    create_table_query = text("""
        CREATE TABLE IF NOT EXISTS maintenance.confirmed_rca_cases (
            case_id UUID PRIMARY KEY,
            machine VARCHAR(100),
            symptom VARCHAR(255),
            diagnosed_root_cause TEXT,
            actual_root_cause TEXT,
            investigation_steps TEXT,
            diagnosis_accuracy BOOLEAN,
            created_at TIMESTAMP
        )
    """)
    try:
        with pg_engine.connect() as conn:
            conn.execute(create_table_query)
            conn.commit()
    except Exception as e:
        print(f"Table initialization error (may already exist): {e}")


_initialize_known_case_templates_table()
_initialize_confirmed_rca_cases_table()

# =========================================================
# SHARED HELPERS
# =========================================================


def _embed_text(text: str) -> list[float]:
    response = OPENAI_CLIENT.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL,
    )

    return response.data[0].embedding


def _format_cm_context(results: list[dict]) -> str:
    context = ""

    for result in results:
        payload = result["payload"]

        intervention_id = payload.get("id", "N/A")

        context += (
            f"[SOURCE: {intervention_id}]\n"
            f"Machine: {payload.get('machine', 'N/A')}\n"
            f"Date: {payload.get('date_start', 'N/A')}\n"
            f"Summary: {payload.get('summary', 'N/A')}\n" + "-" * 40 + "\n"
        )

    return context


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


def _format_proc_context(results: list[dict]) -> str:
    context = ""

    for result in results:
        payload = result["payload"]

        file_name = payload.get("file_name", "N/A")

        page_number = (
            payload.get("page_number")
            or payload.get("page")
            or payload.get("page_num")
            or "N/A"
        )

        chunk_number = payload.get("chunk_number", "N/A")

        if page_number == "N/A":
            source_ref = f"PROC_REF:{file_name}:chunk#{chunk_number}"
        else:
            source_ref = f"PROC_REF:{file_name}:{page_number}:chunk#{chunk_number}"

        context += (
            f"[SOURCE: {source_ref}]\n"
            f"File: {file_name}\n"
            f"Section: {payload.get('section_title', 'N/A')}\n"
            f"Context: {payload.get('context', 'N/A')}\n"
            f"Text: {payload.get('text', 'N/A')}\n" + "-" * 40 + "\n"
        )

    return context


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

    expanded.sort(
        key=lambda x: (
            x["payload"]["file_name"],
            x["payload"]["chunk_number"],
        )
    )

    return expanded


# =========================================================
# SENSOR FUNCTIONS
# =========================================================


def get_sensor_catalog(machine: str) -> str:
    query = text("""
        SELECT
            sensor_id,
            tag,
            sensor_name,
            unit,
            nominal_value,
            warn_lo,
            warn_hi,
            fault_correlation,
            active
        FROM maintenance.sensor_catalog
        WHERE machine = :machine
        ORDER BY tag
    """)

    with pg_engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"machine": machine},
        )

    if df.empty:
        return f"No sensors found for machine '{machine}'."

    return df.to_markdown(index=False)


def get_sensor_readings(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
) -> str:
    if tag:
        query = text("""
            SELECT
                timestamp,
                tag,
                sensor_name,
                value,
                unit,
                status,
                warn_lo,
                warn_hi
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
            SELECT
                timestamp,
                tag,
                sensor_name,
                value,
                unit,
                status,
                warn_lo,
                warn_hi
            FROM maintenance.sensor_readings
            WHERE machine = :machine
              AND timestamp >= :start_date
              AND timestamp <= :end_date
            ORDER BY tag, timestamp
        """)

        params = {
            "machine": machine,
            "start_date": start_date,
            "end_date": end_date,
        }

    with pg_engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return (
            f"No readings found for machine "
            f"'{machine}' between {start_date} "
            f"and {end_date}."
        )

    return df.to_markdown(index=False)


def get_sensor_timeline(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str,
) -> str:
    query = text("""
        SELECT
            timestamp,
            tag,
            sensor_name,
            value,
            unit,
            status,
            warn_lo,
            warn_hi
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
        df = pd.read_sql(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return f"No readings found for {tag} on {machine}."

    df["value_prev"] = df["value"].shift(1)

    df["delta"] = df["value"] - df["value_prev"]

    df["trend"] = df["delta"].apply(lambda x: "↑" if x > 0 else ("↓" if x < 0 else "→"))

    anomalies = df[df["status"].isin(["WARNING", "CRITICAL"])]

    summary = ""

    if not anomalies.empty:
        first_anomaly = anomalies.iloc[0]

        summary += (
            f"\nFirst threshold breach: "
            f"{first_anomaly['timestamp']} "
            f"(value={first_anomaly['value']}, "
            f"status={first_anomaly['status']})\n"
        )

    display_df = df[
        [
            "timestamp",
            "tag",
            "sensor_name",
            "value",
            "unit",
            "trend",
            "status",
        ]
    ]

    return (
        f"**Sensor Timeline for {tag}:**\n"
        f"{summary}\n"
        f"{display_df.to_markdown(index=False)}"
    )


def get_threshold_events(
    machine: str,
    timestamp_start: str,
    timestamp_end: str,
) -> str:
    query = text("""
        SELECT
            timestamp,
            tag,
            sensor_name,
            value,
            unit,
            status,
            warn_lo,
            warn_hi
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
        df = pd.read_sql(
            query,
            conn,
            params=params,
        )

    if df.empty:
        return f"No threshold breaches found for {machine}."

    critical_count = len(df[df["status"] == "CRITICAL"])

    warning_count = len(df[df["status"] == "WARNING"])

    summary = f"CRITICAL={critical_count}, WARNING={warning_count}"

    return f"**Threshold Events Summary**\n{summary}\n\n{df.to_markdown(index=False)}"


def get_remaining_life(machine: str) -> str:
    query = text("""
        SELECT
            component_id,
            component_name,
            condition,
            current_hours,
            remaining_hours,
            remaining_pct,
            unit_cost_eur,
            last_inspection,
            next_inspection,
            notes
        FROM maintenance.remaining_life
        WHERE machine = :machine
        ORDER BY remaining_pct ASC
    """)

    with pg_engine.connect() as conn:
        df = pd.read_sql(
            query,
            conn,
            params={"machine": machine},
        )

    if df.empty:
        return f"No component life data found for '{machine}'."

    return df.to_markdown(index=False)


# =========================================================
# FILTER BUILDER
# =========================================================


def _build_filters(
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    mandatory_filters: dict | None = None,
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

    # Machine prefix
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

    # Mandatory filters
    if mandatory_filters:
        for key, value in mandatory_filters.items():
            if isinstance(value, list):
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

    if not conditions:
        return None

    return models.Filter(must=conditions)


# =========================================================
# CM RETRIEVAL
# =========================================================


def _retrieve_cm(
    query: str,
    top_k: int = 5,
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    mandatory_filters: dict | None = None,
):
    query_vector = _embed_text(query)

    query_filter = _build_filters(
        machine=machine,
        machine_prefix=machine_prefix,
        date_start=date_start,
        date_end=date_end,
        mandatory_filters=mandatory_filters,
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


# =========================================================
# PROCEDURE RETRIEVAL
# =========================================================


def _retrieve_procedures(
    query: str,
    top_k: int = 5,
    file_name: str | None = None,
    contains_table: bool | None = None,
    expand_window: bool = True,
    mandatory_filters: dict | None = None,
):
    query_vector = _embed_text(query)

    query_filter = _build_proc_filter(
        file_name=file_name,
        contains_table=contains_table,
    )

    # Add mandatory filters
    if mandatory_filters:
        extra_conditions = []

        for key, value in mandatory_filters.items():
            if isinstance(value, list):
                extra_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                extra_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

        if query_filter:
            query_filter.must.extend(extra_conditions)
        else:
            query_filter = models.Filter(must=extra_conditions)

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


# =========================================================
# TOOLS
def _validate_machine_access_auth(
    machine: str, machine_type: str | None, mandatory_filters: dict | None
) -> str | None:
    """Validate machine access against workspace filters. Returns error if unauthorized, None if authorized."""
    # Use passed filters, or fall back to context variable
    filters = mandatory_filters
    if not filters:
        try:
            from agents.core.auth import get_workspace_context
            from agents.utils.workspace import load_workspace

            workspace_id = get_workspace_context()
            if workspace_id:
                workspace = load_workspace(workspace_id)
                filters = getattr(workspace, "filters", {})
        except (ImportError, AttributeError):
            pass

    if not filters:
        return None

    filter_type = filters.get("machine_type")
    if filter_type and machine_type and machine_type != filter_type:
        return f"✗ Access denied: Machine '{machine}' is a {machine_type}, but workspace is restricted to {filter_type}."
    return None


# =========================================================
@tool
@authorize_tool("check_machine_exists")
def check_machine_exists(
    machine: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Validate whether a machine ID exists in the system and return peer machines of the same type.

    Use FIRST whenever a machine ID is provided in user input.
    Blocks all downstream tools until machine is validated.

    Checks:
    - maintenance interventions database
    - sensor catalog (fallback)

    Returns:
    - confirmation if machine exists with metadata
    - peer machines of the same type
    - or error message if not found
    """

    query = text("""
        SELECT machine,
               machine_type,
               COUNT(*) as intervention_count,
               MIN(date_start) as first_intervention,
               MAX(date_start) as last_intervention
        FROM maintenance.interventions
        WHERE machine = :machine
        GROUP BY machine, machine_type
    """)

    with pg_engine.connect() as conn:
        result = conn.execute(
            query,
            {"machine": machine},
        ).fetchone()

    if result:
        machine_type = result[1]

        # Validate access against workspace filters
        auth_error = _validate_machine_access_auth(
            machine, machine_type, mandatory_filters
        )
        if auth_error:
            return auth_error

        peers_query = text("""
            SELECT DISTINCT machine
            FROM maintenance.interventions
            WHERE machine_type = :machine_type
              AND machine != :machine
            ORDER BY machine
        """)

        with pg_engine.connect() as conn:
            peers = [
                row[0]
                for row in conn.execute(
                    peers_query,
                    {
                        "machine_type": machine_type,
                        "machine": machine,
                    },
                ).fetchall()
            ]

        peers_str = (
            f"- Peer machines (same type '{machine_type}'): "
            f"{', '.join(peers)}\n"
            f"  → Use these machine IDs to broaden graph "
            f"or CM queries when looking for fleet-wide patterns."
            if peers
            else f"- No other machines of type '{machine_type}' found."
        )

        return (
            f"✓ Machine '{machine}' exists in database.\n"
            f"- Machine type: {machine_type}\n"
            f"- Interventions recorded: {result[2]}\n"
            f"- First intervention: {result[3]}\n"
            f"- Last intervention: {result[4]}\n"
            f"{peers_str}"
        )

    query_sensors = text("""
        SELECT DISTINCT machine
        FROM maintenance.sensor_catalog
        WHERE machine = :machine
        LIMIT 1
    """)

    with pg_engine.connect() as conn:
        sensor_result = conn.execute(
            query_sensors,
            {"machine": machine},
        ).fetchone()

    if sensor_result:
        return f"✓ Machine '{machine}' found in sensor catalog (no interventions yet)."

    return f"✗ Machine '{machine}' not found in database. Please verify the machine ID."


@tool
@authorize_tool("list_available_machines")
def list_available_machines(
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    List all machines with recorded maintenance interventions.

    Use when:
    - user does not specify a machine
    - user needs valid machine IDs
    - validating available fleet inventory

    Returns:
    Alphabetically sorted list of machine IDs.
    """

    query = text("""
        SELECT DISTINCT machine
        FROM maintenance.interventions
        ORDER BY machine
    """)

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn)

    if df.empty:
        return "No machines found in database."

    machines = sorted(df["machine"].unique().tolist())

    return "**Available Machines:**\n" + "\n".join(f"- {m}" for m in machines)


@tool
@authorize_tool("get_current_date")
def get_current_date(
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Return the current system date (YYYY-MM-DD).

    Use for:
    - anchoring relative date calculations
    - downstream temporal queries
    """

    return datetime.now().strftime("%Y-%m-%d")


@tool
@authorize_tool("calculate_date_window")
def calculate_date_window(
    reference_date: str,
    days_back: int,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Compute a relative date window from a reference date.

    Inputs:
    - reference_date: YYYY-MM-DD
    - days_back: number of days to look back

    Returns:
    JSON with:
    - start_date
    - end_date
    - label
    - days_span
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
@authorize_tool("get_formatted_cm_context")
def get_formatted_cm_context(
    query: str,
    top_k: int = 5,
    machine: str | None = None,
    machine_prefix: str | None = None,
    date_start: str | None = None,
    date_end: str | None = None,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve and format CM (Corrective Maintenance) context documents.

    This tool performs semantic retrieval over maintenance logs and returns
    formatted contextual information relevant to the provided query.

    Args:
        query:
            Natural language query describing the issue, symptom, or event.
        top_k:
            Maximum number of retrieved records to return.
        machine:
            Exact machine identifier to filter results.
        machine_prefix:
            Prefix-based machine filtering.
        date_start:
            Start date for filtering maintenance events.
        date_end:
            End date for filtering maintenance events.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        A formatted string containing retrieved CM context entries.
    """

    results = _retrieve_cm(
        query=query,
        top_k=top_k,
        machine=machine,
        machine_prefix=machine_prefix,
        date_start=date_start,
        date_end=date_end,
        mandatory_filters=mandatory_filters,
    )

    return _format_cm_context(results)


@tool
@authorize_tool("list_known_issue_categories")
def list_known_issue_categories(
    machine: str | None = None,
    limit: int = 15,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    List top-level known-issue categories from the knowledge graph.

    Use during the Orient stage to:
    - explore major failure patterns
    - understand fleet-wide issue categories
    - identify high-frequency failure modes

    Optional:
    - machine: filter categories touching a target machine
    - limit: max categories returned

    Returns:
    Ranked list with:
    - symptom_name
    - affected machine count
    - machine families
    """

    try:
        points, _ = QDRANT_CLIENT.scroll(
            collection_name="known_issues",
            limit=200,
            with_payload=True,
        )

    except Exception as e:
        return f"Error listing known-issue categories: {e}"

    if not points:
        return "No known-issue categories found in the graph."

    rows = []

    for p in points:
        payload = p.payload or {}

        machines = payload.get("affected_machines", []) or []

        if machine and machine not in machines:
            continue

        rows.append(
            {
                "symptom_name": payload.get(
                    "symptom_name",
                    "N/A",
                ),
                "n_machines": len(machines),
                "families": payload.get(
                    "affected_machine_families",
                    [],
                )
                or [],
                "machines": machines,
            }
        )

    if not rows:
        return (
            f"No known-issue categories found"
            f"{' for machine ' + machine if machine else ''}."
        )

    rows.sort(
        key=lambda r: r["n_machines"],
        reverse=True,
    )

    rows = rows[:limit]

    header = "**Known-issue categories"

    header += f" touching {machine}" if machine else " (fleet-wide)"

    header += f"** (top {len(rows)} by fleet impact)\n\n"

    output = header

    for r in rows:
        families = ", ".join(r["families"]) if r["families"] else "N/A"

        output += (
            f"- **{r['symptom_name']}** "
            f"— {r['n_machines']} machine(s); "
            f"families: {families}\n"
        )

    return output


@tool
@authorize_tool("get_formatted_procedure_context")
def get_formatted_procedure_context(
    query: str,
    top_k: int = 5,
    file_name: str | None = None,
    contains_table: bool | None = None,
    expand_window: bool = True,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve and format procedural documentation context.

    This tool performs semantic search over procedures, SOPs,
    troubleshooting guides, and technical documentation.

    Args:
        query:
            Natural language query describing the procedure or issue.
        top_k:
            Maximum number of retrieved chunks.
        file_name:
            Restrict retrieval to a specific document.
        contains_table:
            Filter chunks containing tabular data.
        expand_window:
            Whether to include neighboring chunks for additional context.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        A formatted string containing relevant procedure excerpts.
    """

    results = _retrieve_procedures(
        query=query,
        top_k=top_k,
        file_name=file_name,
        contains_table=contains_table,
        expand_window=expand_window,
        mandatory_filters=mandatory_filters,
    )

    return _format_proc_context(results)


@tool
@authorize_tool("get_sensor_catalog_tool")
def get_sensor_catalog_tool(
    machine: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
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
@authorize_tool("get_remaining_life_tool")
def get_remaining_life_tool(
    machine: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve remaining useful life (RUL) estimates for machine components.

    Use when: assessing wear-related contribution to failures.

    Returns: component-level RUL and condition metrics, sorted by most critical first.
    """
    return get_remaining_life(machine)


@tool
@authorize_tool("get_sensor_readings_tool")
def get_sensor_readings_tool(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str | None = None,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve aggregated or raw sensor readings for a machine.

    Args:
        machine:
            Machine identifier.
        start_date:
            Start datetime for retrieval.
        end_date:
            End datetime for retrieval.
        tag:
            Optional sensor tag filter.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        Sensor readings as a serialized string payload.
    """

    return get_sensor_readings(
        machine=machine,
        start_date=start_date,
        end_date=end_date,
        tag=tag,
    )


@tool
@authorize_tool("get_sensor_timeline_tool")
def get_sensor_timeline_tool(
    machine: str,
    start_date: str,
    end_date: str,
    tag: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve time-series sensor timeline data.

    Useful for trend analysis, anomaly inspection,
    and threshold event investigations.

    Args:
        machine:
            Machine identifier.
        start_date:
            Start datetime for timeline retrieval.
        end_date:
            End datetime for timeline retrieval.
        tag:
            Sensor tag to retrieve.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        Time-series sensor data as a serialized string payload.
    """

    return get_sensor_timeline(
        machine=machine,
        start_date=start_date,
        end_date=end_date,
        tag=tag,
    )


@tool
@authorize_tool("get_threshold_events_tool")
def get_threshold_events_tool(
    machine: str,
    timestamp_start: str,
    timestamp_end: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve threshold crossing or alert events for a machine.

    Args:
        machine:
            Machine identifier.
        timestamp_start:
            Start timestamp for the search interval.
        timestamp_end:
            End timestamp for the search interval.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        Serialized threshold event records.
    """

    return get_threshold_events(
        machine=machine,
        timestamp_start=timestamp_start,
        timestamp_end=timestamp_end,
    )


@tool
@authorize_tool("find_similar_machines")
def find_similar_machines(
    machine: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Find machines similar to the target based on:
    - same family
    - co-occurring known-issue clusters

    Use during the Orient stage when broadening retrieval
    beyond the target machine.

    Returns:
    - family peers
    - graph co-occurrence peers
    - cluster names
    """

    try:
        points, _ = QDRANT_CLIENT.scroll(
            collection_name="known_issues",
            limit=500,
            with_payload=True,
        )

    except Exception as e:
        return f"Error querying graph: {e}"

    target_families: set[str] = set()

    graph_peers: dict[str, list[str]] = {}

    family_peers: set[str] = set()

    for p in points:
        payload = p.payload or {}

        machines = set(payload.get("affected_machines", []) or [])

        families = set(payload.get("affected_machine_families", []) or [])

        symptom_name = payload.get(
            "symptom_name",
            "N/A",
        )

        if machine in machines:
            target_families.update(families)

            for m in machines:
                if m == machine:
                    continue

                graph_peers.setdefault(m, [])

                if symptom_name not in graph_peers[m]:
                    graph_peers[m].append(symptom_name)

    if target_families:
        for p in points:
            payload = p.payload or {}

            machines = (
                payload.get(
                    "affected_machines",
                    [],
                )
                or []
            )

            families = set(
                payload.get(
                    "affected_machine_families",
                    [],
                )
                or []
            )

            if families & target_families:
                for m in machines:
                    if m != machine:
                        family_peers.add(m)

    if not graph_peers and not family_peers:
        return f"No similar machines found for '{machine}' in the knowledge graph."

    output = f"**Similar machines to {machine}**\n\n"

    if target_families:
        output += f"Target families: {', '.join(sorted(target_families))}\n\n"

    if graph_peers:
        output += "### Co-occurring in same known-issue clusters\n"

        for m, clusters in sorted(
            graph_peers.items(),
            key=lambda kv: -len(kv[1]),
        ):
            output += f"- **{m}** — shared clusters: {', '.join(clusters)}\n"

        output += "\n"

    family_only = family_peers - set(graph_peers.keys())

    if family_only:
        output += "### Same machine family (no shared incidents yet)\n"

        for m in sorted(family_only):
            output += f"- {m}\n"

    return output


@tool
@authorize_tool("list_procedure_sections")
def list_procedure_sections(
    machine: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    List section titles and fault-code headers from troubleshooting manuals.

    Use FIRST during the Orient stage before retrieving procedure chunks.

    Helps:
    - identify available troubleshooting sections
    - choose the best file_name
    - create focused procedure queries

    Returns:
    Structured list of:
    - file_name
    - section_title
    - chunk ranges
    """

    try:
        points, _ = QDRANT_CLIENT.scroll(
            collection_name=PROC_COLLECTION,
            limit=500,
            with_payload=True,
        )

    except Exception as e:
        return f"Error listing procedure sections: {e}"

    if not points:
        return "No procedure documents found."

    # =====================================================
    # GROUP SECTIONS
    # =====================================================

    by_section: dict[tuple[str, str], dict] = {}

    for p in points:
        payload = p.payload or {}

        fname = payload.get(
            "file_name",
            "unknown",
        )

        section = (payload.get("section_title") or "").strip()

        chunk_num = payload.get("chunk_number")

        if not section or chunk_num is None:
            continue

        key = (fname, section)

        if key not in by_section:
            by_section[key] = {
                "min_chunk": chunk_num,
                "max_chunk": chunk_num,
            }

        else:
            by_section[key]["min_chunk"] = min(
                by_section[key]["min_chunk"],
                chunk_num,
            )

            by_section[key]["max_chunk"] = max(
                by_section[key]["max_chunk"],
                chunk_num,
            )

    if not by_section:
        return "Procedure documents exist but no section titles were found."

    # =====================================================
    # PRIORITIZE MATCHES
    # =====================================================

    exact_matches = {}

    prefix_matches = {}

    other_files = {}

    for (fname, section), meta in by_section.items():
        if fname == f"{machine}_Troubleshooting_Procedures":
            exact_matches.setdefault(fname, []).append((section, meta))

        elif machine in fname or fname.startswith(machine[:3]):
            prefix_matches.setdefault(fname, []).append((section, meta))

        else:
            other_files.setdefault(fname, []).append((section, meta))

    # =====================================================
    # BUILD OUTPUT
    # =====================================================

    output = f"**Procedure sections for {machine}**\n\n"

    # Exact matches
    for fname, sections in exact_matches.items():
        output += f"### {fname} (EXACT MATCH)\n"

        for section, meta in sorted(
            sections,
            key=lambda x: x[1]["min_chunk"],
        ):
            output += (
                f"- **{section}** (chunks {meta['min_chunk']}-{meta['max_chunk']})\n"
            )

        output += "\n"

    # Prefix matches
    for fname, sections in prefix_matches.items():
        output += f"### {fname}\n"

        for section, meta in sorted(
            sections,
            key=lambda x: x[1]["min_chunk"],
        ):
            output += (
                f"- **{section}** (chunks {meta['min_chunk']}-{meta['max_chunk']})\n"
            )

        output += "\n"

    # Other files
    if other_files:
        output += "**Other procedure files** (less relevant):\n"

        for fname, sections in other_files.items():
            output += f"- {fname}: {len(sections)} sections\n"

    return output


def _get_intervention_detail_impl(intervention_id: str) -> str:
    query_id = (
        intervention_id
        if intervention_id.startswith("INT-")
        else f"INT-{intervention_id}"
    )

    query = text("""
        SELECT
            id, machine, machine_type, date_start, date_end, duration_min,
            fault_description, intervention_type, severity, technician,
            supervisor, subsystem, fault_code, comments
        FROM maintenance.interventions
        WHERE id = :intervention_id
    """)

    with pg_engine.connect() as conn:
        result = conn.execute(query, {"intervention_id": query_id}).fetchone()

    if not result:
        return f"Intervention '{intervention_id}' not found."

    columns = [
        "ID",
        "Machine",
        "Machine Type",
        "Start Date",
        "End Date",
        "Duration (min)",
        "Fault Description",
        "Type",
        "Severity",
        "Technician",
        "Supervisor",
        "Subsystem",
        "Fault Code",
        "Comments",
    ]

    output = f"**Intervention Detail: {intervention_id}**\n\n"
    for col, val in zip(columns, result):
        if val is not None:
            output += f"**{col}:** {val}\n"

    return output


@tool
@authorize_tool("get_intervention_detail")
def get_intervention_detail(
    intervention_id: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve full details of a specific corrective maintenance intervention.

    Use AFTER finding a relevant intervention in CM search results to expand
    the summary into actionable details:
    - parts replaced
    - duration
    - outcomes
    - technician notes

    Inputs:
    - intervention_id: ID from CM search result

    Returns:
    Complete intervention record with all metadata.
    """

    return _get_intervention_detail_impl(intervention_id)


@tool
@authorize_tool("query_known_issues_graph")
def query_known_issues_graph(
    query: str,
    machine: str | None = None,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Search the known issues knowledge graph using semantic similarity.

    This tool embeds the query and retrieves the most relevant
    known issues from a vector database.

    Args:
        query:
            Natural language description of the symptom or issue.
        machine:
            Optional machine filter to prioritize related issues.
        mandatory_filters:
            Additional structured metadata filters.
        runtime:
            Injected runtime context (internal use only).

    Returns:
        A formatted summary of the most relevant known issues.
    """

    query_vector = _embed_text(query)

    should_conditions = []

    if machine:
        should_conditions.append(
            models.FieldCondition(
                key="affected_machines",
                match=models.MatchValue(value=machine),
            )
        )

    must_conditions = []

    if mandatory_filters:
        for key, value in mandatory_filters.items():
            if isinstance(value, list):
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchAny(any=value),
                    )
                )
            else:
                must_conditions.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value),
                    )
                )

    query_filter = models.Filter(
        must=must_conditions,
        should=should_conditions,
    )

    search_result = QDRANT_CLIENT.query_points(
        collection_name="known_issues",
        query=query_vector,
        query_filter=query_filter,
        limit=3,
    ).points

    if not search_result:
        return "No known issues found."

    output = []

    for hit in search_result:
        payload = hit.payload

        output.append(
            f"""
Symptom: {payload.get("symptom_name")}
Description: {payload.get("description")}
Affected Machines: {payload.get("affected_machines")}
"""
        )

    return "\n---\n".join(output)


@tool
@authorize_tool("get_recent_formatted_cm_context")
def get_recent_formatted_cm_context(
    query: str,
    machine: str,
    top_k: int = 5,
    days_span: int = 30,
    date_end: str | None = None,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
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
        mandatory_filters=mandatory_filters,
    )

    return (
        f"**Recent Interventions (Last {days_span} days until {date_end}):**\n"
        + _format_cm_context(results)
    )


@tool
@authorize_tool("get_long_formatted_cm_context")
def get_long_formatted_cm_context(
    query: str,
    machine: str | None = None,
    machine_prefix: str | None = None,
    top_k: int = 10,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
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
        mandatory_filters=mandatory_filters,
    )

    return "**Historical Interventions (Full History):**\n" + _format_cm_context(
        results
    )


@tool
@authorize_tool("get_fleet_impact_for_symptom")
def get_fleet_impact_for_symptom(
    symptom_query: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Search the Knowledge Graph for known issue patterns matching the symptom,
    then return which machines across the fleet have been affected by each pattern.

    Use to determine if a failure is isolated to one machine or is a fleet-wide recurrence.

    Returns: symptom name, description, affected machines list, machine families,
    and representative intervention IDs.
    """
    query_vector = _embed_text(symptom_query)

    try:
        search_result = QDRANT_CLIENT.query_points(
            collection_name="known_issues",
            query=query_vector,
            using=EMBEDDING_MODEL,
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


@tool
@authorize_tool("get_sensor_anomaly_summary")
def get_sensor_anomaly_summary(
    machine: str,
    start_date: str,
    end_date: str,
    top_n: int = 5,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
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
        [
            "tag",
            "sensor_name",
            "has_critical",
            "event_count",
            "first_breach",
            "avg_value",
            "warn_lo",
            "warn_hi",
        ]
    ].copy()
    display_df.columns = [
        "Tag",
        "Sensor",
        "Critical?",
        "Events",
        "First Breach",
        "Avg Value",
        "Warn Low",
        "Warn High",
    ]
    display_df["Critical?"] = display_df["Critical?"].map({1: "Yes", 0: "No"})

    return f"**Sensor Anomaly Summary (Top {top_n}):**\n{display_df.to_markdown(index=False)}"


@tool
@authorize_tool("get_known_case_templates")
def get_known_case_templates(
    machine: str | None = None,
    symptom: str | None = None,
    limit: int = 5,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Retrieve saved known case templates for RCA agent reference.

    Use when: diagnosing a machine issue to quickly find similar past patterns with proven resolutions.

    Args:
        machine: Optional - filter by affected machine ID (e.g., 'HX-200')
        symptom: Optional - search by symptom name keyword
        limit: Maximum templates to return (default 5)

    Returns: Formatted list of known case templates with metadata and intervention references.
    """
    try:
        conditions = []
        if machine:
            conditions.append(f"'{machine}' = ANY(affected_machines)")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        query = text(f"""
            SELECT
                template_id, symptom_name, description, root_causes,
                affected_machines, representative_intervention_ids,
                created_at, created_by_agent
            FROM maintenance.known_case_templates
            WHERE {where_clause}
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        with pg_engine.connect() as conn:
            df = pd.read_sql(query, conn, params={"limit": limit})

        if df.empty:
            filters = []
            if machine:
                filters.append(f"machine '{machine}'")
            if symptom:
                filters.append(f"symptom '{symptom}'")
            filter_str = " and ".join(filters) if filters else "any"
            return f"No known case templates found for {filter_str}."

        output = f"**Known Case Templates ({len(df)} found):**\n\n"
        for idx, row in df.iterrows():
            affected_machines = row["affected_machines"]
            rep_ids = row["representative_intervention_ids"]
            output += f"### Template {idx + 1}: {row['symptom_name']}\n"
            output += f"**ID:** `{row['template_id']}`\n"
            output += f"**Description:** {row['description']}\n"
            output += f"**Affected Machines:** {', '.join(affected_machines) if affected_machines else 'N/A'}\n"
            output += f"**Evidence Base:** {len(rep_ids)} interventions\n"
            output += f"**Created:** {row['created_at']}\n\n"

        return output

    except Exception as e:
        return f"Error retrieving known case templates: {str(e)}"


@tool
@authorize_tool("save_confirmed_rca_case")
def save_confirmed_rca_case(
    machine: str,
    symptom: str,
    diagnosed_root_cause: str,
    actual_root_cause: str,
    investigation_steps: str,
    diagnosis_accuracy: bool,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    Save confirmed RCA diagnosis as a case for future reference.

    Use after operator confirms the diagnosis was correct.

    Args:
        machine: Machine ID (e.g., 'HX-200')
        symptom: Initial symptom reported
        diagnosed_root_cause: What the RCA agent diagnosed
        actual_root_cause: What operator confirmed was the real cause
        investigation_steps: Summary of steps taken
        diagnosis_accuracy: True if diagnosed == actual, False if operator corrected

    Returns: dict with case_id and confirmation.
    """
    try:
        case_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        insert_query = text("""
            INSERT INTO maintenance.confirmed_rca_cases (
                case_id, machine, symptom, diagnosed_root_cause,
                actual_root_cause, investigation_steps, diagnosis_accuracy, created_at
            ) VALUES (
                :case_id, :machine, :symptom, :diagnosed_root_cause,
                :actual_root_cause, :investigation_steps, :diagnosis_accuracy, :created_at
            )
        """)

        params = {
            "case_id": case_id,
            "machine": machine,
            "symptom": symptom,
            "diagnosed_root_cause": diagnosed_root_cause,
            "actual_root_cause": actual_root_cause,
            "investigation_steps": investigation_steps,
            "diagnosis_accuracy": diagnosis_accuracy,
            "created_at": now,
        }

        with pg_engine.connect() as conn:
            conn.execute(insert_query, params)
            conn.commit()

        return {
            "status": "success",
            "case_id": case_id,
            "message": f"RCA case saved with ID: {case_id}",
            "accuracy": "✓ Diagnosis correct"
            if diagnosis_accuracy
            else "⚠ Diagnosis corrected by operator",
        }

    except Exception as e:
        return {"status": "error", "message": f"Failed to save RCA case: {str(e)}"}


@tool
@authorize_tool("summarize_intervention")
def summarize_intervention(
    intervention_id: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    Generate a concise 2-3 phrase overview of a specific maintenance intervention.

    Use when: you need a quick summary of what was done in a particular intervention
    without full details.

    Args:
        intervention_id: The intervention ID (e.g., INT-001) to summarize.

    Returns: A formatted summary with INT-ID and overview for template building.
    """
    intervention_history = _get_intervention_detail_impl(intervention_id)

    prompt = f"""Describe the intervention as an overview of max 2-3 phrases.

Here is the intervention history:
{intervention_history}
"""

    try:
        response = _llm_overview.invoke(prompt)
    except Exception as e:
        return f"[INT: {intervention_id}] Error: {e}"

    return f"[INT: {intervention_id}] {response.overview}"


@tool
@authorize_tool("build_known_case_template")
def build_known_case_template(
    summaries: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    Process intervention summaries and create a normalized "Known Issue" template.

    Use when consolidating multiple similar interventions into a reusable knowledge base entry.

    Args:
        summaries: Multiple intervention summaries to normalize (with [INT: ID] markers).

    Returns: Structured dict with symptom_name, description, root_causes, affected_machines,
    families, and representative_intervention_ids.
    """
    prompt = f"""Your goal is to process intervention summaries and create a normalized Known Issue record.

CRITICAL: Extract ALL INT-IDs marked as [INT: XXX-XXXX] from the summaries. Include EVERY INT-ID found.

Output must be JSON:
- symptom_name: normalized failure name (no machine IDs)
- description: 2-4 sentences about the failure pattern
- root_causes: list of objects with root_cause and actions array
- affected_machines: list of machine IDs from summaries
- affected_machine_families: inferred machine types (CNC, Oven, etc)
- representative_intervention_ids: COMPLETE list of ALL INT-IDs found in the summaries (do not omit any)

Example for symptom_name: use "Coil Cooling Flow Fault", not "IH-300 cooling flow fault".

Process these summaries:

{summaries}

Return valid JSON only. Ensure representative_intervention_ids contains ALL INT-IDs from the input."""

    try:
        response = _llm_known_issue.invoke(prompt)
    except Exception as e:
        return f"Error: {e}"

    return response


@tool
@authorize_tool("save_known_case_template")
def save_known_case_template(
    symptom_name: str,
    description: str,
    root_causes_json: str,
    affected_machines: str,
    affected_machine_families: str,
    representative_intervention_ids: str,
    created_by_agent: str,
    validation_status: str = "valid",
    validation_issues_json: str = "[]",
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> dict:
    """
    Save a validated known case template to the database with full traceability.

    Use after template validation passes to persist it for RCA agent retrieval.

    Args:
        symptom_name: Normalized failure symptom name
        description: Description of the issue pattern
        root_causes_json: JSON string of root causes (list of {root_cause, actions})
        affected_machines: Comma-separated or JSON list of machine IDs
        affected_machine_families: Comma-separated or JSON list of machine families
        representative_intervention_ids: Comma-separated or JSON list of INT-IDs
        created_by_agent: Name of the agent creating this template
        validation_status: "valid" or "invalid"
        validation_issues_json: JSON list of validation issues

    Returns: dict with template_id, status, and metadata.
    """
    try:
        template_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        root_causes = (
            json.loads(root_causes_json)
            if isinstance(root_causes_json, str)
            else root_causes_json
        )
        validation_issues = (
            json.loads(validation_issues_json)
            if isinstance(validation_issues_json, str)
            else validation_issues_json
        )

        if isinstance(affected_machines, str):
            try:
                affected_machines = json.loads(affected_machines)
            except json.JSONDecodeError:
                affected_machines = [m.strip() for m in affected_machines.split(",")]

        if isinstance(affected_machine_families, str):
            try:
                affected_machine_families = json.loads(affected_machine_families)
            except json.JSONDecodeError:
                affected_machine_families = [
                    f.strip() for f in affected_machine_families.split(",")
                ]

        if isinstance(representative_intervention_ids, str):
            try:
                representative_intervention_ids = json.loads(
                    representative_intervention_ids
                )
            except json.JSONDecodeError:
                representative_intervention_ids = [
                    i.strip() for i in representative_intervention_ids.split(",")
                ]

        insert_query = text("""
            INSERT INTO maintenance.known_case_templates (
                template_id, symptom_name, description, root_causes,
                affected_machines, affected_machine_families,
                representative_intervention_ids, created_at, created_by_agent,
                validation_status, validation_issues, validation_date
            ) VALUES (
                :template_id, :symptom_name, :description, :root_causes,
                :affected_machines, :affected_machine_families,
                :representative_intervention_ids, :created_at, :created_by_agent,
                :validation_status, :validation_issues, :validation_date
            )
        """)

        params = {
            "template_id": template_id,
            "symptom_name": symptom_name,
            "description": description,
            "root_causes": json.dumps(root_causes),
            "affected_machines": affected_machines,
            "affected_machine_families": affected_machine_families,
            "representative_intervention_ids": representative_intervention_ids,
            "created_at": now,
            "created_by_agent": created_by_agent,
            "validation_status": validation_status,
            "validation_issues": validation_issues,
            "validation_date": now,
        }

        with pg_engine.connect() as conn:
            conn.execute(insert_query, params)
            conn.commit()

        return {
            "status": "success",
            "template_id": template_id,
            "symptom_name": symptom_name,
            "created_at": now,
            "affected_machines": affected_machines,
            "representative_intervention_ids": representative_intervention_ids,
            "message": f"Template saved successfully with ID: {template_id}",
        }

    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to save template: {str(e)}",
            "error": str(e),
        }


@tool
@authorize_tool("list_intervention_ids_by_date")
def list_intervention_ids_by_date(
    machine: str,
    start_date: str,
    end_date: str,
    mandatory_filters: dict | None = None,
    runtime: ToolRuntime = None,
) -> str:
    """
    List all intervention IDs for a machine within a date range.

    Use when: you need to retrieve all intervention IDs for a specific machine during a
    time period before processing or summarizing them individually.

    Args:
        machine: Machine ID (e.g., 'CNC-500')
        start_date: Start date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)
        end_date: End date in ISO format (YYYY-MM-DD or YYYY-MM-DDTHH:MM:SS)

    Returns: List of intervention IDs sorted chronologically with their dates.
    """
    query = text("""
        SELECT id, date_start
        FROM maintenance.interventions
        WHERE machine = :machine
          AND date_start >= :start_date
          AND date_start <= :end_date
        ORDER BY date_start DESC
    """)
    params = {
        "machine": machine,
        "start_date": start_date,
        "end_date": end_date,
    }

    with pg_engine.connect() as conn:
        df = pd.read_sql(query, conn, params=params)

    if df.empty:
        return f"No interventions found for machine '{machine}' between {start_date} and {end_date}."

    output = f"**Interventions for {machine} ({start_date} to {end_date}):**\n\n"
    for _, row in df.iterrows():
        output += f"- **{row['id']}** ({row['date_start']})\n"

    output += f"\n**Total: {len(df)} intervention(s)**"
    return output
