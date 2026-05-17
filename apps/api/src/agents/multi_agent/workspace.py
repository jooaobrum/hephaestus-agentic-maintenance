"""Workspace registry and file tools for sub-agents.

Each agent session gets an isolated directory keyed by thread_id.
File tools (read_file, write_file, edit_file) operate within that sandbox.
Large tool outputs are offloaded to workspace files instead of filling the context window.
"""

import json
from pathlib import Path

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from agents.multi_agent.config import WORKSPACE_BASE_DIR


class WorkspaceRegistry:
    """Maps thread_id → isolated workspace directory."""

    def __init__(self, prefix: str, seed_files: dict | None = None):
        self._prefix = prefix
        self._seed_files = seed_files or {}
        self._registry: dict[str, Path] = {}

    def get_or_create(self, thread_id: str) -> Path:
        if thread_id not in self._registry:
            ws = WORKSPACE_BASE_DIR / f"{self._prefix}_{thread_id[:8]}"
            ws.mkdir(parents=True, exist_ok=True)
            for fname, content in self._seed_files.items():
                p = ws / fname
                if not p.exists():
                    p.write_text(content)
            self._registry[thread_id] = ws
        return self._registry[thread_id]

    def get_path(self, thread_id: str) -> Path | None:
        return self._registry.get(thread_id)


_troubleshooting_registry = WorkspaceRegistry(
    "troubleshooting",
    {"hypotheses.md": "# Hypotheses\n\n*Not yet investigated.*\n"},
)
_summarizer_registry = WorkspaceRegistry(
    "summarizer",
    {"summaries.md": "# Intervention Summaries\n\n*No summaries yet.*\n"},
)


def get_troubleshooting_workspace(thread_id: str) -> Path:
    return _troubleshooting_registry.get_or_create(thread_id)


def get_summarizer_workspace(thread_id: str) -> Path:
    return _summarizer_registry.get_or_create(thread_id)


def _make_file_tools(registry: WorkspaceRegistry) -> list:
    """Return read_file / write_file / edit_file tools bound to a specific registry."""

    def _resolve(file_path: str, config: RunnableConfig) -> Path:
        ws = registry.get_or_create(config["configurable"]["thread_id"])
        return ws / file_path.lstrip("/")

    @tool("read_file")
    def read_file_impl(file_path: str, config: RunnableConfig) -> str:
        """Read a file from the workspace. file_path: virtual path e.g. /hypotheses.md. Returns numbered lines."""
        full = _resolve(file_path, config)
        if not full.exists():
            return f"File not found: {file_path}"
        lines = full.read_text().splitlines()
        return "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(lines))

    @tool("write_file")
    def write_file_impl(file_path: str, content: str, config: RunnableConfig) -> str:
        """Write a NEW file. Fails if already exists — use edit_file to update."""
        full = _resolve(file_path, config)
        if full.exists():
            return (
                f"Cannot write to {file_path}: already exists. Use edit_file instead."
            )
        full.write_text(content)
        return f"Written: {file_path}"

    @tool("edit_file")
    def edit_file_impl(
        file_path: str, old_string: str, new_string: str, config: RunnableConfig
    ) -> str:
        """Edit an existing file by replacing old_string with new_string (first match)."""
        full = _resolve(file_path, config)
        if not full.exists():
            return f"File not found: {file_path}"
        content = full.read_text()
        if old_string not in content:
            return f"String not found in {file_path}. Use read_file first."
        full.write_text(content.replace(old_string, new_string, 1))
        return f"Replaced in '{file_path}'"

    return [read_file_impl, write_file_impl, edit_file_impl]


TROUBLESHOOTING_FILE_TOOLS = _make_file_tools(_troubleshooting_registry)
SUMMARIZER_FILE_TOOLS = _make_file_tools(_summarizer_registry)


# ── Offloading helpers ────────────────────────────────────────────────────────


def _make_offloader(registry: WorkspaceRegistry):
    def _offload(content: str, vpath: str, config: RunnableConfig) -> str:
        try:
            ws = registry.get_or_create(config["configurable"]["thread_id"])
            full = ws / vpath.lstrip("/")
            full.write_text(content)
        except Exception:
            pass
        return content

    return _offload


_troubleshooting_offload = _make_offloader(_troubleshooting_registry)
_summarizer_offload = _make_offloader(_summarizer_registry)


# ── Offloading wrappers for troubleshooting tools ─────────────────────────────


def make_troubleshooting_offload_tools():
    from agents.multi_agent.tools import (
        get_formatted_procedure_context as _raw_proc_ctx,
        get_recent_formatted_cm_context as _raw_cm_recent,
        get_long_formatted_cm_context as _raw_cm_long,
        get_sensor_readings_tool as _raw_sensor_readings,
        get_sensor_timeline_tool as _raw_sensor_timeline,
        get_threshold_events_tool as _raw_threshold_events,
    )

    @tool
    def get_formatted_procedure_context(
        query: str,
        config: RunnableConfig,
        top_k: int = 5,
        file_name: str | None = None,
        contains_table: bool | None = None,
        expand_window: bool = True,
    ) -> str:
        """Retrieve troubleshooting procedure docs via hybrid search. Large outputs → /proc_context.md."""
        result = _raw_proc_ctx.func(
            query=query,
            top_k=top_k,
            file_name=file_name,
            contains_table=contains_table,
            expand_window=expand_window,
        )
        return _troubleshooting_offload(result, "/proc_context.md", config)

    @tool
    def get_recent_formatted_cm_context(
        query: str,
        machine: str,
        config: RunnableConfig,
        top_k: int = 5,
        days_span: int = 30,
        date_end: str | None = None,
    ) -> str:
        """Retrieve recent CM interventions (30-day window). Always provide date_end. Large outputs → /cm_recent.md."""
        result = _raw_cm_recent.func(
            query=query,
            machine=machine,
            top_k=top_k,
            days_span=days_span,
            date_end=date_end,
        )
        return _troubleshooting_offload(result, "/cm_recent.md", config)

    @tool
    def get_long_formatted_cm_context(
        query: str,
        config: RunnableConfig,
        machine: str | None = None,
        machine_prefix: str | None = None,
        top_k: int = 10,
    ) -> str:
        """Full-history CM interventions, no time filter. Large outputs → /cm_long.md."""
        result = _raw_cm_long.func(
            query=query, machine=machine, machine_prefix=machine_prefix, top_k=top_k
        )
        return _troubleshooting_offload(result, "/cm_long.md", config)

    @tool
    def get_sensor_readings_tool(
        machine: str,
        start_date: str,
        end_date: str,
        config: RunnableConfig,
        tag: str | None = None,
    ) -> str:
        """Raw sensor readings for a machine/time window. Large outputs → /sensor_readings.md."""
        result = _raw_sensor_readings.func(
            machine=machine, start_date=start_date, end_date=end_date, tag=tag
        )
        return _troubleshooting_offload(result, "/sensor_readings.md", config)

    @tool
    def get_sensor_timeline_tool(
        machine: str,
        start_date: str,
        end_date: str,
        tag: str,
        config: RunnableConfig,
    ) -> str:
        """Detailed time-series for one sensor with trend analysis. Large outputs → /sensor_timeline.md."""
        result = _raw_sensor_timeline.func(
            machine=machine, start_date=start_date, end_date=end_date, tag=tag
        )
        return _troubleshooting_offload(result, "/sensor_timeline.md", config)

    @tool
    def get_threshold_events_tool(
        machine: str,
        timestamp_start: str,
        timestamp_end: str,
        config: RunnableConfig,
    ) -> str:
        """All sensor threshold breaches (WARNING/CRITICAL) in a window. Large outputs → /threshold_events.md."""
        result = _raw_threshold_events.func(
            machine=machine,
            timestamp_start=timestamp_start,
            timestamp_end=timestamp_end,
        )
        return _troubleshooting_offload(result, "/threshold_events.md", config)

    return [
        get_formatted_procedure_context,
        get_recent_formatted_cm_context,
        get_long_formatted_cm_context,
        get_sensor_readings_tool,
        get_sensor_timeline_tool,
        get_threshold_events_tool,
    ]


# ── Offloading wrappers for summarizer tools ──────────────────────────────────


def make_summarizer_offload_tools():
    from agents.multi_agent.tools import (
        get_formatted_cm_context,
        get_known_case_templates,
    )

    @tool
    def get_cm_context_summarizer(
        query: str,
        config: RunnableConfig,
        top_k: int = 5,
        machine: str | None = None,
        machine_prefix: str | None = None,
        date_start: str | None = None,
        date_end: str | None = None,
    ) -> str:
        """Historical CM interventions by semantic search. Large outputs → /cm_history.md."""
        result = get_formatted_cm_context.func(
            query=query,
            top_k=top_k,
            machine=machine,
            machine_prefix=machine_prefix,
            date_start=date_start,
            date_end=date_end,
        )
        return _summarizer_offload(result, "/cm_history.md", config)

    @tool
    def get_known_case_templates_tool(
        config: RunnableConfig,
        symptom_query: str | None = None,
        machine: str | None = None,
    ) -> str:
        """Existing known case templates — check before building to detect duplicates. Large outputs → /known_templates.md."""
        result = get_known_case_templates.func(symptom=symptom_query, machine=machine)
        if isinstance(result, dict):
            result = json.dumps(result, indent=2)
        return _summarizer_offload(str(result), "/known_templates.md", config)

    @tool
    def validate_template(template_json: str) -> str:
        """Validate a built known-case template before saving. Returns VALID or blocking issues."""
        issues = []
        try:
            data = (
                json.loads(template_json)
                if isinstance(template_json, str)
                else template_json
            )
        except (json.JSONDecodeError, TypeError):
            return "INVALID: template_json is not valid JSON."
        for field in [
            "symptom_name",
            "description",
            "root_causes",
            "affected_machines",
            "representative_intervention_ids",
        ]:
            if not data.get(field):
                issues.append(f"Missing or empty: '{field}'")
        int_ids = data.get("representative_intervention_ids", [])
        if isinstance(int_ids, list):
            if not int_ids:
                issues.append("representative_intervention_ids is empty")
            for id_ in int_ids:
                if not str(id_).startswith("INT-"):
                    issues.append(f"Invalid ID format: '{id_}' (expected INT-XXXX)")
        return (
            "VALID"
            if not issues
            else "INVALID:\n" + "\n".join(f"- {i}" for i in issues)
        )

    return [get_cm_context_summarizer, get_known_case_templates_tool, validate_template]
