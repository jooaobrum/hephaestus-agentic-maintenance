"""Sub-agent sessions backed by create_react_agent + PostgresSaver.

Each session wraps a LangGraph ReAct agent with its own checkpointer thread
so it maintains its own conversation history independently of the supervisor.
"""

import uuid
from pathlib import Path

import psycopg
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.config import merge_configs
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

from agents.agentic_rag.utils.prompt_management import prompt_template_config
from agents.multi_agent.tools import (
    check_machine_exists,
    find_similar_machines,
    list_procedure_sections,
    list_known_issue_categories,
    get_current_date,
    calculate_date_window,
    list_available_machines,
    get_formatted_cm_context,
    get_intervention_detail,
    get_fleet_impact_for_symptom,
    query_known_issues_graph,
    get_sensor_catalog_tool,
    get_sensor_anomaly_summary,
    get_remaining_life_tool,
    save_confirmed_rca_case,
    summarize_intervention,
    build_known_case_template,
    save_known_case_template,
    list_intervention_ids_by_date,
)
from agents.multi_agent.config import (
    GENERATION_MODEL,
    POSTGRES_URL,
    DOMAIN_KNOWLEDGE_MD,
    CASES_DIR,
    TEMPLATES_DIR,
    get_domain_hint,
)
from agents.multi_agent.workspace import (
    TROUBLESHOOTING_FILE_TOOLS,
    SUMMARIZER_FILE_TOOLS,
    get_troubleshooting_workspace,
    get_summarizer_workspace,
    make_troubleshooting_offload_tools,
    make_summarizer_offload_tools,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_TROUBLESHOOTING_OFFLOAD_TOOLS = make_troubleshooting_offload_tools()
_SUMMARIZER_OFFLOAD_TOOLS = make_summarizer_offload_tools()

TROUBLESHOOTING_TOOLS = (
    [
        check_machine_exists,
        find_similar_machines,
        list_procedure_sections,
        list_known_issue_categories,
        get_current_date,
        get_formatted_cm_context,
        get_intervention_detail,
        get_fleet_impact_for_symptom,
        query_known_issues_graph,
        get_sensor_catalog_tool,
        get_sensor_anomaly_summary,
        get_remaining_life_tool,
        save_confirmed_rca_case,
    ]
    + _TROUBLESHOOTING_OFFLOAD_TOOLS
    + TROUBLESHOOTING_FILE_TOOLS
)

SUMMARIZER_TOOLS = (
    [
        get_current_date,
        calculate_date_window,
        check_machine_exists,
        list_available_machines,
        get_intervention_detail,
        summarize_intervention,
        build_known_case_template,
        save_known_case_template,
        list_intervention_ids_by_date,
    ]
    + _SUMMARIZER_OFFLOAD_TOOLS
    + SUMMARIZER_FILE_TOOLS
)


def _make_checkpointer() -> PostgresSaver:
    conn = psycopg.connect(POSTGRES_URL)
    cp = PostgresSaver(conn)
    cp.setup()
    return cp


_troubleshooting_checkpointer = _make_checkpointer()
_summarizer_checkpointer = _make_checkpointer()


def _create_troubleshooting_agent():
    system_prompt = prompt_template_config(
        _PROMPTS_DIR / "rca_troubleshooting.yml", "rca_troubleshooting"
    ).render()
    llm = ChatOpenAI(
        model=GENERATION_MODEL, temperature=0, timeout=90, max_retries=2, streaming=True
    )
    return create_react_agent(
        model=llm,
        tools=TROUBLESHOOTING_TOOLS,
        prompt=SystemMessage(content=system_prompt),
        checkpointer=_troubleshooting_checkpointer,
    )


def _create_summarizer_agent():
    system_prompt = prompt_template_config(
        _PROMPTS_DIR / "summarizer.yml", "summarizer"
    ).render()
    llm = ChatOpenAI(
        model=GENERATION_MODEL, temperature=0, timeout=90, max_retries=2, streaming=True
    )
    return create_react_agent(
        model=llm,
        tools=SUMMARIZER_TOOLS,
        prompt=SystemMessage(content=system_prompt),
        checkpointer=_summarizer_checkpointer,
    )


_troubleshooting_agent = _create_troubleshooting_agent()
_summarizer_agent = _create_summarizer_agent()


def _extract_last_response(result: dict) -> str:
    for msg in reversed(result.get("messages", [])):
        if isinstance(msg, AIMessage) and msg.content:
            content = msg.content
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                    for b in content
                )
            if content.strip():
                return content
    return ""


# Dictionaries of active sub-agent sessions are managed at the node level
# checkpointer state handles message histories natively


# ── Troubleshooting Session ───────────────────────────────────────────────────


class TroubleshootingContext(BaseModel):
    machine_id: str
    symptom: str


class TroubleshootingSession:
    def __init__(self, context: TroubleshootingContext, thread_id: str | None = None):
        self.context = context
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 60,
        }
        self.workspace = get_troubleshooting_workspace(self.thread_id)
        self._result: dict | None = None
        self._seed_workspace()

    def _seed_workspace(self):
        dk = self.workspace / "domain_knowledge.md"
        if not dk.exists():
            dk.write_text(DOMAIN_KNOWLEDGE_MD)

    def _first_message(self, task: str | None = None, period: str | None = None) -> str:
        lines = [
            f"Machine: {self.context.machine_id}",
            f"Symptom: {self.context.symptom}",
        ]
        if task:
            lines.append(f"Task: {task}")
        if period:
            lines.append(f"Period: {period}")
        hint = get_domain_hint(self.context.machine_id)
        if hint:
            lines.append(hint)
        lines.append("/domain_knowledge.md available in workspace.")
        lines.append("Begin the investigation arc.")
        return "\n".join(lines)

    def _prepare_config(self, extra_config: RunnableConfig | None = None) -> dict:
        cfg = (
            merge_configs(self.config, extra_config)
            if extra_config
            else self.config.copy()
        )
        cfg["configurable"] = cfg.get("configurable", {}).copy()
        cfg["configurable"]["thread_id"] = self.thread_id
        cfg["configurable"].pop("checkpoint_id", None)
        cfg["configurable"].pop("checkpoint_ns", None)
        return cfg

    def start(
        self,
        task: str | None = None,
        period: str | None = None,
        extra_config: RunnableConfig | None = None,
    ) -> "TroubleshootingSession":
        cfg = self._prepare_config(extra_config)
        self._result = _troubleshooting_agent.invoke(
            {"messages": [HumanMessage(content=self._first_message(task, period))]},
            config=cfg,
        )
        return self

    def _restore_result_if_needed(self):
        if self._result is not None:
            return
        try:
            checkpoint_tuple = _troubleshooting_checkpointer.get_tuple(self.config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_vals = checkpoint_tuple.checkpoint.get("channel_values", {})
                messages = channel_vals.get("messages", [])
                if messages:
                    self._result = {"messages": messages}
        except Exception:
            pass

    def reply(
        self, user_input: str, extra_config: RunnableConfig | None = None
    ) -> "TroubleshootingSession":
        cfg = self._prepare_config(extra_config)
        self._restore_result_if_needed()
        self._result = _troubleshooting_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=cfg,
        )
        return self

    @property
    def last_response(self) -> str:
        return _extract_last_response(self._result) if self._result else ""

    def persist_case(self) -> str | None:
        src = self.workspace / "case.md"
        if not src.exists():
            return None
        CASES_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        safe = self.context.machine_id.replace("/", "-")
        dest = CASES_DIR / f"{safe}_{ts}.md"
        dest.write_text(src.read_text())
        return str(dest)


# ── Summarizer Session ────────────────────────────────────────────────────────


class SummarizerContext(BaseModel):
    machine_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    symptom: str | None = None


class SummarizerSession:
    def __init__(
        self, context: SummarizerContext | None = None, thread_id: str | None = None
    ):
        self.context = context or SummarizerContext()
        self.thread_id = thread_id or str(uuid.uuid4())
        self.config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": 60,
        }
        self.workspace = get_summarizer_workspace(self.thread_id)
        self._result: dict | None = None

    def _first_message(self, task: str | None = None, period: str | None = None) -> str:
        ctx = self.context
        parts = []
        if ctx.machine_id:
            parts.append(f"Machine: {ctx.machine_id}")
        if ctx.start_date:
            parts.append(f"Start date: {ctx.start_date}")
        if ctx.end_date:
            parts.append(f"End date: {ctx.end_date}")
        if ctx.symptom:
            parts.append(f"Symptom: {ctx.symptom}")
        if task:
            parts.append(f"Task: {task}")
        if period:
            parts.append(f"Period: {period}")
        if parts:
            parts.append("Begin the summarizer arc.")
            return "\n".join(parts)
        return "Hello! I need help building a known case template."

    def _prepare_config(self, extra_config: RunnableConfig | None = None) -> dict:
        cfg = (
            merge_configs(self.config, extra_config)
            if extra_config
            else self.config.copy()
        )
        cfg["configurable"] = cfg.get("configurable", {}).copy()
        cfg["configurable"]["thread_id"] = self.thread_id
        cfg["configurable"].pop("checkpoint_id", None)
        cfg["configurable"].pop("checkpoint_ns", None)
        return cfg

    def start(
        self,
        task: str | None = None,
        period: str | None = None,
        extra_config: RunnableConfig | None = None,
    ) -> "SummarizerSession":
        cfg = self._prepare_config(extra_config)
        self._result = _summarizer_agent.invoke(
            {"messages": [HumanMessage(content=self._first_message(task, period))]},
            config=cfg,
        )
        return self

    def _restore_result_if_needed(self):
        if self._result is not None:
            return
        try:
            checkpoint_tuple = _summarizer_checkpointer.get_tuple(self.config)
            if checkpoint_tuple and checkpoint_tuple.checkpoint:
                channel_vals = checkpoint_tuple.checkpoint.get("channel_values", {})
                messages = channel_vals.get("messages", [])
                if messages:
                    self._result = {"messages": messages}
        except Exception:
            pass

    def reply(
        self, user_input: str, extra_config: RunnableConfig | None = None
    ) -> "SummarizerSession":
        cfg = self._prepare_config(extra_config)
        self._restore_result_if_needed()
        self._result = _summarizer_agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config=cfg,
        )
        return self

    @property
    def last_response(self) -> str:
        return _extract_last_response(self._result) if self._result else ""

    def persist_template(self) -> str | None:
        src = self.workspace / "known_case_template.md"
        if not src.exists():
            return None
        TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        from datetime import datetime

        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        dest = TEMPLATES_DIR / f"template_{ts}.md"
        dest.write_text(src.read_text())
        return str(dest)
