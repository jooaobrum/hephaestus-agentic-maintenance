"""Supervisor graph nodes.

The coordinator classifies intent and routes to the appropriate sub-agent session.
Troubleshooting and summarizer nodes delegate to their respective sessions.
"""

import uuid
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI

from agents.agentic_rag.utils.prompt_management import prompt_template_config
from agents.multi_agent.config import (
    GENERATION_MODEL,
    CONFIRM_SIGNAL,
    TEMPLATE_SAVED_SIGNAL,
)
from agents.multi_agent.sessions import (
    TroubleshootingContext,
    TroubleshootingSession,
    SummarizerContext,
    SummarizerSession,
)
from agents.multi_agent.states import (
    SupervisorState,
    Plan,
    FinalCoordinatorResponse,
)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_coordinator_llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0).bind_tools(
    [Plan, FinalCoordinatorResponse], tool_choice="any", parallel_tool_calls=False
)

# Module-level session stores — persist across supervisor graph invocations
_troubleshooting_sessions: dict[str, TroubleshootingSession] = {}
_summarizer_sessions: dict[str, SummarizerSession] = {}


def coordinator_node(state: SupervisorState) -> dict:
    # Fast-path: active agent already in progress — skip LLM classification
    active = state.get("active_agent")
    if active == "troubleshooting":
        return {"coordinator_next": "troubleshooting", "coordinator_final": False}
    if active == "summarizer":
        return {"coordinator_next": "summarizer", "coordinator_final": False}

    system_prompt = prompt_template_config(
        _PROMPTS_DIR / "coordinator.yml", "coordinator"
    ).render()

    response = _coordinator_llm.invoke(
        [
            SystemMessage(content=system_prompt),
            *state["messages"],
        ]
    )

    final_answer = False
    answer = ""
    next_agent = ""
    task = state.get("task")
    period = state.get("period")
    messages_out = [response]

    if response.tool_calls:
        call = response.tool_calls[0]
        if call["name"] == "Plan":
            args = call["args"]
            next_agent = args.get("next_agent", "")
            plan = args.get("plan", [])
            if plan:
                first = plan[0]
                if first.get("machine_id"):
                    machine_id = first["machine_id"]
                if first.get("symptom"):
                    symptom = first["symptom"]
                if first.get("task"):
                    task = first["task"]
                if first.get("period"):
                    period = first["period"]
            messages_out.append(
                ToolMessage(
                    content=f"Routing to {next_agent}.",
                    tool_call_id=call["id"],
                )
            )
        else:  # FinalCoordinatorResponse
            answer = call["args"].get("answer", "")
            final_answer = True
            messages_out.append(
                ToolMessage(
                    content=answer,
                    tool_call_id=call["id"],
                )
            )

    return {
        "messages": messages_out,
        "coordinator_next": next_agent,
        "coordinator_final": final_answer,
        "active_agent": next_agent if not final_answer else None,
        "machine_id": machine_id,
        "symptom": symptom,
        "task": task,
        "period": period,
        "answer": answer,
    }


def troubleshooting_node(state: SupervisorState, config: RunnableConfig) -> dict:
    parent_tid = config.get("configurable", {}).get("thread_id")
    tid = state.get("troubleshooting_thread_id") or (f"{parent_tid}_troubleshooting" if parent_tid else str(uuid.uuid4()))
    is_new_session = tid not in _troubleshooting_sessions
    is_first_turn = state.get("troubleshooting_thread_id") is None

    if is_new_session:
        ctx = TroubleshootingContext(
            machine_id=state.get("machine_id") or "",
            symptom=state.get("symptom") or "",
        )
        session = TroubleshootingSession(ctx, thread_id=tid)
        _troubleshooting_sessions[tid] = session
        if is_first_turn:
            session.start(
                task=state.get("task"),
                period=state.get("period"),
                extra_config=config,
            )
        else:
            # API restart — send latest human message to continue from DB history
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    session.reply(msg.content, extra_config=config)
                    break
    else:
        session = _troubleshooting_sessions[tid]
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                session.reply(msg.content, extra_config=config)
                break

    response = session.last_response
    confirmed = CONFIRM_SIGNAL in response
    clean = response.replace(CONFIRM_SIGNAL, "").strip()

    if confirmed:
        session.persist_case()

    return {
        "messages": [AIMessage(content=clean)],
        "answer": clean,
        "troubleshooting_thread_id": tid,
        "active_agent": None if confirmed else "troubleshooting",
    }


def summarizer_node(state: SupervisorState, config: RunnableConfig) -> dict:
    parent_tid = config.get("configurable", {}).get("thread_id")
    tid = state.get("summarizer_thread_id") or (f"{parent_tid}_summarizer" if parent_tid else str(uuid.uuid4()))
    is_new_session = tid not in _summarizer_sessions
    is_first_turn = state.get("summarizer_thread_id") is None

    if is_new_session:
        import re
        task = state.get("task")
        period = state.get("period")
        dates = re.findall(r"\d{4}-\d{2}-\d{2}", period or task or "")
        start_date = dates[0] if len(dates) > 0 else None
        end_date = dates[1] if len(dates) > 1 else None

        ctx = SummarizerContext(
            machine_id=state.get("machine_id"),
            symptom=state.get("symptom"),
            start_date=start_date,
            end_date=end_date,
        )
        session = SummarizerSession(ctx, thread_id=tid)
        _summarizer_sessions[tid] = session
        if is_first_turn:
            session.start(
                task=task,
                period=period,
                extra_config=config,
            )
        else:
            for msg in reversed(state["messages"]):
                if isinstance(msg, HumanMessage):
                    session.reply(msg.content, extra_config=config)
                    break
    else:
        session = _summarizer_sessions[tid]
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                session.reply(msg.content, extra_config=config)
                break

    response = session.last_response
    done = TEMPLATE_SAVED_SIGNAL in response
    clean = response.replace(TEMPLATE_SAVED_SIGNAL, "").strip()

    if done:
        session.persist_template()

    return {
        "messages": [AIMessage(content=clean)],
        "answer": clean,
        "summarizer_thread_id": tid,
        "active_agent": None if done else "summarizer",
    }
