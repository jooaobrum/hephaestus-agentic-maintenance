"""Supervisor graph: assembles coordinator + sub-agent nodes and exposes run/stream API."""

import uuid

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, START, END

from agents.multi_agent.config import POSTGRES_URL
from agents.multi_agent.states import SupervisorState
from agents.multi_agent.nodes import (
    coordinator_node,
    troubleshooting_node,
    summarizer_node,
)
from agents.multi_agent.edges import coordinator_edge

_NODE_STATUS = {
    "coordinator": "Coordinator is routing...",
    "troubleshooting": "Troubleshooting agent is thinking...",
    "summarizer": "Summarizer agent is thinking...",
}

_TERMINAL_NODES = {"coordinator", "troubleshooting", "summarizer"}


def create_workflow() -> StateGraph:
    workflow = StateGraph(SupervisorState)

    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("troubleshooting", troubleshooting_node)
    workflow.add_node("summarizer", summarizer_node)

    workflow.add_edge(START, "coordinator")
    workflow.add_conditional_edges(
        "coordinator",
        coordinator_edge,
        {
            "troubleshooting": "troubleshooting",
            "summarizer": "summarizer",
            END: END,
        },
    )
    workflow.add_edge("troubleshooting", END)
    workflow.add_edge("summarizer", END)

    return workflow


def _process_astream_event(event: dict) -> dict | None:
    """Convert a raw astream_events event into an SSE payload, or None to skip."""
    evt_type = event.get("event", "")
    node = event.get("metadata", {}).get("langgraph_node", "")

    if (
        evt_type == "on_chain_start"
        and event.get("name") == node
        and node in _NODE_STATUS
    ):
        return {"event": "status", "data": _NODE_STATUS[node]}

    if evt_type == "on_chat_model_stream":
        chunk = event["data"].get("chunk")
        if chunk:
            content = chunk.content
            if isinstance(content, str) and content:
                return {"event": "token", "data": content}
            if isinstance(content, list):
                text = "".join(
                    (
                        b.get("text", "")
                        if isinstance(b, dict)
                        else getattr(b, "text", "")
                    )
                    for b in content
                )
                if text:
                    return {"event": "token", "data": text}

    if evt_type == "on_tool_start" and node:
        tool_name = event.get("name", "")
        if tool_name:
            return {"event": "tool_calls", "data": [tool_name]}

    if evt_type == "on_chain_end" and node in _TERMINAL_NODES:
        output = event["data"].get("output") or {}
        if isinstance(output, dict):
            if (
                node == "coordinator"
                and output.get("coordinator_final")
                and output.get("answer")
            ):
                return {"event": "answer", "data": output["answer"]}
            if node in ("troubleshooting", "summarizer") and output.get("answer"):
                return {"event": "answer", "data": output["answer"]}

    return None


def run_agent(query: str, thread_id: str) -> dict:
    initial_state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        checkpointer.setup()
        graph = create_workflow().compile(checkpointer=checkpointer)
        return graph.invoke(initial_state, config)


async def stream_agent(query: str, thread_id: str):
    """Async generator: yield SSE event dicts from the supervisor graph."""
    run_id = uuid.uuid4()
    yield {"event": "trace_id", "trace_id": str(run_id)}

    initial_state = {"messages": [HumanMessage(content=query)]}
    config = {"configurable": {"thread_id": thread_id}, "run_id": run_id}

    async with AsyncPostgresSaver.from_conn_string(POSTGRES_URL) as checkpointer:
        await checkpointer.setup()
        graph = create_workflow().compile(checkpointer=checkpointer)

        async for event in graph.astream_events(
            initial_state, config=config, version="v2"
        ):
            result = _process_astream_event(event)
            if result is not None:
                yield result
