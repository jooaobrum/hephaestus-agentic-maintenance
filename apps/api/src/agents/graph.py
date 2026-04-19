from typing import Any, Annotated, List
from operator import add

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from agents.agents import (
    agent_node,
    intent_router_node,
    UsedReference,
    ALL_TOOLS,
)


# --- State ---


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: Annotated[List[UsedReference], add] = []
    trace_id: str = ""


# --- Routing ---


def tool_router(state: State) -> str:
    if state.final_answer or state.iteration > 2:
        return "end"
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "end"


def intent_router_conditional_edges(state: State) -> str:
    return "agent_node" if state.question_relevant else "end"


# --- Graph ---


def create_graph():
    _tool_node = ToolNode(ALL_TOOLS)

    _workflow = StateGraph(State)
    _workflow.add_node("tool_node", _tool_node)
    _workflow.add_node("intent_router_node", intent_router_node)
    _workflow.add_node("agent_node", agent_node)

    _workflow.add_edge(START, "intent_router_node")
    _workflow.add_conditional_edges(
        "intent_router_node",
        intent_router_conditional_edges,
        {"agent_node": "agent_node", "end": END},
    )
    _workflow.add_conditional_edges(
        "agent_node",
        tool_router,
        {"tools": "tool_node", "end": END},
    )
    _workflow.add_edge("tool_node", "agent_node")

    return _workflow


# --- Event Processing ---


def _tool_to_text(tool_call: dict) -> str:
    name = tool_call["name"]
    args = tool_call["args"]
    if name == "get_formatted_cm_context":
        return f"Searching maintenance history: '{args['query']}'"
    elif name == "get_formatted_procedure_context":
        return f"Looking up procedures: '{args['query']}'"
    elif name == "get_sensor_catalog_tool":
        return f"Fetching sensor catalog for {args['machine']}"
    elif name == "get_sensor_readings_tool":
        tag_info = f", tag={args['tag']}" if args.get("tag") else ""
        return f"Reading sensors for {args['machine']} ({args['start_date']} to {args['end_date']}{tag_info})"
    elif name == "get_remaining_life_tool":
        return f"Checking component life for {args['machine']}"
    else:
        return f"Calling {name}({args})"


def process_graph_event(chunk) -> dict | None:
    """Convert a raw graph stream chunk into a simple event dict, or None to skip."""
    is_node_start = chunk[0] == "debug" and chunk[1].get("type") == "task"
    is_node_end = chunk[0] == "updates"

    if is_node_start:
        node_name = chunk[1]["payload"]["name"]
        if node_name == "intent_router_node":
            return {"event": "status", "data": "Routing question..."}
        elif node_name == "agent_node":
            return {"event": "status", "data": "Agent is thinking..."}
        elif node_name == "tool_node":
            return {"event": "status", "data": "Executing tools..."}

    elif is_node_end:
        update = chunk[1]

        if "intent_router_node" in update:
            data = update["intent_router_node"]
            if data.get("question_relevant"):
                event = {"event": "status", "data": "Question is relevant, proceeding to agent"}
            else:
                event = {"event": "status", "data": f"Question not relevant: {data.get('answer', '')}"}
            if data.get("trace_id"):
                event["trace_id"] = data["trace_id"]
            return event

        elif "agent_node" in update:
            data = update["agent_node"]
            messages = data.get("messages", [])
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    tool_descriptions = [_tool_to_text(tc) for tc in msg.tool_calls]
                    return {"event": "tool_calls", "data": tool_descriptions}
                elif hasattr(msg, "content") and msg.content:
                    if data.get("final_answer"):
                        return {"event": "answer", "data": msg.content, "references": [ref.model_dump() for ref in data.get("references", [])]}

        elif "tool_node" in update:
            count = len(update["tool_node"].get("messages", []))
            return {"event": "status", "data": f"Got {count} tool result(s)"}

    return None


# --- Execution ---


def run_agent(query: str, thread_id: str) -> dict:
    initial_state = {"messages": [{"role": "user", "content": query}]}

    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        checkpointer.setup()

        _workflow = create_graph()
        graph = _workflow.compile(checkpointer=checkpointer)

        return graph.invoke(initial_state, config)


def stream_agent(query: str, thread_id: str):
    """Yield processed event dicts from the graph stream."""
    initial_state = {"messages": [{"role": "user", "content": query}]}

    config = {"configurable": {"thread_id": thread_id}}

    with PostgresSaver.from_conn_string(
        "postgresql://langgraph_user:langgraph_password@postgres:5432/langgraph_db"
    ) as checkpointer:
        checkpointer.setup()

        _workflow = create_graph()
        graph = _workflow.compile(checkpointer=checkpointer)

        for chunk in graph.stream(
            initial_state,
            config=config,
            stream_mode=["updates", "debug"],
        ):
            event = process_graph_event(chunk)
            if event is not None:
                yield event
