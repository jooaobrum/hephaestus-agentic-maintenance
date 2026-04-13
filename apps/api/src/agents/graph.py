from typing import Any, Annotated, List
from operator import add

from pydantic import BaseModel, Field

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from agents.agents import agent_node, intent_router_node, RAGUsedContext, RETRIEVAL_TOOLS


# --- State ---

class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []


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

_tool_node = ToolNode(RETRIEVAL_TOOLS)

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

graph = _workflow.compile()


# --- Execution ---

def run_agent(query: str) -> dict:
    initial_state = {"messages": [{"role": "user", "content": query}]}
    return graph.invoke(initial_state)
