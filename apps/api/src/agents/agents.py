from typing import Union

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage, AIMessage
from langsmith import traceable, get_current_run_tree

from core.config import config
from agents.tools import (
    get_formatted_cm_context,
    get_formatted_procedure_context,
    get_remaining_life_tool,
    get_sensor_catalog_tool,
    get_sensor_readings_tool,
)

from agents.utils.prompt_management import prompt_template_config


# --- Prompts ---
AGENT_PROMPT = prompt_template_config(
    config.PROMPTS_PATH, "retrieval_generation"
).render()
INTENT_ROUTER_PROMPT = prompt_template_config(
    config.PROMPTS_PATH, "intent_router"
).render()


# --- Response Models ---


class UsedReference(BaseModel):
    source_type: str = Field(
        description="Type of source: 'intervention', 'procedure', 'sensor', or 'component_life'"
    )
    id: str = Field(
        description="Identifier — intervention ID (e.g. INT-2023-0070), procedure section, sensor tag, or component ID"
    )
    machine: str = Field(default="", description="Machine ID (e.g. HX-200)")
    detail: str = Field(
        default="",
        description="Short summary: intervention summary, procedure title, sensor reading, or component condition",
    )


class FinalResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[UsedReference] = Field(
        description="List of all sources used to answer the question"
    )


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


# --- LLM setup ---

ALL_TOOLS = [
    get_formatted_cm_context,
    get_formatted_procedure_context,
    get_sensor_catalog_tool,
    get_sensor_readings_tool,
    get_remaining_life_tool,
]

_llm = ChatOpenAI(model=config.GENERATION_MODEL)
_llm_with_tools = _llm.bind_tools(ALL_TOOLS, tool_choice="auto")
_llm_structured = _llm.with_structured_output(FinalResponse)
_llm_intent = _llm.with_structured_output(IntentRouterResponse)


# --- Nodes ---


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": config.GENERATION_MODEL},
)
def agent_node(state) -> dict:
    system_message = SystemMessage(content=AGENT_PROMPT)
    messages = state.messages

    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    last_message = messages[-1] if messages else None
    last_has_pending_tool_calls = (
        last_message is not None
        and hasattr(last_message, "tool_calls")
        and len(last_message.tool_calls) > 0
    )

    if has_tool_results and not last_has_pending_tool_calls:
        response: FinalResponse = _llm_structured.invoke([system_message, *messages])
        return {
            "messages": [AIMessage(content=response.answer)],
            "iteration": state.iteration + 1,
            "answer": response.answer,
            "final_answer": True,
            "references": response.references,
        }
    else:
        response = _llm_with_tools.invoke([system_message, *messages])
        return {
            "messages": [response],
            "iteration": state.iteration + 1,
            "answer": "",
            "final_answer": False,
            "references": [],
        }


@traceable(
    name="intent_router_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": config.GENERATION_MODEL},
)
def intent_router_node(state) -> dict:
    response: IntentRouterResponse = _llm_intent.invoke(
        [
            SystemMessage(content=INTENT_ROUTER_PROMPT),
            *state.messages,
        ]
    )

    current_run = get_current_run_tree()

    if current_run:
        trace_id = str(getattr(current_run, "trace_id", current_run.id))
    else:
        trace_id = ""

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
        "trace_id": trace_id,
    }
