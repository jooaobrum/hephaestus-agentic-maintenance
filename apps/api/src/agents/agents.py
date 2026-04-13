from typing import Union

import yaml

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, SystemMessage
from langsmith import traceable

from core.config import config
from agents.tools import get_formatted_cm_context


# --- Prompts ---
with open(config.PROMPTS_PATH) as f:
    AGENT_PROMPT = yaml.safe_load(f)["prompts"][config.PROMPT_NAME]

with open(config.INTENT_ROUTER_PROMPTS_PATH) as f:
    INTENT_ROUTER_PROMPT = yaml.safe_load(f)["prompts"][config.INTENT_ROUTER_PROMPT_NAME]


# --- Response Models ---

class RAGUsedContext(BaseModel):
    id: Union[int, str] = Field(description="ID of the intervention")
    machine: str = Field(description="Machine of the intervention")
    date_start: str = Field(description="Date of the intervention")
    summary: str = Field(description="Summary of the intervention")


class FinalResponse(BaseModel):
    answer: str = Field(description="Answer to the question")
    references: list[RAGUsedContext] = Field(description="List of contexts used to answer the question")


class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str


# --- LLM setup ---

RETRIEVAL_TOOLS = [get_formatted_cm_context]

_llm = ChatOpenAI(model=config.GENERATION_MODEL)
_llm_with_tools = _llm.bind_tools(RETRIEVAL_TOOLS, tool_choice="auto")
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
            "messages": [],
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
    response: IntentRouterResponse = _llm_intent.invoke([
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        *state.messages,
    ])
    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
    }
