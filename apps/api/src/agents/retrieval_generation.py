from typing import Any, Annotated, List, Union
from operator import add

import yaml
import cohere

from openai import OpenAI

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage, SystemMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode

from langsmith import traceable

from qdrant_client import QdrantClient, models

from core.config import config


# --- Prompts ---
with open(config.PROMPTS_PATH) as f:
    AGENT_PROMPT = yaml.safe_load(f)["prompts"][config.PROMPT_NAME]

with open(config.INTENT_ROUTER_PROMPTS_PATH) as f:
    INTENT_ROUTER_PROMPT = yaml.safe_load(f)["prompts"][config.INTENT_ROUTER_PROMPT_NAME]


# --- Clients ---
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
cohere_client = cohere.Client(api_key=config.CO_API_KEY)
qdrant_client = QdrantClient(url=config.QDRANT_URL)


# --- Data Models ---

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


class State(BaseModel):
    messages: Annotated[List[Any], add] = []
    question_relevant: bool = False
    iteration: int = 0
    answer: str = ""
    final_answer: bool = False
    references: Annotated[List[RAGUsedContext], add] = []


# --- Retrieval helpers ---

@traceable(name="embed_query", run_type="embedding")
def embed_text(text: str) -> list[float]:
    response = openai_client.embeddings.create(input=text, model=config.EMBEDDING_MODEL)
    return response.data[0].embedding


@traceable(name="data_retrieval", run_type="retriever")
def retrieve_data(query: str, top_k: int = 5) -> list[dict]:
    query_vector = embed_text(query)
    search_results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=query_vector, using=config.EMBEDDING_MODEL, limit=top_k // 2),
            models.Prefetch(
                query=models.Document(text=query, model="qdrant/" + config.KEYWORD_MODEL),
                using=config.KEYWORD_MODEL,
                limit=top_k // 2,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1, 1])),
        limit=top_k,
    ).points
    return [{"id": point.id, "payload": point.payload, "score": point.score} for point in search_results]


@traceable(name="reranking", run_type="retriever")
def rerank_results(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
    if not results:
        return []
    contexts = [result["payload"]["summary"] for result in results]
    response = cohere_client.rerank(
        model=config.RERANKING_MODEL, query=query, documents=contexts, top_n=top_k
    )
    return [results[res.index] for res in response.results]


@traceable(name="format_cm_context", run_type="prompt")
def format_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        context += (
            f"ID: {payload.get('id', 'N/A')}\n"
            f"Machine: {payload.get('machine', 'N/A')}\n"
            f"Date: {payload.get('date_start', 'N/A')}\n"
            f"Summary: {payload.get('summary', 'N/A')}\n"
            + "-" * 40 + "\n"
        )
    return context


@tool
def get_formatted_cm_context(query: str, top_n: int = 10, top_k: int = 5) -> str:
    """Retrieve and rerank documents for a query, returning formatted context.

    Performs hybrid retrieval (dense + BM25) from Qdrant, reranks the candidates
    with Cohere, and returns them as a readable string.

    Args:
        query: The search query string.
        top_n: Number of candidates to retrieve before reranking. Defaults to 10.
        top_k: Number of results to keep after reranking. Defaults to 5.

    Returns:
        A formatted string with the top-k reranked intervention records,
        each showing ID, machine, date, and summary.
    """
    results = retrieve_data(query, top_k=top_n)
    reranked_results = rerank_results(query, results, top_k=top_k)
    return format_context(reranked_results)


# --- Module-level LLM setup — single source of truth ---

RETRIEVAL_TOOLS = [get_formatted_cm_context]

_llm = ChatOpenAI(model=config.GENERATION_MODEL)
_llm_with_tools = _llm.bind_tools(RETRIEVAL_TOOLS, tool_choice="auto")
_llm_structured = _llm.with_structured_output(FinalResponse)
_llm_intent = _llm.with_structured_output(IntentRouterResponse)


# --- Graph nodes ---

def tool_router(state: State) -> str:
    if state.final_answer:
        return "end"
    if state.iteration > 2:
        return "end"
    last_message = state.messages[-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "tools"
    return "end"


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={"ls_provider": "openai", "ls_model_name": config.GENERATION_MODEL},
)
def agent_node(state: State) -> dict:
    system_message = SystemMessage(content=AGENT_PROMPT)
    messages = state.messages

    has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
    last_message = messages[-1] if messages else None
    last_has_pending_tool_calls = last_message is not None and hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0

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
def intent_router_node(state: State) -> dict:
    response: IntentRouterResponse = _llm_intent.invoke([
        SystemMessage(content=INTENT_ROUTER_PROMPT),
        *state.messages,
    ])

    return {
        "question_relevant": response.question_relevant,
        "answer": response.answer,
    }


def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "agent_node"
    else:
        return "end"


# --- Build & compile graph (once at module load) ---

_tools = [get_formatted_cm_context]
_tool_node = ToolNode(_tools)

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


# --- Public entry point ---

@traceable(name="agentic_rag_pipeline")
def agentic_rag_pipeline(query: str) -> dict:
    initial_state = {"messages": [{"role": "user", "content": query}]}
    result = graph.invoke(initial_state)
    return {
        "answer": result["answer"],
        "references": result["references"],
        "question_relevant": result["question_relevant"],
    }
