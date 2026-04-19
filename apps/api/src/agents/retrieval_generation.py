from langsmith import traceable

from agents.graph import run_agent, stream_agent


@traceable(name="agentic_rag_pipeline")
def agentic_rag_pipeline(
    query: str, thread_id: str, langsmith_extra: dict = None
) -> dict:
    result = run_agent(query, thread_id)
    return {
        "answer": result["answer"] or "I could not find a relevant answer.",
        "references": result["references"],
        "question_relevant": result["question_relevant"],
        "trace_id": result["trace_id"],
    }


def stream_agentic_rag_pipeline(query: str, thread_id: str):
    """Yield event dicts from the streaming graph."""
    yield from stream_agent(query, thread_id)
