from langsmith import traceable

from agents.graph import run_agent


@traceable(name="agentic_rag_pipeline")
def agentic_rag_pipeline(query: str) -> dict:
    result = run_agent(query)
    return {
        "answer": result["answer"] or "I could not find a relevant answer.",
        "references": result["references"],
        "question_relevant": result["question_relevant"],
    }
