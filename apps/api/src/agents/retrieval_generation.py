from openai import OpenAI
from qdrant_client import QdrantClient

from core.config import config

from langsmith import traceable, get_current_run_tree

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)


@traceable(name="embed_query", run_type="embedding")
def embed_text(text: str, model: str = "text-embedding-3-small") -> list[float]:

    response = openai_client.embeddings.create(input=text, model=model)

    run = get_current_run_tree()
    if run:
        run.metadata["embedding_model"] = model
        run.metadata["usage_metadata"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.data[0].embedding


@traceable(name="data_retrieval", run_type="retriever")
def retrieve_data(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embedding_model: str = "text-embedding-3-small",
    top_k: int = 5,
) -> list[dict]:
    run = get_current_run_tree()
    if run:
        run.metadata["embedding_model"] = embedding_model

    query_vector = embed_text(query, embedding_model)
    search_result = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
    )
    return [
        {"id": point.id, "payload": point.payload, "score": point.score}
        for point in search_result.points
    ]


@traceable(name="format_context", run_type="prompt")
def format_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        context += f"ID: {payload.get('id', 'N/A')}\n"
        context += f"Machine: {payload.get('machine', 'N/A')}\n"
        context += f"Date: {payload.get('date_start', 'N/A')}\n"
        context += f"Summary: {payload.get('summary', 'N/A')}\n"
        context += "-" * 40 + "\n"
    return context


@traceable(name="build_prompt", run_type="prompt")
def build_prompt(context: str, query: str) -> str:
    return f"""You are a maintenance assistant that can answer questions about past interventions, like possible root causes and actions for a given symptom.

You will be given a question and a list of contexts.

Instructions:
- Use the contexts to answer the question.
- Be concise and to the point.
- Do not use markdown formatting.

Question: {query}

Contexts:
{context}
"""


@traceable(name="generate_answer", run_type="llm")
def generate_answer(prompt: str, generation_model: str = "gpt-5.4-nano") -> str:

    response = openai_client.chat.completions.create(
        model=generation_model,
        messages=[{"role": "system", "content": prompt}],
    )

    run = get_current_run_tree()
    if run:
        run.metadata["ls_model_name"] = generation_model
        run.metadata["ls_model_type"] = "chat"
        run.metadata["usage_metadata"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.choices[0].message.content.strip()


@traceable(name="rag_pipeline")
def rag_pipeline(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embedding_model: str = "text-embedding-3-small",
    generation_model: str = "gpt-5.4-nano",
    top_k: int = 5,
) -> dict:
    results = retrieve_data(client, collection_name, query, embedding_model, top_k)
    context = format_context(results)
    prompt = build_prompt(context, query)
    answer = generate_answer(prompt, generation_model)

    final_answer = {
        "answer": answer,
        "query": query,
        "retrieved_context_ids": [result["payload"].get("id") for result in results],
        "similarity_scores": [result["score"] for result in results],
    }

    return final_answer
