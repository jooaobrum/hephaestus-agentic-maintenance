from openai import OpenAI
from qdrant_client import QdrantClient, models

from core.config import config

from langsmith import traceable, get_current_run_tree

import instructor

from pydantic import BaseModel, Field

import cohere

from agents.utils.promp_management import prompt_template_config

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
cohere_client = cohere.Client(api_key=config.CO_API_KEY)


class RAGUsedContext(BaseModel):
    id: int = Field("ID of the intervention")
    machine: str = Field("Machine of the intervention")
    date_start: str = Field("Date of the intervention")
    summary: str = Field("Summary of the intervention")


class RAGGenerationResponse(BaseModel):
    answer: str = Field("Answer to the question")
    references: list[RAGUsedContext] = Field(
        "List of problems that answer the question"
    )


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
    keyword_model: str = "bm25",
    top_k: int = 5,
) -> list[dict]:
    run = get_current_run_tree()
    if run:
        run.metadata["embedding_model"] = embedding_model

    query_vector = embed_text(query, embedding_model)
    search_results = client.query_points(
        collection_name=collection_name,
        prefetch=[
            models.Prefetch(
                query=query_vector, using=embedding_model, limit=top_k // 2
            ),
            models.Prefetch(
                query=models.Document(text=query, model="qdrant/" + keyword_model),
                using=keyword_model,
                limit=top_k // 2,
            ),
        ],
        query=models.RrfQuery(rrf=models.Rrf(weights=[1, 1])),
        limit=top_k,
    ).points

    return [
        {"id": point.id, "payload": point.payload, "score": point.score}
        for point in search_results
    ]


@traceable(name="reranking", run_type="retriever")
def rerank_results(
    query: str,
    results: list[dict],
    model: str = "rerank-v4.0-pro",
    top_k: int = 5,
) -> list[dict]:

    contexts = [result["payload"]["summary"] for result in results]

    response = cohere_client.rerank(
        model=model, query=query, documents=contexts, top_n=top_k
    )

    reranked_response = [results[res.index] for res in response.results]

    return reranked_response


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
    prompt_template = prompt_template_config(str(config.PROMPTS_PATH), config.PROMPT_NAME)

    return prompt_template.render(context=context, query=query)


@traceable(name="generate_answer", run_type="llm")
def generate_answer(prompt: str, generation_model: str = "gpt-5.4-nano"):

    client = instructor.from_provider(
        "openai/gpt-5.4-nano", mode=instructor.Mode.RESPONSES_TOOLS
    )

    response, raw_response = client.create_with_completion(
        messages=[{"role": "system", "content": prompt}],
        reasoning={"effort": "low"},
        response_model=RAGGenerationResponse,
    )

    run = get_current_run_tree()
    if run:
        run.metadata["ls_model_name"] = generation_model
        run.metadata["ls_model_type"] = "chat"
        run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.input_tokens,
            "output_tokens": raw_response.usage.output_tokens,
            "total_tokens": raw_response.usage.total_tokens,
        }

    return response, raw_response


@traceable(name="rag_pipeline")
def rag_pipeline(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embedding_model: str = "text-embedding-3-small",
    keyword_model: str = "bm25",
    generation_model: str = "gpt-5.4-nano",
    top_n: int = 5,
    top_k: int = 5,
) -> dict:
    results = retrieve_data(
        client, collection_name, query, embedding_model, keyword_model, top_n
    )
    reranked_results = rerank_results(query, results, config.RERANKING_MODEL, top_k)
    context = format_context(reranked_results)
    prompt = build_prompt(context, query)
    answer, raw_answer = generate_answer(prompt, generation_model)

    final_answer = {
        "data_object": answer,
        "answer": answer.answer,
        "references": answer.references,
        "query": query,
        "retrieved_context_ids": [result["payload"].get("id") for result in results],
        "retrieved_context": context,
        "similarity_score": [result["score"] for result in results],
    }

    return final_answer
