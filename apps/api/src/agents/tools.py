import cohere
from openai import OpenAI
from qdrant_client import QdrantClient, models
from langchain_core.tools import tool
from langsmith import traceable

from core.config import config

openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
cohere_client = cohere.Client(api_key=config.CO_API_KEY)
qdrant_client = QdrantClient(url=config.QDRANT_URL)


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
            models.Prefetch(
                query=query_vector, using=config.EMBEDDING_MODEL, limit=top_k // 2
            ),
            models.Prefetch(
                query=models.Document(
                    text=query, model="qdrant/" + config.KEYWORD_MODEL
                ),
                using=config.KEYWORD_MODEL,
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


# @traceable(name="reranking", run_type="retriever")
# def rerank_results(query: str, results: list[dict], top_k: int = 5) -> list[dict]:
#     if not results:
#         return []
#     contexts = [result["payload"]["summary"] for result in results]
#     response = cohere_client.rerank(
#         model=config.RERANKING_MODEL, query=query, documents=contexts, top_n=top_k
#     )
#     return [results[res.index] for res in response.results]


@traceable(name="format_cm_context", run_type="prompt")
def format_context(results: list[dict]) -> str:
    context = ""
    for result in results:
        payload = result["payload"]
        context += (
            f"ID: {payload.get('id', 'N/A')}\n"
            f"Machine: {payload.get('machine', 'N/A')}\n"
            f"Date: {payload.get('date_start', 'N/A')}\n"
            f"Summary: {payload.get('summary', 'N/A')}\n" + "-" * 40 + "\n"
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
    # reranked_results = rerank_results(query, results, top_k=top_k)
    return format_context(results)
