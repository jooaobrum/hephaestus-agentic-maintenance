import logging

from fastapi import APIRouter, Request
from qdrant_client import QdrantClient

from api.models import RAGRequest, RAGResponse
from agents.retrieval_generation import rag_pipeline
from core.config import config

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

qdrant_client = QdrantClient(url=config.QDRANT_URL)

rag_router = APIRouter()


@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    """
    Endpoint to handle chat requests.
    """
    logger.info("Received a chat request.")
    result = rag_pipeline(
        client=qdrant_client,
        collection_name=config.QDRANT_COLLECTION,
        query=payload.query,
        embedding_model=config.EMBEDDING_MODEL,
        keyword_model=config.KEYWORD_MODEL,
        generation_model=config.GENERATION_MODEL,
    )
    logger.info("Chat response generated successfully.")
    logger.info(f"Chat response: {result}")

    answer = result["answer"]
    return RAGResponse(answer=answer)


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
