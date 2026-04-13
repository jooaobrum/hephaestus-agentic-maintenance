import logging

from fastapi import APIRouter, Request

from api.models import RAGRequest, RAGResponse
from agents.retrieval_generation import agentic_rag_pipeline

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()


@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    """
    Endpoint to handle chat requests.
    """
    logger.info("Received a chat request.")
    result = agentic_rag_pipeline(query=payload.query)
    logger.info("Chat response generated successfully.")

    return RAGResponse(answer=result["answer"], references=[ref.model_dump() for ref in result["references"]])


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
