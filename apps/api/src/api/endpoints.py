import json
import logging

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api.models import RAGRequest, RAGResponse, FeedbackRequest, FeedbackResponse
from agents.retrieval_generation import agentic_rag_pipeline, stream_agentic_rag_pipeline
from api.processors.submit_feedback import submit_feedback

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

rag_router = APIRouter()
feedback_router = APIRouter()


@rag_router.post("/")
def chat(request: Request, payload: RAGRequest) -> RAGResponse:
    """
    Endpoint to handle chat requests.
    """
    logger.info("Received a chat request.")
    result = agentic_rag_pipeline(
        query=payload.query,
        thread_id=payload.thread_id,
        langsmith_extra={"metadata": {"thread_id": payload.thread_id}},
    )
    logger.info("Chat response generated successfully.")

    return RAGResponse(
        answer=result["answer"],
        references=[ref.model_dump() for ref in result["references"]],
        trace_id=result["trace_id"],
    )


@rag_router.post("/stream")
def chat_stream(request: Request, payload: RAGRequest):
    """
    SSE streaming endpoint for the RAG pipeline.
    """
    logger.info("Received a streaming chat request.")

    def event_generator():
        for event in stream_agentic_rag_pipeline(
            query=payload.query, thread_id=payload.thread_id
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@feedback_router.post("/")
def send_feedback(request: Request, payload: FeedbackRequest) -> FeedbackResponse:
    """
    Endpoint to handle chat requests.
    """
    logger.info("Received a feedback request.")
    status, message = submit_feedback(
        payload.trace_id, payload.feedback_value, payload.feedback_text
    )
    logger.info("Feedback response sent successfully.")

    return FeedbackResponse(status=status, message=message)


api_router = APIRouter()
api_router.include_router(rag_router, prefix="/rag", tags=["RAG"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["FEEDBACK"])
