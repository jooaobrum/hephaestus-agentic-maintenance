import json
import logging

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from agents.multi_agent import stream_agent as stream_multiagent_pipeline
from api.models import FeedbackRequest, FeedbackResponse, StreamRequest
from api.processors.submit_feedback import submit_feedback

logger = logging.getLogger(__name__)

feedback_router = APIRouter()


@feedback_router.post("/")
def send_feedback(payload: FeedbackRequest) -> FeedbackResponse:
    logger.info("Received a feedback request.")
    status, message = submit_feedback(
        payload.trace_id, payload.feedback_value, payload.feedback_text
    )
    logger.info("Feedback response sent successfully.")

    return FeedbackResponse(status=status, message=message)


multiagent_router = APIRouter()


@multiagent_router.post("/stream")
async def multiagent_stream(payload: StreamRequest):
    logger.info("Received a multi-agent streaming chat request.")

    async def event_generator():
        async for event in stream_multiagent_pipeline(
            query=payload.query,
            thread_id=payload.thread_id,
            workspace_id=payload.workspace_id,
            mode=payload.mode,
        ):
            yield f"data: {json.dumps(event)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


api_router = APIRouter()
api_router.include_router(multiagent_router, prefix="/multiagent", tags=["MULTIAGENT"])
api_router.include_router(feedback_router, prefix="/submit_feedback", tags=["FEEDBACK"])
