from pydantic import BaseModel

from core.config import app


class StreamRequest(BaseModel):
    query: str
    thread_id: str
    workspace_id: str = app.agents.default_workspace_id
    mode: str = "auto"


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_value: int
    feedback_text: str


class FeedbackResponse(BaseModel):
    status: str
    message: str
