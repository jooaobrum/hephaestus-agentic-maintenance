from typing import Union

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str
    thread_id: str


class UsedReference(BaseModel):
    source_type: str = Field(
        description="Type of source: 'intervention', 'procedure', 'sensor', or 'component_life'"
    )
    id: str = Field(
        description="Identifier — intervention ID (e.g. INT-2023-0070), procedure section, sensor tag, or component ID"
    )
    machine: str = Field(default="", description="Machine ID (e.g. HX-200)")
    detail: str = Field(
        default="",
        description="Short summary: intervention summary, procedure title, sensor reading, or component condition",
    )


class RAGResponse(BaseModel):
    answer: str
    references: list[UsedReference] = []
    trace_id: str


class FeedbackRequest(BaseModel):
    trace_id: str
    feedback_value: int
    feedback_text: str


class FeedbackResponse(BaseModel):
    status: str
    message: str
