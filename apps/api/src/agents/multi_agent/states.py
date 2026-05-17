from typing import List
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState


class Delegation(BaseModel):
    agent: str = Field(description="Agent to invoke: 'troubleshooting' or 'summarizer'")
    task: str = Field(description="Self-contained task description for the agent")
    machine_id: str | None = None
    symptom: str | None = None
    period: str | None = None


class Plan(BaseModel):
    next_agent: str = Field(description="'troubleshooting' or 'summarizer'")
    plan: List[Delegation]


class FinalCoordinatorResponse(BaseModel):
    answer: str = Field(description="Direct answer to the user (no routing needed)")


class SupervisorState(MessagesState):
    answer: str
    active_agent: str | None
    coordinator_next: str
    coordinator_final: bool
    troubleshooting_thread_id: str | None
    summarizer_thread_id: str | None
    machine_id: str | None
    symptom: str | None
    task: str | None
    period: str | None
