"""
investigate/models.py — PHILM POC
"""
from __future__ import annotations
from enum import Enum
from typing import Annotated, Any, Literal, Optional
from pydantic import BaseModel, Field, model_validator
from langgraph.graph.message import add_messages


# ── Phases ────────────────────────────────────────────────────────

class Phase(str, Enum):
    CONTEXT_GATHERING = "CONTEXT_GATHERING"
    INVESTIGATE       = "INVESTIGATE"
    HYPOTHESIZE       = "HYPOTHESIZE"
    VALIDATE          = "VALIDATE"
    DISCRIMINATE      = "DISCRIMINATE"
    RECOMMEND         = "RECOMMEND"
    DONE              = "DONE"


# ── Context ───────────────────────────────────────────────────────

class MachineContext(BaseModel):
    symptom:              str
    machine_id:           Optional[str] = None
    family:               Optional[str] = None   # machine family / type
    period:               Optional[str] = None   # when did it start / observation window
    tried_actions:        Optional[str] = None
    recent_maintenances:  Optional[str] = None
    operating_conditions: Optional[str] = None
    additional_context:   Optional[str] = None   # free text for anything else


# ── Evidence ──────────────────────────────────────────────────────

class EvidenceItem(BaseModel):
    id:      str
    type:    Literal["procedure", "intervention", "user_answer"]
    label:   str
    content: Optional[str] = None
    source:  str


# ── Actions ───────────────────────────────────────────────────────

class RecommendedAction(BaseModel):
    priority:    Literal["immediate", "short_term", "preventive"]
    action:      str
    component:   Optional[str]       = None
    parts_needed: Optional[list[str]] = None


# ── Compound finding ──────────────────────────────────────────────

class HypothesisBranch(BaseModel):
    id:                  str
    rank:                int   = 0
    cause:               str
    confidence:          float
    escalate_to_expert:  bool  = False
    evidence_ids:        list[str] = Field(default_factory=list)
    discriminating_check: str
    action_plan:         list[RecommendedAction]


class CompoundFinding(BaseModel):
    session_id:             str
    machine_id:             str
    symptom:                str
    branches:               list[HypothesisBranch]
    recommended_first_check: str
    status: Literal["draft", "approved", "rejected"] = "draft"
    resolved_branch_id:     Optional[str] = None

    @model_validator(mode="after")
    def sort_and_rank(self) -> "CompoundFinding":
        self.branches.sort(key=lambda b: b.confidence, reverse=True)
        for i, b in enumerate(self.branches):
            b.rank = i + 1
        return self

    @property
    def top_branch(self) -> HypothesisBranch:
        return self.branches[0]

    def apply_delta(self, branch_id: str, delta: float,
                    threshold: float = 0.05) -> list[str]:
        for b in self.branches:
            if b.id == branch_id:
                b.confidence = max(0.0, min(1.0, b.confidence + delta))
        eliminated = [b.id for b in self.branches if b.confidence < threshold]
        self.branches = [b for b in self.branches if b.confidence >= threshold]
        total = sum(b.confidence for b in self.branches)
        if total > 0:
            for b in self.branches:
                b.confidence = round(b.confidence / total, 4)
        self.branches.sort(key=lambda b: b.confidence, reverse=True)
        for i, b in enumerate(self.branches):
            b.rank = i + 1
        return eliminated

    def is_converged(self, threshold: float = 0.80) -> bool:
        return bool(self.branches) and self.top_branch.confidence >= threshold


# ── Questions ─────────────────────────────────────────────────────
# sensor_id is the evolution seam:
#   POC:    None  → ask_user via interrupt()
#   Future: set   → auto-evaluate from tool, skip ask_user

class ValidationQuestion(BaseModel):
    id:                str
    question:          str
    options:           list[str]
    confidence_deltas: dict[str, float]   # option → delta
    sensor_id:         Optional[str] = None   # evolution seam


class DiscriminationQuestion(BaseModel):
    id:             str
    question:       str
    options:        list[str]
    branch_impacts: dict[str, dict[str, float]]   # option → {branch_id: delta}
    sensor_id:      Optional[str] = None   # evolution seam


# ── SSE events ────────────────────────────────────────────────────

class EventType(str, Enum):
    PHASE_CHANGE     = "phase_change"
    TEXT_DELTA       = "text_delta"
    TOOL_CALL        = "tool_call"
    TOOL_RESULT      = "tool_result"
    INTERRUPT        = "interrupt"
    CONTEXT_READY    = "context_ready"
    COMPOUND_FINDING = "compound_finding"
    BRANCH_UPDATE    = "branch_update"
    DONE             = "done"
    ERROR            = "error"


class AgentEvent(BaseModel):
    type:         EventType
    phase:        Optional[str]             = None
    content:      Optional[str]             = None
    tool_name:    Optional[str]             = None
    tool_result:  Optional[str]             = None
    evidence:     Optional[EvidenceItem]    = None
    question:     Optional[str]             = None
    options:      Optional[list[str]]       = None
    interrupt_id: Optional[str]             = None
    meta:         Optional[dict]            = None
    context:      Optional[MachineContext]  = None
    compound:     Optional[CompoundFinding] = None
    eliminated:   Optional[list[str]]       = None
    error:        Optional[str]             = None

    def to_sse(self) -> str:
        return f"data: {self.model_dump_json()}\n\n"


# ── LangGraph state ───────────────────────────────────────────────

class InvestigationState(BaseModel):
    machine_id:      str
    session_id:      str
    phase:           Phase = Phase.CONTEXT_GATHERING
    machine_context: Optional[MachineContext]   = None
    messages:        Annotated[list, add_messages] = Field(default_factory=list)
    evidence:        list[EvidenceItem]           = Field(default_factory=list)
    compound:        Optional[CompoundFinding]    = None
    rejection_reason: Optional[str]  = None
    iteration:        int            = 0

    class Config:
        arbitrary_types_allowed = True


# ── API schemas ───────────────────────────────────────────────────

class StartRequest(BaseModel):
    machine_id:      str
    initial_message: Optional[str] = None


class ReplyRequest(BaseModel):
    session_id:   str
    message:      str
    interrupt_id: Optional[str] = None


class ApproveRequest(BaseModel):
    session_id:  str
    approved_by: str


class MarkResolvedRequest(BaseModel):
    session_id:  str
    branch_id:   str
    resolved_by: str
    was_correct: bool = True
