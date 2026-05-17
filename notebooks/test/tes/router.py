"""
investigate/router.py — PHILM POC

  POST /investigate/start          Start investigation → SSE
  POST /investigate/{id}/reply     Resume after interrupt → SSE
  GET  /investigate/{id}/state     Current state (reconnect)
  POST /investigate/{id}/approve   Approve finding
  POST /investigate/{id}/resolve   Mark which branch was correct
"""

from __future__ import annotations
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .models import (
    StartRequest,
    ReplyRequest,
    ApproveRequest,
    MarkResolvedRequest,
    CompoundFinding,
    AgentEvent,
    EventType,
)
from .graph import stream_investigation, resume_investigation, investigation_graph

router = APIRouter(prefix="/investigate", tags=["investigate"])
_sessions: dict[str, dict] = {}  # replace with Redis in production


# ── Helpers ───────────────────────────────────────────────────────


def _session(sid: str) -> dict:
    if sid not in _sessions:
        raise HTTPException(404, f"Session {sid} not found")
    return _sessions[sid]


def _graph_state(sid: str):
    return investigation_graph.get_state({"configurable": {"thread_id": sid}})


def _sse(gen: AsyncGenerator, headers: dict = {}) -> StreamingResponse:
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", **headers},
    )


async def _keepalive(gen: AsyncGenerator[str, None], interval: int = 15):
    import asyncio

    it = gen.__aiter__()
    while True:
        try:
            yield await asyncio.wait_for(it.__anext__(), timeout=interval)
        except asyncio.TimeoutError:
            yield ": keepalive\n\n"
        except StopAsyncIteration:
            break


# ── Endpoints ─────────────────────────────────────────────────────


@router.post("/start")
async def start(body: StartRequest):
    """Start a new investigation. Session ID returned in X-Session-Id header."""
    sid = str(uuid.uuid4())
    _sessions[sid] = {
        "session_id": sid,
        "machine_id": body.machine_id,
        "status": "running",
    }

    async def stream():
        try:
            async for evt in _keepalive(
                stream_investigation(sid, body.machine_id, body.initial_message)
            ):
                yield evt
        except Exception as e:
            yield AgentEvent(type=EventType.ERROR, error=str(e)).to_sse()

    return _sse(stream(), {"X-Session-Id": sid})


@router.post("/{sid}/reply")
async def reply(sid: str, body: ReplyRequest):
    """Resume after any interrupt (context, investigate, validate, discriminate, recommend)."""
    _session(sid)

    async def stream():
        try:
            async for evt in _keepalive(resume_investigation(sid, body.message)):
                yield evt
        except Exception as e:
            yield AgentEvent(type=EventType.ERROR, error=str(e)).to_sse()

    return _sse(stream())


@router.get("/{sid}/state")
async def get_state(sid: str):
    """Current phase + compound finding. Used for frontend reconnect."""
    _session(sid)
    checkpoint = _graph_state(sid)
    values = checkpoint.values if checkpoint else {}
    return {
        "phase": values.get("phase"),
        "context": values.get("machine_context"),
        "compound": values.get("compound"),
    }


@router.post("/{sid}/approve")
async def approve(sid: str, body: ApproveRequest):
    """Approve the compound finding → waiting_approval."""
    checkpoint = _graph_state(sid)
    if not checkpoint:
        raise HTTPException(404, "No graph state")

    compound: CompoundFinding = checkpoint.values.get("compound")
    if not compound:
        raise HTTPException(400, "No finding to approve")

    compound.status = "approved"
    _sessions[sid]["status"] = "done"

    # TODO: await db.findings.upsert(compound)
    # TODO: await task_queue.enqueue("capitalize", session_id=sid)

    return {"status": "approved", "compound": compound}


@router.post("/{sid}/resolve")
async def resolve(sid: str, body: MarkResolvedRequest):
    """
    Record which branch was the actual root cause after the fix.
    was_correct=False flags the investigation for review.
    Feeds knowledge base prior updates.
    """
    checkpoint = _graph_state(sid)
    if not checkpoint:
        raise HTTPException(404, "No graph state")

    compound: CompoundFinding = checkpoint.values.get("compound")
    if not compound:
        raise HTTPException(400, "No finding to resolve")

    branch = next((b for b in compound.branches if b.id == body.branch_id), None)
    if not branch:
        raise HTTPException(404, f"Branch {body.branch_id} not found")

    compound.resolved_branch_id = body.branch_id
    compound.status = "approved"
    _sessions[sid]["status"] = "done"

    # TODO: await db.findings.upsert(compound)
    # TODO: await task_queue.enqueue("update_priors", {
    #     "machine_id":     body.machine_id if hasattr(body, 'machine_id') else compound.machine_id,
    #     "family":         checkpoint.values.get("machine_context", {}).get("family"),
    #     "symptom":        compound.symptom,
    #     "resolved_branch": body.branch_id,
    #     "was_correct":    body.was_correct,
    # })

    return {"resolved_branch": branch, "compound": compound}
