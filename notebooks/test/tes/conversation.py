"""
investigate/conversation.py — PHILM POC
Haiku intent classifier used by router on /reply calls.
Only needed when you want to route outside the graph (e.g. mark_resolved).
The recommend_node handles approve/reject/constraint conversationally without this.
"""
from __future__ import annotations
import json, re
from typing import Optional, Any
from pydantic import BaseModel
import anthropic

from .models import InvestigationState

client = anthropic.AsyncAnthropic()


class IntentResult(BaseModel):
    intent:    str
    branch_id: Optional[str] = None
    confidence: float = 1.0


SYSTEM = """Classify a maintenance technician's message during a finding review.

Intents:
  follow_up      Question or request for more info.
  approve        User approves: "approve", "looks right", "correct", "confirm"
  reject         User rejects: "wrong", "start over", "I disagree"
  mark_resolved  User confirms which branch fixed the issue:
                 "branch 1 was correct", "it was the bearing", "cause 2 fixed it"
  other          Anything else

Return JSON only: {"intent": "...", "branch_id": null, "confidence": 0.0-1.0}"""


async def classify(message: str, state: InvestigationState) -> IntentResult:
    """Fast Haiku routing call. Falls back to follow_up on any error."""
    try:
        r = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=128,
            system=SYSTEM,
            messages=[{"role": "user", "content": message}],
        )
        raw = re.sub(r"```(?:json)?", "", r.content[0].text).strip(" `\n")
        return IntentResult(**json.loads(raw))
    except Exception:
        return IntentResult(intent="follow_up", confidence=0.5)
