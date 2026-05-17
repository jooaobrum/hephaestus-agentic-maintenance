"""
investigate/graph.py — PHILM POC
─────────────────────────────────
6 nodes, full flow from day one.

  context_gathering → investigate → hypothesize → validate → discriminate → recommend

Evolution seam: sensor_id on questions.
  POC:    None → ask_user  |  Future: set → auto-evaluate, skip ask_user
"""

from __future__ import annotations
import asyncio
import json
import re
import uuid
from typing import AsyncGenerator

import anthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command

from .models import (
    AgentEvent,
    EventType,
    Phase,
    MachineContext,
    EvidenceItem,
    CompoundFinding,
    HypothesisBranch,
    ValidationQuestion,
    DiscriminationQuestion,
    InvestigationState,
)
from .tools import execute_tool  # pgvector RAG — implement separately

client = anthropic.AsyncAnthropic()
checkpointer = MemorySaver()


# ══════════════════════════════════════════════════════════════════
# Tools
# ══════════════════════════════════════════════════════════════════

ASK_USER = {
    "name": "ask_user",
    "description": (
        "Ask the technician a question that data sources cannot answer. "
        "Prefer structured options. Never ask what you can look up."
    ),
    "input_schema": {
        "type": "object",
        "required": ["question", "intent"],
        "properties": {
            "question": {"type": "string"},
            "intent": {"type": "string"},
            "options": {"type": "array", "items": {"type": "string"}},
        },
    },
}

SUBMIT_CONTEXT = {
    "name": "submit_context",
    "description": (
        "Submit gathered context to start investigation. "
        "Minimum: symptom + one of family/period/tried_actions."
    ),
    "input_schema": MachineContext.model_json_schema(),
}

SUBMIT_FINDING = {
    "name": "submit_compound_finding",
    "description": (
        "Submit root cause hypotheses. Always call this — even for a single cause. "
        "1 branch = high confidence. 2–4 branches = genuinely ambiguous."
    ),
    "input_schema": {
        "type": "object",
        "required": ["symptom", "branches", "recommended_first_check"],
        "properties": {
            "symptom": {"type": "string"},
            "recommended_first_check": {"type": "string"},
            "branches": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "required": [
                        "id",
                        "cause",
                        "confidence",
                        "discriminating_check",
                        "action_plan",
                    ],
                    "properties": {
                        "id": {"type": "string"},
                        "cause": {"type": "string"},
                        "confidence": {"type": "number"},
                        "escalate_to_expert": {
                            "type": "boolean",
                            "description": "True when fewer than 3 historical cases found.",
                        },
                        "discriminating_check": {
                            "type": "string",
                            "description": "Single observation that confirms or kills this branch.",
                        },
                        "action_plan": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["priority", "action"],
                                "properties": {
                                    "priority": {
                                        "type": "string",
                                        "enum": [
                                            "immediate",
                                            "short_term",
                                            "preventive",
                                        ],
                                    },
                                    "action": {"type": "string"},
                                    "component": {"type": "string"},
                                    "parts_needed": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        },
    },
}

INVESTIGATE_TOOLS = [
    {
        "name": "search_procedures",
        "description": "Search maintenance procedures for this symptom and machine family.",
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "family": {"type": "string"},
            },
        },
    },
    {
        "name": "search_past_interventions",
        "description": "Search historical maintenance records for similar failures.",
        "input_schema": {
            "type": "object",
            "required": ["query"],
            "properties": {
                "query": {"type": "string"},
                "family": {"type": "string"},
            },
        },
    },
    ASK_USER,
    SUBMIT_FINDING,
]


# ══════════════════════════════════════════════════════════════════
# Evolution seam helper
# ══════════════════════════════════════════════════════════════════


async def _resolve(
    q: ValidationQuestion | DiscriminationQuestion,
    phase: Phase,
) -> tuple[str, AgentEvent | None]:
    """
    POC:    sensor_id is None → interrupt and ask.
    Future: sensor_id set → call tool, return answer automatically.
    """
    if q.sensor_id:
        answer, _ = await execute_tool(
            "evaluate_sensor_check", {"sensor_id": q.sensor_id}, ""
        )
        return str(answer), None

    iid = str(uuid.uuid4())
    evt = AgentEvent(
        type=EventType.INTERRUPT,
        question=q.question,
        options=q.options,
        interrupt_id=iid,
        meta={"phase": phase.value},
    )
    answer = interrupt({"interrupt_id": iid, "question": q.question})
    return str(answer), evt


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════


def _ctx_block(ctx: MachineContext) -> str:
    lines = [
        f"  Symptom:    {ctx.symptom}",
        f"  Machine ID: {ctx.machine_id or 'unknown'}",
        f"  Family:     {ctx.family or 'unknown'}",
        f"  Period:     {ctx.period or 'unknown'}",
        f"  Tried:      {ctx.tried_actions or 'nothing reported'}",
        f"  Last maint: {ctx.recent_maintenances or 'unknown'}",
        f"  Conditions: {ctx.operating_conditions or 'unknown'}",
    ]
    if ctx.additional_context:
        lines.append(f"  Notes:      {ctx.additional_context}")
    return "\n".join(lines)


def _parse_json(text: str) -> dict:
    clean = re.sub(r"```(?:json)?", "", text).strip(" `\n")
    return json.loads(clean)


async def _interrupt_user(
    question: str, options: list[str] | None, phase: Phase, intent: str = ""
) -> tuple[str, AgentEvent]:
    iid = str(uuid.uuid4())
    evt = AgentEvent(
        type=EventType.INTERRUPT,
        question=question,
        options=options,
        interrupt_id=iid,
        meta={"intent": intent, "phase": phase.value},
    )
    answer = interrupt({"interrupt_id": iid, "question": question})
    return str(answer), evt


# ══════════════════════════════════════════════════════════════════
# Node 1 — CONTEXT_GATHERING
# ══════════════════════════════════════════════════════════════════

CONTEXT_SYS = """You are a maintenance AI gathering context before investigating a failure.

You need:
  - Symptom (what exactly is happening)
  - Machine family / type
  - When it started (period)
  - What has been tried already
  - Last maintenance performed
  - Operating conditions

Ask naturally — group related questions, never re-ask answered fields.
Minimum to proceed: symptom + one more field.
Call submit_context when ready."""


async def context_gathering_node(state: InvestigationState) -> dict:
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.CONTEXT_GATHERING)]
    messages = list(state.messages) or [
        {"role": "user", "content": "I need help troubleshooting a machine problem."}
    ]

    while True:
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=CONTEXT_SYS,
            tools=[ASK_USER, SUBMIT_CONTEXT],
            messages=messages,
        )

        content = []
        done = False

        for block in response.content:
            if block.type == "text":
                events.append(AgentEvent(type=EventType.TEXT_DELTA, content=block.text))
                content.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                content.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )

                if block.name == "ask_user":
                    answer, evt = await _interrupt_user(
                        block.input["question"],
                        block.input.get("options"),
                        Phase.CONTEXT_GATHERING,
                        block.input.get("intent", ""),
                    )
                    events.append(evt)
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": answer,
                                }
                            ],
                        }
                    )
                    content = []

                elif block.name == "submit_context":
                    ctx = MachineContext(**block.input)
                    events.append(AgentEvent(type=EventType.CONTEXT_READY, context=ctx))
                    messages.append({"role": "assistant", "content": content})
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": "Context recorded. Starting investigation.",
                                }
                            ],
                        }
                    )
                    done = True
                    break

        if done:
            return {
                "phase": Phase.INVESTIGATE,
                "machine_context": ctx,
                "machine_id": ctx.machine_id or "unknown",
                "messages": messages,
                "_events": events,
            }
        if content:
            messages.append({"role": "assistant", "content": content})


# ══════════════════════════════════════════════════════════════════
# Node 2 — INVESTIGATE
# ══════════════════════════════════════════════════════════════════


def _investigate_sys(ctx: MachineContext) -> str:
    return f"""You are a maintenance AI performing root cause analysis.

CONTEXT:
{_ctx_block(ctx)}

TOOLS:
  search_procedures         → standard procedures for this symptom / machine family
  search_past_interventions → historical records and past resolutions
  ask_user                  → fill gaps that data sources cannot answer

HYPOTHESIS TRACKING — write this after every tool result or user answer:

  [HYPOTHESIS UPDATE]
  Cause A — XX%: brief reason
  Cause B — XX%: brief reason

Update explicitly. Do not silently accumulate evidence.

RULES:
  1. search_procedures first, then search_past_interventions.
  2. ask_user only when both sources leave a genuine gap.
  3. If the technician asks a question mid-investigation, answer it with
     search_procedures then continue.
  4. Call submit_compound_finding when hypothesis is stable (2 iterations unchanged),
     or after 6 tool calls, or when one hypothesis exceeds 85%.
  5. escalate_to_expert = true when fewer than 3 historical cases found.
  6. discriminating_check = the single observation that confirms or kills this branch."""


async def investigate_node(state: InvestigationState) -> dict:
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.INVESTIGATE)]
    ctx = state.machine_context
    messages = list(state.messages)
    evidence = list(state.evidence)
    iteration = 0

    # Parallel baseline — search both sources simultaneously
    events.append(
        AgentEvent(
            type=EventType.TEXT_DELTA,
            content="Loading procedures and intervention history...",
        )
    )
    baseline = await asyncio.gather(
        execute_tool(
            "search_procedures",
            {"query": ctx.symptom, "family": ctx.family or ""},
            state.machine_id,
        ),
        execute_tool(
            "search_past_interventions",
            {"query": ctx.symptom, "family": ctx.family or ""},
            state.machine_id,
        ),
        return_exceptions=True,
    )
    for name, res in zip(["search_procedures", "search_past_interventions"], baseline):
        if isinstance(res, Exception):
            continue
        result, ev = res
        if ev:
            evidence.append(ev)
        events.append(
            AgentEvent(
                type=EventType.TOOL_RESULT,
                tool_name=name,
                tool_result=json.dumps(result)[:400],
            )
        )
        messages.append(
            {"role": "user", "content": f"[Baseline — {name}]:\n{json.dumps(result)}"}
        )

    # ReAct loop
    system = _investigate_sys(ctx)

    while iteration < 8:
        iteration += 1
        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=system,
            tools=INVESTIGATE_TOOLS,
            messages=messages,
        )

        acontent = []
        tool_res = []

        for block in response.content:
            if block.type == "text":
                tag = (
                    EventType.TEXT_DELTA
                    if "[HYPOTHESIS UPDATE]" not in block.text
                    else EventType.TEXT_DELTA
                )
                events.append(AgentEvent(type=tag, content=block.text))
                acontent.append({"type": "text", "text": block.text})

            elif block.type == "tool_use":
                acontent.append(
                    {
                        "type": "tool_use",
                        "id": block.id,
                        "name": block.name,
                        "input": block.input,
                    }
                )
                events.append(
                    AgentEvent(type=EventType.TOOL_CALL, tool_name=block.name)
                )

                if block.name == "ask_user":
                    answer, evt = await _interrupt_user(
                        block.input["question"],
                        block.input.get("options"),
                        Phase.INVESTIGATE,
                        block.input.get("intent", ""),
                    )
                    events.append(evt)
                    ev = EvidenceItem(
                        id=f"usr_{uuid.uuid4().hex[:6]}",
                        type="user_answer",
                        label=block.input["question"][:60],
                        content=answer,
                        source="ask_user",
                    )
                    evidence.append(ev)
                    events.append(
                        AgentEvent(
                            type=EventType.TOOL_RESULT,
                            tool_name="ask_user",
                            tool_result=answer,
                            evidence=ev,
                        )
                    )
                    messages.append({"role": "assistant", "content": acontent})
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": block.id,
                                    "content": answer,
                                }
                            ],
                        }
                    )
                    acontent = []
                    tool_res = []
                    iteration -= 1
                    continue

                if block.name == "submit_compound_finding":
                    branches = [
                        HypothesisBranch(**b) for b in block.input.pop("branches")
                    ]
                    compound = CompoundFinding(
                        session_id=state.session_id,
                        machine_id=state.machine_id,
                        branches=branches,
                        **block.input,
                    )
                    events.append(
                        AgentEvent(type=EventType.COMPOUND_FINDING, compound=compound)
                    )
                    return {
                        "phase": Phase.HYPOTHESIZE,
                        "compound": compound,
                        "evidence": evidence,
                        "messages": messages,
                        "_events": events,
                    }

                result, ev = await execute_tool(
                    block.name, block.input, state.machine_id
                )
                if ev:
                    evidence.append(ev)
                events.append(
                    AgentEvent(
                        type=EventType.TOOL_RESULT,
                        tool_name=block.name,
                        tool_result=json.dumps(result)[:400],
                        evidence=ev,
                    )
                )
                tool_res.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result),
                    }
                )

        if acontent:
            messages.append({"role": "assistant", "content": acontent})
        if tool_res:
            messages.append({"role": "user", "content": tool_res})
        if response.stop_reason == "end_turn":
            break

    return {
        "phase": Phase.HYPOTHESIZE,
        "evidence": evidence,
        "messages": messages,
        "_events": events,
    }


# ══════════════════════════════════════════════════════════════════
# Node 3 — HYPOTHESIZE  (revision node)
# ══════════════════════════════════════════════════════════════════


async def hypothesize_node(state: InvestigationState) -> dict:
    """Called on rejection or when validate finds an alternative hypothesis."""
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.HYPOTHESIZE)]
    ctx = state.machine_context
    reason = state.rejection_reason or "Revision requested."

    response = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system=f"""You are revising a root cause hypothesis.

CONTEXT:
{_ctx_block(ctx)}
REVISION REASON: {reason}
EVIDENCE:
{json.dumps([e.model_dump() for e in state.evidence], indent=2)}

Call submit_compound_finding with a revised hypothesis.
All branches must reflect the evidence and the revision reason.""",
        tools=[SUBMIT_FINDING],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": "Revise hypothesis now."}],
    )

    compound = state.compound
    for block in response.content:
        if block.type == "text":
            events.append(AgentEvent(type=EventType.TEXT_DELTA, content=block.text))
        elif block.type == "tool_use":
            branches = [HypothesisBranch(**b) for b in block.input.pop("branches")]
            compound = CompoundFinding(
                session_id=state.session_id,
                machine_id=state.machine_id,
                branches=branches,
                **block.input,
            )
            events.append(
                AgentEvent(type=EventType.COMPOUND_FINDING, compound=compound)
            )

    return {
        "phase": Phase.VALIDATE,
        "compound": compound,
        "rejection_reason": None,
        "_events": events,
    }


# ══════════════════════════════════════════════════════════════════
# Node 4 — VALIDATE
# ══════════════════════════════════════════════════════════════════


async def validate_node(state: InvestigationState) -> dict:
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.VALIDATE)]
    compound = state.compound
    confidence = compound.top_branch.confidence

    # Generate questions
    r = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=f"""Generate 2-3 validation questions for this hypothesis.

TOP HYPOTHESIS: {compound.top_branch.cause} ({confidence:.0%})
OTHER BRANCHES: {[b.cause for b in compound.branches[1:]]}

Each question must target a specific evidence gap.
Return JSON only:
{{
  "questions": [
    {{
      "id": "v1",
      "question": "...",
      "options": ["A", "B", "C"],
      "confidence_deltas": {{"A": 0.10, "B": -0.08, "C": 0}},
      "sensor_id": null
    }}
  ]
}}""",
        messages=[{"role": "user", "content": "Generate questions."}],
    )

    try:
        questions = [
            ValidationQuestion(**q) for q in _parse_json(r.content[0].text)["questions"]
        ]
    except Exception:
        questions = []

    alt_found = False
    for i, q in enumerate(questions):
        answer, evt = await _resolve(q, Phase.VALIDATE)
        if evt:
            events.append(evt)
        events.append(
            AgentEvent(
                type=EventType.TEXT_DELTA,
                content=f"[VALIDATE Q{i+1}] {q.question} → {answer}",
            )
        )
        delta = q.confidence_deltas.get(answer, 0.0)
        confidence = max(0.0, min(1.0, confidence + delta))
        events.append(
            AgentEvent(
                type=EventType.BRANCH_UPDATE,
                meta={"confidence": confidence, "delta": delta, "step": i + 1},
            )
        )
        if delta < -0.15:
            alt_found = True

    # Holistic re-assessment
    ra = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=200,
        system=f"""Reassess hypothesis after validation.
Hypothesis: {compound.top_branch.cause}
Running confidence: {confidence:.2f}

Return JSON only:
{{"final_confidence": 0.0-1.0,
  "routing": "discriminate|re_investigate|re_hypothesize",
  "reason": "one sentence"}}""",
        messages=[{"role": "user", "content": "Reassess."}],
    )
    try:
        ra_data = _parse_json(ra.content[0].text)
    except Exception:
        ra_data = {
            "final_confidence": confidence,
            "routing": "discriminate",
            "reason": "fallback",
        }

    compound.branches[0].confidence = ra_data["final_confidence"]
    if alt_found:
        ra_data["routing"] = "re_hypothesize"

    events.append(AgentEvent(type=EventType.COMPOUND_FINDING, compound=compound))
    events.append(
        AgentEvent(
            type=EventType.TEXT_DELTA,
            content=f"Validation done — {ra_data['final_confidence']:.0%}. {ra_data['reason']}",
        )
    )

    next_phase = {
        "discriminate": Phase.DISCRIMINATE,
        "re_investigate": Phase.INVESTIGATE,
        "re_hypothesize": Phase.HYPOTHESIZE,
    }.get(ra_data["routing"], Phase.DISCRIMINATE)

    return {
        "phase": next_phase,
        "compound": compound,
        "rejection_reason": ra_data["reason"]
        if ra_data["routing"] != "discriminate"
        else None,
        "_events": events,
    }


# ══════════════════════════════════════════════════════════════════
# Node 5 — DISCRIMINATE
# ══════════════════════════════════════════════════════════════════


async def discriminate_node(state: InvestigationState) -> dict:
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.DISCRIMINATE)]
    compound = state.compound

    if len(compound.branches) <= 1:
        return {"phase": Phase.RECOMMEND, "_events": events}

    branches_txt = "\n".join(
        f"  [{b.id}] {b.cause} — {b.confidence:.0%} | check: {b.discriminating_check}"
        for b in compound.branches
    )
    r = await client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1200,
        system=f"""Generate 2-3 questions to discriminate between competing hypotheses.

BRANCHES:
{branches_txt}

Each question must be able to eliminate at least one branch.
Return JSON only:
{{
  "questions": [
    {{
      "id": "d1",
      "question": "...",
      "options": ["A", "B", "C"],
      "branch_impacts": {{"A": {{"b1": 0.15, "b2": -0.12}}}},
      "sensor_id": null
    }}
  ]
}}""",
        messages=[{"role": "user", "content": "Generate questions."}],
    )

    try:
        questions = [
            DiscriminationQuestion(**q)
            for q in _parse_json(r.content[0].text)["questions"]
        ]
    except Exception:
        questions = []

    for i, q in enumerate(questions):
        answer, evt = await _resolve(q, Phase.DISCRIMINATE)
        if evt:
            events.append(evt)
        events.append(
            AgentEvent(
                type=EventType.TEXT_DELTA,
                content=f"[DISCRIM Q{i+1}] {q.question} → {answer}",
            )
        )

        impacts = q.branch_impacts.get(answer, {})
        eliminated = []
        for bid, delta in impacts.items():
            eliminated.extend(compound.apply_delta(bid, delta))

        events.append(
            AgentEvent(
                type=EventType.BRANCH_UPDATE,
                compound=compound,
                eliminated=eliminated or None,
            )
        )

        if compound.is_converged() or len(compound.branches) == 1:
            events.append(
                AgentEvent(
                    type=EventType.TEXT_DELTA,
                    content=f"Converged — {compound.top_branch.cause} "
                    f"({compound.top_branch.confidence:.0%})",
                )
            )
            break

    events.append(AgentEvent(type=EventType.COMPOUND_FINDING, compound=compound))
    return {"phase": Phase.RECOMMEND, "compound": compound, "_events": events}


# ══════════════════════════════════════════════════════════════════
# Node 6 — RECOMMEND
# ══════════════════════════════════════════════════════════════════


def _recommend_sys(compound: CompoundFinding, ctx: MachineContext) -> str:
    return f"""You are discussing a maintenance finding with a technician.

MACHINE: {ctx.machine_id or 'unknown'} — {ctx.symptom}

FINDING:
{compound.model_dump_json(indent=2)}

The technician can:
  - Ask questions about any branch or action
  - Provide constraints ("we don't have that part in stock")
  - Say "approve" to finalize
  - Say "reject" to restart analysis
  - Say "branch X was correct" to record resolution after fixing

If a constraint changes the action plan, call submit_compound_finding with the revision.
Respond concisely. Be specific."""


async def recommend_node(state: InvestigationState) -> dict:
    events = [AgentEvent(type=EventType.PHASE_CHANGE, phase=Phase.RECOMMEND)]
    compound = state.compound
    ctx = state.machine_context
    messages = list(state.messages)

    while True:
        iid = str(uuid.uuid4())
        events.append(
            AgentEvent(
                type=EventType.INTERRUPT,
                interrupt_id=iid,
                meta={"phase": Phase.RECOMMEND.value},
            )
        )
        user_input = str(interrupt({"interrupt_id": iid}))
        messages.append({"role": "user", "content": user_input})

        response = await client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=_recommend_sys(compound, ctx),
            tools=[SUBMIT_FINDING],
            messages=messages,
        )

        acontent = []
        for block in response.content:
            if block.type == "text":
                events.append(AgentEvent(type=EventType.TEXT_DELTA, content=block.text))
                acontent.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                branches = [HypothesisBranch(**b) for b in block.input.pop("branches")]
                compound = CompoundFinding(
                    session_id=state.session_id,
                    machine_id=state.machine_id,
                    branches=branches,
                    **block.input,
                )
                events.append(
                    AgentEvent(type=EventType.COMPOUND_FINDING, compound=compound)
                )

        messages.append({"role": "assistant", "content": acontent})

        last = " ".join(
            b.get("text", "") for b in acontent if b.get("type") == "text"
        ).lower()
        if any(
            kw in last
            for kw in [
                "approved",
                "recorded",
                "finalized",
                "submitted",
                "resolution confirmed",
            ]
        ):
            break

    return {
        "phase": Phase.DONE,
        "compound": compound,
        "messages": messages,
        "_events": events,
    }


# ══════════════════════════════════════════════════════════════════
# Routing
# ══════════════════════════════════════════════════════════════════


def _route_validate(state: InvestigationState) -> str:
    p = state.phase
    if p == Phase.INVESTIGATE:
        return "investigate"
    if p == Phase.HYPOTHESIZE:
        return "hypothesize"
    return "discriminate"


# ══════════════════════════════════════════════════════════════════
# Graph
# ══════════════════════════════════════════════════════════════════


def build_graph():
    g = StateGraph(InvestigationState)

    g.add_node("context_gathering", context_gathering_node)
    g.add_node("investigate", investigate_node)
    g.add_node("hypothesize", hypothesize_node)
    g.add_node("validate", validate_node)
    g.add_node("discriminate", discriminate_node)
    g.add_node("recommend", recommend_node)

    g.add_edge(START, "context_gathering")
    g.add_edge("context_gathering", "investigate")
    g.add_edge("investigate", "hypothesize")
    g.add_edge("hypothesize", "validate")

    g.add_conditional_edges(
        "validate",
        _route_validate,
        {
            "discriminate": "discriminate",
            "investigate": "investigate",
            "hypothesize": "hypothesize",
        },
    )

    g.add_edge("discriminate", "recommend")
    g.add_edge("recommend", END)

    return g.compile(checkpointer=checkpointer)


investigation_graph = build_graph()


# ══════════════════════════════════════════════════════════════════
# SSE generators
# ══════════════════════════════════════════════════════════════════


async def stream_investigation(
    session_id: str,
    machine_id: str,
    initial_message: str | None = None,
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 60}
    initial = {
        "machine_id": machine_id,
        "session_id": session_id,
        "phase": Phase.CONTEXT_GATHERING,
        "messages": []
        if not initial_message
        else [{"role": "user", "content": initial_message}],
    }
    async for chunk in investigation_graph.astream(initial, config=config):
        for _, out in chunk.items():
            if isinstance(out, dict):
                for evt in out.get("_events", []):
                    yield evt.to_sse()
    yield AgentEvent(type=EventType.DONE).to_sse()


async def resume_investigation(
    session_id: str, message: str
) -> AsyncGenerator[str, None]:
    config = {"configurable": {"thread_id": session_id}}
    async for chunk in investigation_graph.astream(
        Command(resume=message), config=config
    ):
        for _, out in chunk.items():
            if isinstance(out, dict):
                for evt in out.get("_events", []):
                    yield evt.to_sse()
    yield AgentEvent(type=EventType.DONE).to_sse()
