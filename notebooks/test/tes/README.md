# PHILM — AI Maintenance Diagnosis Agent
**POC · 1 Data Scientist · 2 Months**

---

## Problem

When a machine fails, technicians at ST Microelectronics must manually gather context
from multiple people, search through procedure documents and intervention history,
form a hypothesis based on experience, and decide on corrective action — often without
a structured validation step.

This process is slow, inconsistent across sites, and loses knowledge when experienced
technicians leave. Recurring failures are not prevented because root causes are not
captured in a structured, searchable form.

**PHILM** conducts a structured root cause analysis through conversation with the
technician. It gathers context, searches procedures and intervention history, builds
ranked hypotheses with action plans, validates and narrows them down through targeted
questions, and produces a structured finding that seeds a growing knowledge base.

---

## Project Architecture

```
┌──────────────────────────────────────────┐
│  Frontend  (React / Next.js)             │
│  Investigation view · Fleet dashboard    │
│  SSE client — useInvestigation.ts        │
└────────────────┬─────────────────────────┘
                 │ SSE + REST
┌────────────────▼─────────────────────────┐
│  Backend  (FastAPI)                      │
│  router.py — 5 endpoints                 │
│  In-memory sessions (Redis in prod)      │
└────────────────┬─────────────────────────┘
                 │ LangGraph
┌────────────────▼─────────────────────────┐
│  Agent  (LangGraph + Claude Sonnet)      │
│  graph.py        6 nodes                 │
│  models.py       Pydantic schemas        │
│  conversation.py Haiku intent routing    │
└──────────┬─────────────────┬─────────────┘
           │ pgvector RAG    │ Anthropic API
    ┌──────▼──────┐   ┌──────▼──────────────┐
    │ PostgreSQL   │   │ Claude Sonnet 4      │
    │ procedures   │   │ Claude Haiku (route) │
    │ interventions│   └──────────────────────┘
    └─────────────┘
```

| Layer | Choice | Reason |
|---|---|---|
| Agent graph | LangGraph | interrupt() for conversation, checkpointing, clear evolution path |
| LLM | Claude Sonnet 4 | Best reasoning for RCA |
| Routing | Claude Haiku | Fast cheap classification calls |
| Vector DB | PostgreSQL + pgvector | Already in stack |
| Backend | FastAPI | SSE support, async |
| Sessions | In-memory dict | Sufficient for POC |

---

## Agent Architecture

### Flow

```
START
  │
  ▼
CONTEXT_GATHERING     Agent interviews the technician.
  │                   Collects: symptom, machine family, period, tried actions,
  │                   recent maintenances, operating conditions.
  │                   Uses ask_user in a loop → submit_context.
  ▼
INVESTIGATE           Parallel baseline + ReAct loop.
  │                   Phase 1: search_procedures + search_past_interventions in parallel.
  │                   Phase 2: targeted follow-up. ask_user fills what data can't answer.
  │                   Agent tracks hypotheses explicitly after each evidence piece.
  │                   Ends with submit_compound_finding.
  ▼
HYPOTHESIZE           Revision node.
  │                   Called only on rejection or when validate finds an alternative.
  │                   Re-reasons over existing evidence + new context. No new tool calls.
  ▼
VALIDATE              Confirms the top hypothesis.
  │                   Generates 2-3 targeted questions on evidence gaps.
  │                   POC: all answered via ask_user.
  │                   Future: sensor_id set → auto-evaluated from data.
  │                   Routes: ok → DISCRIMINATE · collapsed → INVESTIGATE · alt → HYPOTHESIZE
  ▼
DISCRIMINATE          Narrows competing branches.
  │                   Generates 2-3 questions that eliminate branches.
  │                   Same evolution seam: ask_user now, sensor check later.
  │                   Exits early when converged or one branch left.
  ▼
RECOMMEND             Conversational approval.
  │                   Handles: questions, constraints, approve, reject, mark resolved.
  │                   Constraint → agent revises action plan inline.
  ▼
END → finding stored, seeds knowledge base
```

### Tool set (POC)

| Tool | Used in | Purpose |
|---|---|---|
| `ask_user` | All phases | Primary gap-filling mechanism |
| `search_procedures` | INVESTIGATE | RAG over procedure documents |
| `search_past_interventions` | INVESTIGATE | RAG over intervention history |
| `submit_context` | CONTEXT_GATHERING | Structured intake output |
| `submit_compound_finding` | INVESTIGATE / HYPOTHESIZE / RECOMMEND | Primary output |

### Evolution path

```
POC  → ask_user fills all gaps (validation, discrimination, investigation)
v1   → + query_alarm_history, query_active_sensors  (ask_user drops ~40%)
v2   → + query_sensor_trend, query_fft              (sensor discrimination)
v3   → + query_cmms, query_parts_stock              (constraint auto-answer)
```

The graph nodes and routing are identical across all versions.
Only the tool set and `sensor_id` fields on questions change.

### Key simplifications for POC

- `CompoundFinding` always — one output type, even for single-cause diagnoses
- `sensor_id` on every question — the evolution hook, null in POC
- No intent classifier in the hot path — `recommend_node` handles naturally
- No safety rule engine — kept as a simple list if needed
- No audience adaptation — single conversational style
- Sessions in-memory — no Redis needed for 2-month POC

---

## Files

```
philm/
  models.py        All Pydantic schemas
  graph.py         6 nodes + routing + SSE generators
  conversation.py  Haiku intent classifier (used by router for mark_resolved)
  router.py        5 FastAPI endpoints
  tools.py         pgvector RAG implementations  ← implement separately
```

---

## 8-Week Timeline

| Week | Work |
|---|---|
| 1–2 | Chunk + embed procedures and interventions into pgvector. Implement tools.py. |
| 3–4 | graph.py end-to-end with mock tools. Tune system prompts on real scenarios. |
| 5   | Connect real pgvector tools. First full investigation runs. |
| 6   | Frontend SSE integration. Session management. |
| 7   | End-to-end testing with real technician scenarios. Edge case handling. |
| 8   | Langfuse observability. Prompt tuning. Demo preparation. |
