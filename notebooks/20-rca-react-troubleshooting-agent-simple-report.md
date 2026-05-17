# Report: Simple Conversational RCA Agent

A walkthrough of [20-rca-react-troubleshooting-agent-simple.ipynb](20-rca-react-troubleshooting-agent-simple.ipynb) — a ReAct-style agent that runs a Root Cause Analysis (RCA) investigation conversationally.

---

## 1. High-Level Design

The agent is a **two-pass LangGraph ReAct loop** with a strict separation of concerns:

| Pass | Role | LLM binding | Output |
|---|---|---|---|
| **THINKER** | Decides which tools / schemas to call given the current ledger | `_llm_with_tools` (executable tools + ledger schemas, `tool_choice="auto"`) | Tool calls only — no prose |
| **SPEAKER** | Writes the user-facing reply against the *post-mutation* ledger | `_speaker_llm` (only `FinalResponse`, `tool_choice="FinalResponse"`) | One `FinalResponse(answer=...)` |

Why split them? It makes the user message **physically incapable of lying about state**: the speaker only sees the ledger after mutations have been applied, so it cannot claim a hypothesis exists that the thinker forgot to create (or vice versa).

---

## 2. Core Data Model

Defined in the **State & Models** cell:

- **`Hypothesis`** — `hypothesis_id`, `statement`, `explanation`, `confidence` (0-1), `status` (`ACTIVE`/`LIKELY`/`CONFIRMED`/`REJECTED`), `sources` (list of source ids).
- **`SavedCase`** — closed-out investigation: `case_id`, `machine_id`, `symptom`, `root_cause`, `actions`.
- **`AgentState`** — `messages` (append-only via `Annotated[..., add]`), `hypotheses`, `confirmed_root_cause`, `saved_case`, `answer`, `final_answer` (turn-end flag), `iteration`.
- **`FinalResponse`** — Pydantic schema the speaker is forced to call; its `answer` field is the literal user reply.

---

## 3. Tools and Schemas

Two distinct categories, both bound to the thinker via `bind_tools`:

**Executable tools** (run real queries through `ToolNode`):
- `check_machine_exists`
- `get_formatted_procedure_context` — manufacturer procedure RAG
- `get_recent_formatted_cm_context` — recent CM history
- `query_known_issues_graph` — past interventions graph
- `get_intervention_detail` — drill into one INT-... case

**Ledger schemas** (Pydantic-only — never executed; mutations applied locally in `agent_node`):
- `CreateHypothesis`
- `UpdateHypothesis`
- `ConfirmRootCause`
- `SaveCase`
- `SummarizeInvestigation`

`SCHEMA_NAMES = LEDGER_NAMES | {"FinalResponse"}` is used by `_strip_schema_calls` to drop schema calls before they reach `ToolNode` (which would try to execute them and crash).

---

## 4. Ledger Helpers

In the **Helpers** cell:

- **`_refresh_status(h)`** — auto-derives status from confidence: `≤0.05 → REJECTED`, `≥0.85 → LIKELY`, else `ACTIVE`. Skips if already `CONFIRMED` (terminal, set only by `ConfirmRootCause`).
- **`_render_ledger(...)`** — markdown table sorted by confidence; rendered into both system prompts every turn.
- **`_merge_sources`** — append-unique for source ids (preserves order).
- **`_clamp`** — keeps confidence in `[0, 1]`.
- **`_apply_ledger_mutations(state, response)`** — the heart of state mutation. Walks `response.tool_calls`, dispatches by name, and:
  - On `CreateHypothesis` with an existing id → treats it as an update (idempotent).
  - On `SaveCase` *without* a prior `ConfirmRootCause` → silently refuses (the LLM sees no `saved_case` and is forced to confirm first).
  - Defensively reads every field — bad/incomplete tool calls are skipped, not raised.

---

## 5. Agent Node — One ReAct Step

`agent_node` does:

1. **Thinker** — Builds `_build_system_message` (system prompt + ledger snapshot + status) and invokes `_llm_with_tools`.
2. **Apply mutations** — `_apply_ledger_mutations` returns updated `(hypotheses, confirmed, saved)`.
3. **Strip schema calls** — `_strip_schema_calls` separates real tool calls from schema calls. Only real tool calls remain on the sanitized `AIMessage`.
4. **Branch:**
   - **If executable tools are pending** → return the sanitized message; the graph will route to `tools`, run them, then loop back. Speaker is **not** invoked this turn.
   - **Else** → call `_speak(...)` which builds a SPEAKER system message (with the post-mutation ledger as ground truth) and forces a `FinalResponse`. The thinker's empty AIMessage is replaced by the spoken `AIMessage` to keep history clean.
5. **`final_answer=True`** signals the router to end the turn.

`_fallback_answer` is a deterministic safety net if the speaker fails to produce a reply.

---

## 6. Routing

`route_after_agent` ends the turn when:
- `final_answer` is set (speaker ran), or
- `iteration ≥ MAX_ITERATIONS` (8), or
- The last message has no tool calls.

Otherwise it routes to `tools`. The graph: `START → agent ⇄ tools → END`, compiled with `MemorySaver` so each `thread_id` keeps its own ledger across turns.

---

## 7. The System Prompts

**Thinker prompt** (`SYSTEM_PROMPT`) is a strict playbook:
- §1 Validate inputs (machine + symptom).
- §2 Gather first evidence — `get_formatted_procedure_context` AND `query_known_issues_graph` in parallel.
- §3 **STRICT FILTERING** — drop chunks for other machines, other fault codes, other symptoms before hypothesizing.
- §3a **Dedup** — read the ledger first; if a cause already exists, `UpdateHypothesis`, never `CreateHypothesis` again.
- §4 Each `source_id` must actually mention the cause it's attached to.
- §5 Drill down with `get_intervention_detail` when graph cites a promising INT-... id.
- §6 User feedback only **refines** existing hypotheses (never creates new ones).
- §7-8 `ConfirmRootCause` / `SaveCase` only after explicit user yes.
- Each retrieval tool: at most once per investigation.

**Speaker prompt** (`_SPEAKER_SYSTEM`) is a tone/format guide with templates A–H matching turn type (procedure retrieved / graph retrieved / hypothesizing / user reported result / one cause dominates / user confirmed / accepted plan / asked for recap).

---

## 8. The Golden Path (from the test run)

Concrete trace from the notebook test, machine `HX-200`, symptom *high oil temperature, >80C*:

### Turn 1 — User describes symptom
```
You: HX-200 has high oil temperature, >80C, started today
```
**Thinker** —
1. Calls `check_machine_exists("HX-200")` → tools node runs it → loops back.
2. Calls `get_formatted_procedure_context` and `query_known_issues_graph` **in parallel**.
3. Filters to `E-002: High Oil Temperature` (HX-200 only) and graph entries for the same symptom.
4. Emits 6 `CreateHypothesis` calls — one per distinct cause from the procedure's `Common Root Causes` + graph's `Potential Root Causes`. Sources merged when both mention the same cause (e.g. H1 cooler fouling: `PROC_REF:HX-200_Troubleshooting_Procedures:chunk#10` + `GRAPH:INT-2025-0062`).

**Speaker** — Composes Template A+B+C: opens with the procedure section, summarizes graph patterns, lists the 6 candidates, ends with a soft narrowing question.

**Resulting ledger**:
```
H1 ACTIVE 0.50 Cooler HE-501 fins fouled — PROC_REF:...:chunk#10, GRAPH:INT-2025-0062
H2 ACTIVE 0.50 Cooling fan motor failure
H3 ACTIVE 0.50 Ambient temperature >40C
H4 ACTIVE 0.50 Oil contaminated
H5 ACTIVE 0.50 Excessive cycle rate
H6 ACTIVE 0.50 Low oil level
```

### Turn 2 — User reports a check result confirming H1
**Thinker** — Per §6, calls `UpdateHypothesis(H1, +0.30, source_ids=["USER"])`. No new hypothesis.
**Speaker** — Template D: acknowledges, then since H1 still isn't yet `LIKELY` (now 0.80), offers to draft an action plan. (Note: in the actual run the speaker jumped slightly ahead of the threshold rule — see §10.)

```
H1 ACTIVE 0.80 ... + USER
```

### Turn 3 — User accepts the action plan offer
**Thinker** — Calls `ConfirmRootCause(H1)`. `_apply_ledger_mutations` marks H1 `CONFIRMED conf=1.0` and every other hypothesis `REJECTED`.
**Speaker** — Template G: numbered 6-step action plan (LOTO → inspect → clean fins → verify fan → restart → monitor), ends with "Want me to save this case?".

### Turn 4 — User says yes, save
**Thinker** — Calls `SaveCase(machine_id="HX-200", symptom="High oil temperature", root_cause="Cooler HE-501 fins fouled", actions=[...])`. `_apply_ledger_mutations` checks `confirmed != ""` → creates `SavedCase` with id `CASE-0B54667F`.
**Speaker** — Final acknowledgement.

The investigation is now closed; the ledger and saved case are persisted in the `MemorySaver` checkpoint under that thread id.

---

## 9. Why the Two-Pass Design Matters

- **No state lies.** The speaker can only describe what's already in the ledger, because the ledger is rebuilt before the speaker runs.
- **Idempotent mutations.** `CreateHypothesis` on an existing id becomes an update — protects against thinker churn.
- **`SaveCase` gated on `ConfirmRootCause`.** A "save without confirm" is silently dropped, then the next thinker turn sees nothing was saved and re-tries the right way.
- **Schema calls never reach `ToolNode`.** `_strip_schema_calls` filters them, so `ToolNode` only ever sees real, executable tools.

---

## 10. Notable Observations from the Sample Run

- The deserialization warnings (`Deserializing unregistered type __main__.Hypothesis`) are because `Hypothesis` is defined in the notebook's `__main__` and `MemorySaver` flags this for future strict mode. Fix is to register the type or move it to a module.
- The thinker called `ConfirmRootCause` at H1 confidence ≈0.80 (below the 0.85 `LIKELY` threshold) because the user's "yes, sounds good" satisfied §7's *explicit yes* rule. The threshold is a **trigger to ask**, not a hard gate to confirm — confirmation is gated only on the user's most recent message.
- Sources are correctly attached: H1 ends with three sources (`PROC_REF`, `GRAPH`, `USER`) — exactly the design intent of §4.
