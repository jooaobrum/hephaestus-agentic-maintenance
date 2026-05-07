# RCA Troubleshooting Agent — Coaching Guide

> This guide walks you through building notebook `21-rca-troubleshooting-simple.ipynb` step by step.
> For each step you'll see: **what to build**, **why it matters**, and **how to think about it**.
> Code is yours to write — this guide gives you the concepts, decisions, and traps.

---

## Big Picture: What Are We Building?

A troubleshooting copilot that behaves like a **detective having a conversation** — not a search engine.

The agent is a **minimal LangGraph ReAct loop** — three nodes, no phase routing, no separate ledger node:

| Node | What it does |
|---|---|
| `agent` | LLM call with tools bound. System prompt renders the full hypothesis ledger inline on every call. |
| `tool_node` | LangGraph's built-in `ToolNode`. Executes whatever tools the agent requested. |
| `human_feedback` | Pauses execution via `interrupt()` when the agent needs the user to discriminate between hypotheses. |

That's it. The ledger update happens **inside the system prompt template** — the agent is instructed to emit a structured JSON block alongside its narrative reply whenever it has new tool results to process. LangGraph extracts that block and applies it to state. No separate extractor node, no separate LLM call for ledger updates.

```
START → agent → (tool calls?) → tool_node → agent → ... → END
             → (need user?)  → human_feedback → agent → ...
```

State has three fields: `messages` (the conversation), `hypotheses` (the ledger), and `awaiting_user` (a flag set by the agent when it needs operator input to discriminate between hypotheses). Everything else — machine, symptom, period — lives in the conversation and is rendered back into the system prompt from there.

---

## Architecture Overview

```
User message
     │
     ▼
┌──────────┐     tool_calls     ┌───────────┐
│  agent   │ ─────────────────► │ tool_node │
│          │ ◄───────────────── │           │
│  system  │   tool results     └───────────┘
│  prompt  │
│  (ledger │◄──────────────────────────────────┐
│  inline) │                                   │
└──────────┘                                   │
     │                                         │
     │ need_user_input=true         ┌────────────────────┐
     ├────────────────────────────► │  human_feedback    │
     │                              │  interrupt() waits │
     │ no tool_calls,               │  for operator input│
     │ no user input needed         └────────────────────┘
     ▼
    END
```

The agent node rebuilds the system prompt — including the full hypothesis ledger — on **every single call**. So after `tool_node` returns results, or after the user answers a mid-flight question, the agent sees fresh context AND the current ledger in the same context window, and can update both its narrative and its hypothesis estimates in one shot.

---

## Step-by-Step Implementation

---

### STEP 1 — Imports and LLM Setup

**What to build:** Standard imports + one `ChatOpenAI` instance, bound with all tools.

**Decisions:**
- `temperature=0, seed=42` everywhere — deterministic, debuggable.
- One LLM instance bound to `ALL_TOOLS`. No phase-specific bindings — the system prompt controls which tools the agent should use, not the binding.

**Key LangGraph import:**
```python
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, SystemMessage
```

---

### STEP 2 — State

**What to build:** One `RCAState` with three fields.

```python
from typing import Annotated, List, Any
from operator import add
from pydantic import BaseModel, Field

class RCAState(BaseModel):
    messages: Annotated[List[Any], add] = []
    hypotheses: Annotated[List[Hypothesis], merge_hypotheses] = []
    awaiting_user: bool = False   # True when agent needs operator input to discriminate
```

**Why `Annotated[List, add]`?**
LangGraph merges state updates from each node using reducers. Without `add`, each node's return value *replaces* the list. With `add`, it appends. This is what gives multi-turn memory — messages accumulate across the whole conversation.

**`awaiting_user`:**
A plain `bool` — no reducer needed, last write wins. The agent node sets it to `True` when it emits `need_user_input: true` in the ledger update block. The `human_feedback` node resets it to `False` after capturing the user's answer. The router reads it to decide which exit to take.

**No `phase`, `machine`, `symptom`, `period` in state.**
Those are in the conversation. The system prompt extracts them from `messages` at render time. Storing them separately means keeping two sources of truth in sync — unnecessary complexity.

**`hypotheses` reducer — `merge_hypotheses`:**
A function you write that merges two hypothesis lists by ID, deduplicating evidence. LangGraph calls it whenever a node returns `{"hypotheses": [...]}`.

---

### STEP 3 — Hypothesis and Evidence Models

**What to build:** `Evidence` and `Hypothesis` Pydantic models, plus `merge_hypotheses`.

Keep them minimal:

```python
class Evidence(BaseModel):
    source_id: str
    snippet: str
    direction: str       # "supports" | "opposes"
    weight: float

class Hypothesis(BaseModel):
    id: str
    statement: str
    confidence: float = 0.5
    status: str = "ACTIVE"   # PRIOR | ACTIVE | RECURRENT | CONFIRMED | REJECTED
    evidence: list[Evidence] = []
```

**`merge_hypotheses(h1, h2)`:**
Called by LangGraph as the reducer for `hypotheses`. Merge by `id`: if the same ID appears in both, keep the one with the higher `last_updated_turn` (or merge evidence lists). Return a flat list capped at 5.

**Confidence thresholds** (apply after every update):
- `>= 0.80` → CONFIRMED
- `<= 0.20` → REJECTED
- `>= 0.25` and was PRIOR → ACTIVE

---

### STEP 4 — Tools

**What to build:** Import all tools, build `ALL_TOOLS` list and `TOOL_REGISTRY` dict.

No phase-gating on the binding — bind all tools to the LLM and let the **system prompt** restrict which ones to call per phase. This keeps the graph simple (one LLM, one tool node) while preserving phase discipline through the prompt.

```python
ALL_TOOLS = [get_current_date, check_machine_exists, get_formatted_procedure_context, ...]
TOOL_REGISTRY = {t.name: t for t in ALL_TOOLS}
llm = ChatOpenAI(model=BRAIN_MODEL, temperature=0, seed=42).bind_tools(ALL_TOOLS)
tool_node = ToolNode(ALL_TOOLS)
```

---

### STEP 5 — The System Prompt

**What to build:** A prompt template function `build_system_prompt(state)` that renders the full hypothesis ledger and investigation context inline.

This is the most important piece. The agent does exactly what this prompt says.

**Structure:**
```
[Investigation context — extracted from messages]
Machine: ... | Symptom: ... | Period: ...

[Current hypothesis ledger — rendered from state.hypotheses]
H1 [ACTIVE] ... (conf=0.45) | sources: INT-2024-0031
H2 [PRIOR]  ... (conf=0.20) | sources: CB-200_Troubleshooting.pdf

[Tool use rules]
- Call tools to gather evidence. Do NOT call tools if the user asks for a summary.
- Always use the user's literal words in tool queries.

[Ledger update instruction]
After processing tool results, emit a JSON block:
<ledger_update>
{
  "new_hypotheses": [...],
  "evidence_updates": [{"hypothesis_id": "H1", "delta": 0.15, ...}],
  "need_user_input": false,
  "question": ""
}
</ledger_update>

Set need_user_input: true ONLY when ALL of these are true:
  1. Two or more hypotheses are ACTIVE with confidence within 0.15 of each other.
  2. No tool exists that can discriminate between them — the answer requires physical
     observation or operator knowledge.
  3. A single yes/no answer from the operator could move one hypothesis above 0.80
     or below 0.20.
When need_user_input is true, set "question" to the single most discriminating
yes/no question the operator can answer right now. Do NOT ask for information
that a tool can retrieve.

[Phase instructions — inferred from what's been done, not a phase number]
If no hypotheses yet: call procedure + recent CM tools first.
If hypotheses exist: ask one discriminating question or search deeper history.
If user asks for action plan: synthesize from ledger, no tools.
```

**Why no phase number in state?**
The agent infers what to do next from the conversation and the ledger state. If the ledger is empty, it knows to gather. If it's populated, it knows to narrow. Explicit phase tracking is a crutch — a well-written prompt doesn't need it.

**The `<ledger_update>` block:**
The agent emits this XML-tagged JSON in its reply whenever it has tool results to process. Your code parses it out of the `AIMessage.content`, applies the updates to `state.hypotheses`, sets `awaiting_user`, and strips the block from the displayed response. This is the "structured output in the prompt template" — no separate node, no separate LLM call.

---

### STEP 6 — The Agent Node

**What to build:** A single `agent_node(state)` function.

```python
def agent_node(state: RCAState) -> dict:
    system_prompt = build_system_prompt(state)
    response = llm.invoke([SystemMessage(content=system_prompt)] + state.messages)

    # Parse ledger update out of the response if present
    hypotheses, need_input = parse_and_apply_ledger_update(response.content, state.hypotheses)

    # Strip the <ledger_update> block from the displayed content
    clean_content = strip_ledger_block(response.content)
    response = AIMessage(content=clean_content, tool_calls=response.tool_calls)

    return {"messages": [response], "hypotheses": hypotheses, "awaiting_user": need_input}
```

**`parse_and_apply_ledger_update`:**
- Find `<ledger_update>...</ledger_update>` in the content with a regex
- Parse the JSON
- Create new `Hypothesis` objects from `new_hypotheses`
- Apply `delta` from `evidence_updates` to existing hypotheses
- Clamp confidence, update status
- Read `need_user_input` from the parsed block (default `False` if absent)
- Return `(merged_hypothesis_list, need_user_input)`

**Why parse in the agent node, not a separate node?**
The ledger update and the narrative reply are generated in the same LLM call. Splitting them into separate nodes would require either a second LLM call or passing raw content between nodes. Parsing inline keeps it one call, one node.

---

### STEP 7 — Wire the Graph

**What to build:** Connect agent → tool_node → agent, with a three-way routing function that also handles mid-flight user input.

```python
from langgraph.types import interrupt

def route_after_agent(state: RCAState) -> str:
    last_msg = state.messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tool_node"
    if state.awaiting_user:
        return "human_feedback"
    return END

def human_feedback_node(state: RCAState) -> dict:
    # interrupt() pauses graph execution and surfaces the last AIMessage to the caller.
    # The graph resumes when .invoke() / .stream() is called again with the user's answer.
    user_answer = interrupt("Waiting for operator input")
    return {
        "messages": [HumanMessage(content=user_answer)],
        "awaiting_user": False,
    }

workflow = StateGraph(RCAState)
workflow.add_node("agent", agent_node)
workflow.add_node("tool_node", tool_node)
workflow.add_node("human_feedback", human_feedback_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", route_after_agent,
    ["tool_node", "human_feedback", END])
workflow.add_edge("tool_node", "agent")
workflow.add_edge("human_feedback", "agent")   # user answer goes back into the agent
```

Three nodes. Four edges. The router is the only change from the two-node version.

**`ToolNode`** handles tool execution, error catching, and formatting results as `ToolMessage` objects — no custom tool executor needed.

**`interrupt()`** is LangGraph's built-in mechanism for human-in-the-loop. It raises a special exception that checkpoints the graph state at that exact point. The caller receives an `Interrupt` event with the message you pass. When the graph is resumed (via `.invoke()` again with the same `thread_id`), execution continues from right after the `interrupt()` call — `user_answer` gets the value you pass back in.

**Why `interrupt()` instead of just routing to END?**
Routing to END would lose the graph state. `interrupt()` preserves the full ledger, message history, and hypothesis confidences — so when the user answers, the agent resumes with the complete picture and can immediately apply the new evidence to the ledger.

**Interactive driver update for `interrupt()`:**
```python
from langgraph.types import Command

while True:
    user_input = input("You: ").strip()
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]}, config
    )

    # Check if the graph paused waiting for operator input
    if "__interrupt__" in result:
        interrupt_value = result["__interrupt__"][0].value
        print(f"\nAgent needs input: {interrupt_value}")
        operator_answer = input("Operator: ").strip()
        # Resume the graph by passing the answer back
        result = app.invoke(Command(resume=operator_answer), config)

    # Display last AIMessage and hypothesis ledger
    ...
```

---

### STEP 8 — Interactive Driver

**What to build:** A REPL loop that compiles the graph with a checkpointer and invokes it per turn.

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)
config = {"configurable": {"thread_id": "rca-001"}}

while True:
    user_input = input("You: ").strip()
    result = app.invoke({"messages": [HumanMessage(content=user_input)]}, config)
    # find last AIMessage, print it and the ledger
```

**What to display after each turn:**
- Last `AIMessage.content` (already stripped of the `<ledger_update>` block)
- The hypothesis ledger from `result["hypotheses"]`, sorted by confidence descending

---

## Checklist Before Running

- [ ] `tools/` directory is accessible from `notebooks/`
- [ ] Qdrant is running (`make run-docker-compose`)
- [ ] OpenAI API key is set
- [ ] Collections `cm_interventions_hybrid` and `procedures_hybrid` exist in Qdrant

---

## Test Scenarios

### Scenario A — Golden path
```
You: CB-200 conveyor has belt misalignment since 2025-01-10
```
**Expected:** Agent calls procedure + recent CM tools in one batch. `<ledger_update>` block creates 4–8 hypotheses. Agent narrates findings, asks one yes/no question.

### Scenario B — Missing info
```
You: my conveyor is broken
```
**Expected:** Agent asks for machine ID, symptom, and time period. Does NOT call tools. No `<ledger_update>` block emitted.

### Scenario C — User rules out a hypothesis
```
You: Yes, already checked, idlers are fine
```
**Expected:** Agent emits `<ledger_update>` with a negative delta for the idler hypothesis → confidence drops → agent pivots to next suspect.

### Scenario D — Action plan
```
You: let's wrap up
```
**Expected:** Agent reads hypotheses from the ledger in its system prompt, synthesizes a ranked action plan. No tool calls. No `<ledger_update>` block.

### Scenario E — Mid-flight discrimination
After the agent gathers initial evidence, two hypotheses are close:
```
H1 [ACTIVE] Belt stretch (conf=0.45)
H2 [ACTIVE] Idler wear   (conf=0.42)
```
**Expected:** Agent emits `need_user_input: true` with question:
*"Have you noticed rubber debris accumulating near the tail pulley?"*

Graph pauses at `human_feedback`. Operator answers:
```
Operator: Yes, quite a bit of rubber shavings there.
```
Graph resumes → agent emits `<ledger_update>` with:
- `H1` delta `+0.35` → confidence `0.80` → status CONFIRMED
- `H2` delta `-0.25` → confidence `0.17` → status REJECTED

Agent pivots to action plan for belt stretch. No further tool calls needed.

---

## Common Mistakes to Avoid

| Mistake | Why it's wrong | Fix |
|---|---|---|
| Storing phase/machine/symptom in state | Two sources of truth — state and conversation diverge | Extract them from `messages` at prompt-render time |
| Not rebuilding system prompt on every agent call | Agent reasons from stale ledger confidences | Call `build_system_prompt(state)` inside `agent_node`, not once at startup |
| Binding different tools per phase | Requires multiple LLM instances and routing logic | One binding + prompt instructions that say which tools to use per situation |
| Hypothesis statement = symptom | "Belt is misaligned" is useless | Prompt must require a specific ROOT CAUSE, with wrong/right examples |
| Forgetting to strip `<ledger_update>` from displayed content | User sees raw JSON in the chat | `strip_ledger_block()` before wrapping in `AIMessage` |
| Not clamping confidence to [0, 1] after deltas | Confidence floats above 1 or below 0 — status logic breaks | `max(0.0, min(1.0, h.confidence + delta))` |
| Parsing ledger JSON in a separate node | Extra node, extra complexity, same result | Parse inline in `agent_node` right after the LLM call |
| Routing to END instead of `interrupt()` for user input | Graph state is lost — ledger and history gone | Use `interrupt()` so the checkpointer saves state; resume with `Command(resume=answer)` |
| Asking the user when a tool could answer | Slows the loop, breaks trust | Only set `need_user_input: true` for physical observations tools cannot retrieve |
| Not resetting `awaiting_user` in `human_feedback_node` | Graph loops forever between agent and human_feedback | Always return `{"awaiting_user": False}` from `human_feedback_node` |

---

## What Changed vs the Previous Architecture

| Previous | This notebook |
|---|---|
| 5 nodes: agent, parallel_tool_caller, evidence_ledger, phase0_readiness, phase_router | 3 nodes: agent, tool_node, human_feedback |
| Separate `evidence_ledger_node` — extra LLM call per turn | Ledger update parsed from agent's reply — same LLM call |
| `phase` stored in state, separate readiness check node | Phase inferred from conversation + ledger in system prompt |
| Phase-specific LLM bindings (`_llm_phase0`, `_llm_phase1`, …) | One LLM binding, prompt controls tool usage |
| `Send()` for parallel tool execution | `ToolNode` handles it |
| `parallel_tool_caller` custom node | Replaced by `ToolNode` |
| Phase router node with separate LLM call | Routing is a one-liner on `last_msg.tool_calls` + `awaiting_user` |
| No mid-flight user input — agent runs to completion | `interrupt()` pauses graph; operator discriminates hypotheses mid-flight |

---

## What You've Learned By Building This

1. **Minimal LangGraph**: three nodes is enough — agent + ToolNode + human_feedback covers ReAct with mid-flight user input
2. **Ledger-in-prompt**: rendering state inline eliminates the need for a separate extraction node
3. **Structured output in the prompt**: `<ledger_update>` lets the agent update hypotheses without a second LLM call
4. **`Annotated[List, add]` reducer**: why message history accumulates correctly across turns
5. **`merge_hypotheses` as a LangGraph reducer**: custom merge logic wired directly into state updates
6. **`ToolNode`**: LangGraph's prebuilt tool executor — no custom dispatching needed
7. **Prompt as the control plane**: phase logic, tool restrictions, and ledger updates all live in one place
8. **`interrupt()` for human-in-the-loop**: how to pause a graph mid-execution, preserve full state via checkpointer, and resume with `Command(resume=answer)` — the right pattern whenever the agent needs information that only a human can provide
