# Exercise 01 — LangGraph Basics: State, Nodes & Edges

## Concepts covered
- Defining a `State` with Pydantic
- Building single-node and multi-node graphs
- Linear edges (`add_edge`)
- Conditional routing (`add_conditional_edges`)

---

## Part 1 — Single Node Graph

Build a graph with a single node that takes a user message and a `tone` field, and returns the message reformatted with the given tone appended.

**State fields:**
- `message: str`
- `tone: str`
- `result: str = ""`

**Node — `apply_tone`:**
Returns `{"result": f"{state.message} [{state.tone}]"}`

**Graph structure:**
`START → apply_tone → END`

**Test it with:**
```python
{"message": "Conveyor belt is vibrating", "tone": "URGENT"}
```

**Expected output:**
```
"Conveyor belt is vibrating [URGENT]"
```

---

## Part 2 — Conditional Graph

Extend Part 1 to route the message to different nodes depending on the tone.

**Additional nodes:**
- `handle_urgent` — appends `"🚨 Escalate immediately"`
- `handle_normal` — appends `"📋 Log for review"`
- `handle_low` — appends `"📌 Schedule for next PM"`

**Router function — `tone_router`:**
Reads `state.tone` and returns one of: `"handle_urgent"`, `"handle_normal"`, `"handle_low"`.

**Graph structure:**
```
START → apply_tone → [tone_router] → handle_urgent / handle_normal / handle_low → END
```

**Test with:**
```python
{"message": "Motor overheating", "tone": "urgent"}
{"message": "Oil level check", "tone": "low"}
```

---

## Checklist
- [ ] State uses Pydantic `BaseModel`
- [ ] Node functions return a `dict` (not mutate state directly)
- [ ] Router returns a string matching an edge key
- [ ] Graph compiles and runs without errors
- [ ] `display_graph(graph)` shows the expected shape
