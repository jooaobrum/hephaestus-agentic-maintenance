# Exercise 04 — Query Router: Intent Classification & Full Agentic RAG Pipeline

## Concepts covered
- Intent classification as a routing node
- Combining intent routing + query expansion + parallel retrieval in one graph
- Graceful handling of out-of-scope questions
- Full end-to-end agentic RAG pipeline with LangSmith tracing

---

## Part 1 — Intent Router Node

Build an `intent_router_node` that classifies whether an incoming query is relevant to equipment maintenance.

**Pydantic model:**
```python
class IntentRouterResponse(BaseModel):
    question_relevant: bool
    answer: str  # explanation when not relevant, empty string when relevant
```

**`intent_router_node`:**
- Prompt the LLM with examples of relevant vs. irrelevant questions
- Relevant: fault codes, symptoms, repair history, root causes, scheduled maintenance, component replacements
- Not relevant: HR policies, personal advice, unrelated topics
- Return `{"question_relevant": response.question_relevant, "answer": response.answer}`

**Routing function:**
```python
def intent_router_conditional_edges(state: State) -> str:
    if state.question_relevant:
        return "query_expansion_node"
    else:
        return "end"
```

**Test the router standalone with:**
```python
# Should return relevant=True
"What caused the hydraulic pump failure on HX-350?"

# Should return relevant=False
"What's the weather like today?"

# Should return relevant=False
"Can you help me write a performance review?"

# Should return relevant=True
"Show me all bearing failures on CB-200 last year"
```

---

## Part 2 — Full Pipeline Graph

Combine all pieces into the complete agentic RAG pipeline:

```
START → intent_router_node → [relevant?] → query_expansion_node → [parallel_retriever] → retriever_node (×N) → aggregator_node → END
                                         ↘ END (with explanation message)
```

**State (complete):**
```python
class State(BaseModel):
    # Routing
    question_relevant: bool = False
    # Queries
    initial_query: str = ""
    query: str = ""
    expanded_query: List[ExpandedQuery] = []
    # Retrieval
    retrieved_contexts: Annotated[List[str], add] = []
    top_k: int = 5
    # Output
    answer: str = ""
```

**Nodes to wire:**
- `intent_router_node`
- `query_expansion_node`
- `retriever_node`
- `aggregator_node`

**Test scenarios:**

```python
# Should go through full pipeline
{"initial_query": "The hydraulic pump on the HX-350 is losing efficiency."}

# Should be rejected at router
{"initial_query": "What's the last contract we have in the db?"}

# Should go through full pipeline
{"initial_query": "Any fan or roller issues on CB-200 this year?"}

# Should be rejected at router
{"initial_query": "Write me a Python function to sort a list."}
```

---

## Part 3 — Extend the Router (Challenge)

Add a second routing branch for **procedural questions** (e.g. "How do I replace a bearing?") that routes to a different `procedure_node` instead of the RAG pipeline.

The `procedure_node` can simply return a placeholder:
```python
def procedure_node(state: State) -> dict:
    return {"answer": f"Procedure lookup not yet implemented for: {state.initial_query}"}
```

**Updated routing logic:**
```python
def intent_router_conditional_edges(state: State) -> str:
    # Return one of: "query_expansion_node", "procedure_node", "end"
    ...
```

Update `IntentRouterResponse` to include a `route: Literal["rag", "procedure", "off_topic"]` field.

---

## Checklist
- [ ] `intent_router_node` correctly classifies all 4 test queries
- [ ] Off-topic queries return a human-readable `answer` without hitting retrieval
- [ ] Relevant queries go through expansion → parallel retrieval → aggregation
- [ ] Full pipeline graph compiles and `display_graph` shows all branches
- [ ] `@traceable` decorators on `intent_router_node`, `query_expansion_node`, `retriever_node`, `aggregator_node`
- [ ] (Challenge) Three-way router routes to procedure node for how-to questions
