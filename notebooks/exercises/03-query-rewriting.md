# Exercise 03 — Query Rewriting: Sequential and Parallel RAG Pipelines

## Concepts covered
- Query expansion with LLM + instructor
- Sequential retrieval over multiple queries
- Parallel retrieval using `Send` from `langgraph.types`
- `Annotated[List[str], add]` reducer for fan-in aggregation
- Dynamic `top_k` per expanded query
- LangSmith `@traceable` decoration

---

## Setup

Assume you have working `retrieve_rerank(qdrant_client, openai_client, cohere_client, query, top_k)` and `format_context(results)` functions from notebook 11.

---

## Part 1 — Sequential Query Expansion

Build a graph that:
1. Expands the user query into 1–3 statements
2. Retrieves context for **each statement one by one** (a loop inside the retriever node)
3. Aggregates all contexts and generates a final answer

**State:**
```python
class State(BaseModel):
    initial_query: str = ""
    expanded_query: List[str] = []
    retrieved_contexts: Annotated[List[str], add] = []
    answer: str = ""
```

**`query_expansion_node`:**
- Prompt the LLM to rewrite `state.initial_query` into 1–3 maintenance-specific statements
- Return `{"expanded_query": response.statements}`

**`retriever_node`:**
- Loop over `state.expanded_query`
- Call `retrieve_rerank(...)` for each query
- Return `{"retrieved_contexts": all_contexts}`

**`aggregator_node`:**
- Join `state.retrieved_contexts` with `"\n"`
- Prompt the LLM to answer `state.initial_query` using the combined context
- Return `{"answer": response.answer}`

**Graph:**
```
START → query_expansion_node → retriever_node → aggregator_node → END
```

**Test with:**
```python
{"initial_query": "Fan motor vibration and bearing noise on CB-200"}
```

---

## Part 2 — Parallel Query Expansion (using `Send`)

Refactor so each expanded query fires its own `retriever_node` in **parallel**.

**Updated `ExpandedQuery` model:**
```python
class ExpandedQuery(BaseModel):
    query: str
    top_k: int  # LLM assigns based on query specificity
```

**Updated State — add fields for per-node routing:**
```python
query: str = ""      # used by individual retriever_node instances
top_k: int = 5
```

**`parallel_retriever` function:**
```python
def parallel_retriever(state: State) -> list[Send]:
    return [
        Send("retriever_node", State(query=eq.query, initial_query=state.initial_query, top_k=eq.top_k))
        for eq in state.expanded_query
    ]
```

**Updated `retriever_node`** — operates on a single `state.query` (no loop).

**Graph:**
```
START → query_expansion_node → [parallel_retriever] → retriever_node (×N) → aggregator_node → END
```

Wire the fan-out:
```python
workflow.add_conditional_edges("query_expansion_node", parallel_retriever, ["retriever_node"])
```

**Test with the same query and compare:** Does parallel retrieval return richer context?

---

## Part 3 — Reflection

Answer these questions in comments or a markdown cell:

1. Why does `retrieved_contexts` use `Annotated[List[str], add]` instead of a plain `List[str]`?
2. What happens if two parallel retriever nodes return overlapping context chunks?
3. When would sequential retrieval be preferable over parallel?

---

## Checklist
- [ ] `query_expansion_node` uses `@traceable` decorator
- [ ] Sequential graph loops retrieval inside a single node
- [ ] Parallel graph uses `Send` to dispatch one node per query
- [ ] `retrieved_contexts` accumulates correctly across parallel branches
- [ ] `aggregator_node` uses all contexts to generate the final answer
- [ ] Graphs compile and `display_graph(graph)` renders correctly
