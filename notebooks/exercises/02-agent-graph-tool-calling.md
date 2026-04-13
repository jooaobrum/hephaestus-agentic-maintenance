# Exercise 02 — Agent Graph with Tool Calling

## Concepts covered
- Defining tools with docstrings for auto-description
- Using `instructor` to extract structured tool calls from LLM responses
- Building an agent node that calls tools
- Using `ToolNode` from LangGraph prebuilt
- `Annotated[List[Any], add]` for message accumulation
- Converting messages between OpenAI and LangGraph formats

---

## Setup

Define this tool:

```python
def classify_fault(fault_code: str, machine: str) -> str:
    """Classifies a fault code for a given machine and returns its severity.

    Args:
        fault_code: The fault code reported (e.g. 'E-007', 'H-003').
        machine: The machine identifier (e.g. 'HX-350', 'CB-200').

    Returns:
        A string describing the fault severity: 'critical', 'warning', or 'info'.
    """
    critical = ["E-007", "H-003", "B-006"]
    if fault_code in critical:
        return f"{fault_code} on {machine}: CRITICAL — immediate intervention required"
    return f"{fault_code} on {machine}: WARNING — monitor and log"
```

---

## Part 1 — Sequential Agent (call tool once, then END)

**State fields:**
- `messages: Annotated[List[Any], add] = []`
- `user_query: str = ""`
- `available_tools: List[Dict[str, Any]] = []`
- `tool_calls: List[ToolCall] = []`

**Pydantic models:**
```python
class ToolCall(BaseModel):
    name: str
    arguments: dict

class AgentResponse(BaseModel):
    answer: str
    tool_calls: List[ToolCall]
```

**`agent_node`:**
- Build a system prompt instructing the agent to classify faults using `classify_fault`
- Use `instructor.from_provider("openai/gpt-4.1-mini")` with `response_model=AgentResponse`
- Return `{"messages": [format_ai_message(response)], "tool_calls": response.tool_calls}`

**`tool_router`:**
- If `state.tool_calls` is non-empty → `"tools"`
- Otherwise → `"end"`

**Graph structure:**
```
START → agent_node → [tool_router] → tool_node → END
                                   ↘ END
```

**Test with:**
```python
{"user_query": "I'm seeing fault E-007 on machine HX-350", "available_tools": tool_descriptions}
```

---

## Part 2 — ReAct Agent (loop back after tool execution)

Modify the graph so `tool_node` loops back to `agent_node` instead of going to END.

Add an iteration guard: if `state.iteration > 2`, route to END regardless of tool calls.

**Additional state fields:**
- `iteration: int = 0`
- `answer: str = ""`

**Updated `agent_node`:**
- Include prior `state.messages` in the conversation using `convert_to_openai_messages`
- Increment `iteration` in the returned dict

**Updated `tool_router`:**
```python
if state.iteration > 2:
    return "end"
elif len(state.tool_calls) > 0:
    return "tools"
else:
    return "end"
```

**Graph structure:**
```
START → agent_node → [tool_router] → tool_node → agent_node (loop)
                                   ↘ END
```

**Test:** Run the same query and verify `result["iteration"]` reflects the loop count.

---

## Checklist
- [ ] `get_tool_descriptions([classify_fault])` returns a valid schema
- [ ] `format_ai_message(response)` produces a LangGraph `AIMessage`
- [ ] Sequential graph stops after one tool call
- [ ] ReAct graph loops and stops at iteration limit
- [ ] Final `result["messages"]` contains both `AIMessage` and `ToolMessage` entries
