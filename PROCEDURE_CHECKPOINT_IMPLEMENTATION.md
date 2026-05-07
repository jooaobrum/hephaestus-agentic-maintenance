# Procedure Checkpoint Implementation

## Overview

Added a **Procedure Checkpoint** to the RCA agent that proactively asks technicians if they've already tried a relevant procedure before diving into diagnostic tool calls. This implements the design principle: *"never jumping to conclusions — keep the technician informed and at ease."*

## What Changed

### 1. System Prompt (`rca_react_system_prompt.yml`)

Added **PROCEDURE CHECKPOINT** section with explicit rules:

```yaml
## PROCEDURE CHECKPOINT

If you retrieve a procedure with confidence ≥0.75 AND you have not yet asked the technician about it:
1. Pause the investigation and ask explicitly: "I found [Procedure Name/File]. Have you already followed this procedure?"
2. Set the awaiting_user state to true — do NOT call tools until they respond
3. If they say YES: ask "What was the result?" or "Did any step fail?"
4. If they say NO: offer: "I can walk you through the key diagnostic steps..."
5. Incorporate their procedure experience into the ledger as new evidence
6. After they respond, continue investigation with remaining hypotheses or move to Stage 4
```

### 2. Agent State (`AgentState` class)

Added field to track checkpoint state:

```python
procedure_checkpoint_asked: bool = False  # Track if we've asked about high-conf procedures
```

### 3. Helper Function: `find_procedure_checkpoint_candidate()`

New function that scans hypotheses to find if any have:
- Confidence ≥ 0.75 (LIKELY or higher)
- At least one PROC_REF (procedure) source

```python
def find_procedure_checkpoint_candidate(hypotheses) -> tuple[bool, str, Hypothesis | None]:
    """Returns: (should_checkpoint, procedure_id, hypothesis)"""
    for h in sorted(hypotheses, key=lambda x: x.confidence, reverse=True):
        if h.confidence >= LIKELY and h.source:
            proc_sources = [s for s in h.source if s.source_type == 'PROC_REF']
            if proc_sources:
                return (True, proc_sources[0].source_id, h)
    return (False, "", None)
```

### 4. Agent Node Logic Update

The `agent_node()` function now:

1. **After extracting & merging hypotheses**, checks for procedure checkpoint candidate
2. **If found AND not yet asked**, interrupts investigation with:
   ```
   I found a procedure that covers this issue: **PROC-042**.
   It matches your situation with 82% confidence.
   
   **Have you already followed this procedure?** If so, what was the outcome?
   If not, I can help guide you through the key diagnostic steps.
   ```
3. **Sets `awaiting_user=True`** to pause the graph and wait for response
4. **Sets `procedure_checkpoint_asked=True`** to avoid asking twice about the same procedure
5. **After technician responds**, the graph returns to the agent, which incorporates their feedback and continues

## Flow Example

### Before (Without Checkpoint)
```
User: "HX-200 high oil temperature, started this morning"
Agent: [validates machine] → [calls get_formatted_procedure_context()] 
       → [retrieves procedure context] 
       → [immediately calls sensor_catalog, threshold_events, etc.]
User: "Wait, I already checked that procedure!"
```

### After (With Checkpoint)
```
User: "HX-200 high oil temperature, started this morning"
Agent: [validates machine] → [calls get_formatted_procedure_context()] 
       → [retrieves procedure context, creates H1 with 0.82 confidence, PROC-042 source]
       → [detects high-conf procedure]
       → "I found Procedure PROC-042. Have you already followed it?"
       [AWAITING USER FEEDBACK - pauses graph]

User: "No, not yet"
Agent: "Here are the key diagnostic steps from the procedure..."
       [continues investigation]
```

## Benefits

1. **Technician-centric**: Respects existing knowledge; doesn't waste time on diagnostics they've already done
2. **Efficient**: Uses the most direct answer (documented procedures) before diving into data
3. **Natural conversation**: Feels like talking to a knowledgeable colleague, not an automated tool
4. **Evidence tracking**: Technician's "I already tried X" becomes part of the ledger (−0.25 delta when rejecting hypotheses)

## State Management

The checkpoint respects the graph's multi-turn design:

| State | Agent | Graph | Result |
|-------|-------|-------|--------|
| `procedure_checkpoint_asked=False` + high-conf procedure found | Asks checkpoint question, sets flag=True, awaiting_user=True | Routes to END (wait for next input) | Pauses for technician response |
| `procedure_checkpoint_asked=True` | Uses response as evidence, continues normally | Routes based on next LLM response | Investigation continues |

## Testing

The implementation is backward-compatible:

- If no procedures are retrieved, checkpoint never triggers
- If procedures are < 0.75 confidence, checkpoint never triggers  
- If procedures were already asked about, checkpoint only happens once
- Normal text-only responses (no tool calls) continue to work as before

### Test Case
```python
# Case 1: Procedure found with high confidence
state = AgentState(messages=[
    HumanMessage(content="HX-200, high oil temp, this morning")
])
result = graph.invoke(state, config={"configurable": {"thread_id": "test"}})
# Expected: Agent asks "Have you followed PROC-042?"
# state.awaiting_user == True
# state.procedure_checkpoint_asked == True

# Case 2: Technician responds
state.messages.append(HumanMessage(content="No, not yet"))
result = graph.invoke(state, config={"configurable": {"thread_id": "test"}})
# Expected: Agent continues investigation, incorporates "not tried" as evidence
```

## Next Steps (Optional)

1. **Track procedure outcomes**: Add `tried_procedures: list[dict]` to state to log which steps passed/failed
2. **Integrate procedure guidance**: Surface relevant decision trees from the procedure in follow-up messages
3. **Fleet patterns**: Cross-reference technician's procedure experience with fleet-wide outcomes ("3 other machines had the same issue...")

