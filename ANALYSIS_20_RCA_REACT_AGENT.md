# Analysis: 20-rca-react-agent.ipynb

## Current Agent Capability

The agent **CAN** identify relevant procedures and ask if the technician has tried them. Here's how:

### Current State

1. **Procedure Retrieval Tool Available**
   - `get_formatted_procedure_context(query)` searches the `procedures_hybrid` collection
   - Uses semantic (dense embeddings) + keyword (BM25) hybrid search
   - Returns procedures as formatted text with file name, section title, and context

2. **System Prompt Guidance** (rca_react_system_prompt.yml)
   - Stage 2 explicitly calls for: `get_formatted_procedure_context` during scoping
   - The agent is instructed to "seed hypotheses" from procedures (gain +0.20 confidence boost)
   - Instructions say: **"Use AFTER initial hypothesis generation to ground in procedures"**

3. **Investigation Flow**
   - **Stage 1 (Intake)**: Validate machine ID, get symptom, get time reference
   - **Stage 2 (Scoping)**: Batch-call multiple tools including `get_formatted_procedure_context`
   - **Stage 3 (Deep History)**: Only if confidence < 0.80
   - **Stage 4 (Action Plan)**: Synthesize findings, propose ranked actions

### What's Working

The agent **does** retrieve procedures and uses them as evidence. The system prompt tells it to:
- Display procedures in the ledger with confidence boosts
- Mention which procedure(s) matched (e.g., "Found procedure P-0042: High Oil Temperature Diagnostics")
- Include procedure sources in hypothesis references

### What's Missing

The agent doesn't currently **proactively ask** "Did you try this procedure already?" in a natural, conversational way. Here's why:

1. **Procedure Matching is Silent**
   - Procedures are added as evidence (sources in the ledger) but the response doesn't explicitly highlight them as *actionable next steps*
   - The ledger displays them but doesn't frame them as questions or recommendations to the technician

2. **Action Plan Logic Incomplete**
   - Stage 4 is supposed to synthesize "what to do (ranked actions)" but there's no explicit mechanism to:
     - Prioritize procedures retrieved in Stage 2
     - Surface them as questions ("Have you run this?")
     - Track whether the technician has already tried them

3. **No "Procedure Checkpoint" in the Flow**
   - The agent could pause mid-investigation and ask:
     ```
     "I found a procedure that covers this issue: [Procedure Name].
      Have you already walked through it? If so, what was the result?"
     ```
   - But there's no explicit instruction or state field for this pause point

---

## How to Implement the Missing Piece

### Option A: Add Procedure Checkpoint (Minimal)
**Goal**: After finding a high-confidence procedure match, ask if they've tried it.

**Changes needed**:

1. **Add to `AgentState`** (20-rca-react-agent.ipynb):
   ```python
   class AgentState(BaseModel):
       ...
       procedure_candidate: str = ""  # procedure name/ID if awaiting feedback
   ```

2. **Add to system prompt** (rca_react_system_prompt.yml):
   ```
   ## PROCEDURE CHECKPOINT
   
   If you retrieve a procedure with confidence ≥0.75 AND no sensor data has contradicted it yet:
   - Ask: "I found [Procedure Name]. Have you already tried it?"
   - Set state.awaiting_user = True
   - Pause investigation until they respond
   - If they say "yes" → ask what happened
   - If they say "no" → offer to guide them through it
   ```

3. **Modify `agent_node()`** to recognize when a procedure should be highlighted:
   - After extracting hypotheses, check if any have source_type='PROC_REF' and confidence ≥0.75
   - If so, **ask about it before proceeding** to tool calls

**Example flow**:
```
User: "HX-200 hydraulic press, high oil temperature, started this morning"
Agent: "Got it. Let me check what we know about this.
        
I found a procedure that covers high oil temp on hydraulic systems.
Have you already followed the diagnostics in the High Oil Temperature 
Troubleshooting Guide? 

[If yes: What did you find?]
[If no: I can walk you through the key steps.]"
```

### Option B: Procedure-First Investigation (Moderate)
**Goal**: Use procedures as the primary evidence source before diving into sensors.

**Changes needed**:
- Reorder Stage 2 tools: `get_formatted_procedure_context()` first, then sensors/CM
- If procedure finds a match, ask technician if they've tried the recommended actions
- Use procedure's decision tree to inform which sensors to check next

### Option C: Procedure History Tracking (Advanced)
**Goal**: Track whether specific procedures have been tried on this machine, in this case.

**Changes needed**:
- Add `tried_procedures: list[str]` to `AgentState`
- When technician says "I tried [Procedure X]", add it to the list
- Track their findings (which steps passed, which failed)
- Use that to rule in/out hypotheses based on procedure results

---

## Current Behavior Example

Based on the code, here's what actually happens now:

**Input**: "HX-200 hydraulic press, high oil temperature, this morning"

**What the agent does**:
1. ✓ Validates machine exists
2. ✓ Calls `get_formatted_procedure_context("high oil temperature")`
   - Gets back procedure snippet with diagnostics
3. ✓ Adds procedure as a source to a hypothesis: "H1: Thermal management failure — conf=0.65 (PROC-0042)"
4. ✗ **Does NOT ask**: "Have you tried running the procedure?"
5. → Proceeds to sensor checks, CM history, etc.

**What should happen** (with minimal change):
1. ✓ Validates machine, gets symptom, time period
2. ✓ Calls procedure search
3. ✓ Finds high-confidence match
4. **→ PAUSE and ask**: "I found a procedure for this. Have you tried it?"
5. → Wait for technician response
6. → Adapt investigation based on their answer

---

## Recommendation

**Start with Option A** (Procedure Checkpoint). It's the smallest change with the highest payoff:

- Adds ~20 lines to system prompt (rules for when to ask about procedures)
- Adds 1-2 fields to `AgentState`
- Modifies `agent_node()` to check procedure confidence and ask before proceeding
- Gives technician a natural conversational checkpoint: **"Did you try this?"**

This respects the principle in your system prompt: *"never jumping to conclusions — keep the technician informed."* A procedure is the most direct answer; if it exists and matches, ask if they've tried it before running diagnostics.

### Test Case
```python
# After implementing Procedure Checkpoint:
state = AgentState(messages=[
    HumanMessage(content="HX-200 hydraulic press, high oil temperature, since morning")
])
result = graph.invoke(state, config={"configurable": {"thread_id": "hx200-test"}})

# Expected:
# "I found Procedure P-042: High Oil Temperature Diagnostics for hydraulic systems.
#  Have you already run this procedure? If so, what was the outcome?"
#
# state.awaiting_user should be True
```

---

## Summary Table

| Capability | Current | Gap | Priority |
|---|---|---|---|
| Retrieve procedures | ✓ Yes | — | — |
| Add procedures to evidence ledger | ✓ Yes | — | — |
| Ask if procedure was tried | ✗ No | Procedure not highlighted for checkpoint | High |
| Track procedure results | ✗ No | No state field for tried procedures | Medium |
| Adapt investigation based on procedure outcome | ✗ No | No feedback loop | Medium |

