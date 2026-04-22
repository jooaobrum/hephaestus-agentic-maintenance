# RCA Agent: Lessons Learned & Improvements

## Overview
The Root Cause Analysis (RCA) agent is a 5-phase conversational maintenance assistant that guides technicians through structured diagnostic workflows. This document captures the key lessons and architectural improvements that transformed it from a rigid script-driven system into a flexible, conversational, and citation-aware agent.

---

## Key Improvements & Why They Matter

### 1. **Phase Boundaries with Explicit Tool Restrictions**
**Problem:** Agent was calling tools across phases, pulling old maintenance records when it should focus on recent data.

**Solution:** 
- Added `CRITICAL TOOL RESTRICTIONS` section in system prompt listing forbidden tools per phase
- **Phase 2** (Recent History): Can use `get_formatted_cm_context()` for 7-30 days on target machine ONLY
- **Phase 5** (Open Investigation): Unrestricted `get_formatted_cm_context()` across target + similar machines, any age

**Impact:** Prevents data scope violations. Phase 2 stays focused on recent target-machine state; Phase 5 does historical cross-machine analysis.

---

### 2. **Semantic Intent Routing Instead of Regex Parsing**
**Problem:** Used fragile `[PHASE_JUMP:N]` regex patterns to detect phase changes. Rigid, error-prone.

**Solution:**
- Created `INTENT_ROUTER_PROMPT`: Dedicated LLM call to analyze user input and return structured `PhaseDecision`
- `PhaseDecision` model includes: `target_phase` (1-5), `reason`, `stay_in_phase` (boolean)
- Intent detection rules map user phrases to phases (e.g., "show procedures" → Phase 3, "past cases" → Phase 5)

**Impact:** Dynamic, semantic routing. Agent naturally responds to user intent without explicit phase tags. Conversational flow feels organic.

---

### 3. **Conversational Flow Over Confirmation Questions**
**Problem:** Agent asked "Ready to proceed?" after each phase, forcing user confirmation. Unnatural and repetitive.

**Solution:**
- Removed confirmation questions from phase instructions
- Agent now: presents findings → naturally signals readiness → transitions to next phase
- Example: "Got it. Those steps are worth trying. Let me investigate root causes with historical data." (implicit transition to Phase 5)

**Impact:** Dialog feels like natural conversation, not a checklist. User agency is preserved—they can interrupt, ask questions, or request phase jumps anytime.

---

### 4. **Structured Reference Tracking with UsedReference Model**
**Problem:** Citations scattered across responses. No deterministic way to build final References section.

**Solution:**
- Created `UsedReference` pydantic model with fields:
  - `source_type`: 'intervention', 'procedure', 'sensor', 'component_life'
  - `id`: INT-XXXX, section ID, sensor tag, or component ID
  - `machine`: Machine ID for context
  - `detail`: Summary extracted from surrounding text (100 chars)
- `extract_references()` function parses agent output for INT-XXXX patterns and creates UsedReference objects
- `rca_agent_node()` accumulates references in `state.used_references` across all phases

**Impact:** References are deterministic, structured, and systematically compiled. Phase 5 report can build authoritative References section automatically.

---

### 5. **Explicit Phase Scope Clarification**
**Problem:** Ambiguity about what "recent history" meant. Phase 2 was retrieving 2022-2024 data (not recent).

**Solution:**
- **Phase 2**: Target machine ONLY, 7-30 days window. `get_formatted_cm_context()` allowed but with "recent" context in query
- **Phase 5**: Target machine (any age) + similar machines (any age). Unrestricted retrieval
- Updated phase instructions with explicit examples and time windows

**Impact:** Clear separation of concerns. No more data scope violations. Users understand why Phase 2 data is recent and Phase 5 includes historical patterns.

---

### 6. **Mandatory Report Format for Phase 5**
**Problem:** RCA synthesis was unstructured. Missing citations, vague hypotheses.

**Solution:**
- Phase 5 instructions mandate structured output:
  ```
  **Summary**: 1-2 sentences on likely root cause
  **Evidence**: 4-6 bullets with SPECIFIC NUMBERS and case IDs
  **Top hypotheses**: Ranked HIGH/MEDIUM/LOW with "N of M cases [INT-XXXX]" format
  **Recommended actions**: Decision tree with IF/THEN/ELSE branches, cite cases
  **Data gaps**: Explicitly list what's missing
  **References**: Systematic list from used_references
  ```

**Impact:** Reports are consistent, verifiable, and automatically structured. Every claim traces to a case ID or sensor value.

---

### 7. **RCAState with Message Accumulation**
**Problem:** Single-turn agent couldn't maintain context across phases.

**Solution:**
- `RCAState` with `Annotated[List[Any], add]` for messages (automatic accumulation via LangGraph reducer)
- Tracks: `phase`, `iteration`, `answer`, `final_answer`, `used_references`
- PostgreSQL checkpointer persists state across sessions

**Impact:** Multi-turn conversations work seamlessly. Agent maintains full context and reference history.

---

## Architecture Decisions

### Phase Definitions (1-5)
| Phase | Goal | Tool Scope | User Role |
|-------|------|-----------|-----------|
| 1: Symptom Gathering | Collect machine ID + symptom | None (conversation only) | Describe the issue |
| 2: Recent History | Analyze recent target-machine state (sensors + interventions) | 7-30 days, target machine only | Confirm or request details |
| 3: Procedure Review | Present standard diagnostic steps | Retrieve procedures only | Feedback on familiarity |
| 4: Procedure Validation | Assess user's procedure attempt | None (conversation only) | Report outcome |
| 5: Open Investigation | Synthesize RCA with historical data | All ages, target + similar machines | Review findings |

### Tool Restrictions (Why They Matter)
- **Phase 1**: Forces conversation without shortcuts
- **Phase 2**: Prevents premature historical analysis; recent data only
- **Phase 3**: Focuses on diagnostic steps, not past cases
- **Phase 4**: Gathers qualitative feedback
- **Phase 5**: Enables comprehensive historical synthesis

---

## Citation & Determinism

### Extract → Track → Report
1. **Extract**: `extract_references()` parses INT-XXXX IDs from agent output
2. **Track**: `UsedReference` objects accumulated in `state.used_references`
3. **Report**: Phase 5 final report builds References section from structured list

### Example Citation Format
```
- 1 of 2 HX-200 cases showed piston seal wear [INT-2024-0462]
- Another HX-200 case involved oil contamination [INT-2022-0182]
- 3 HX-350 cases showed cooler fouling [INT-2022-0131, INT-2024-0910, INT-2024-0484]
```

All backed by structured UsedReference objects with source_type, machine, and contextual detail.

---

## Behavioral Guidelines That Improved UX

| Guideline | Why It Works |
|-----------|-------------|
| Be conversational, not script-driven | Users feel heard, not processed |
| Don't ask "Ready?" — signal naturally | Flow feels organic; natural phase transitions |
| Respect phase boundaries strictly | Prevents data scope violations and confusion |
| Extract references systematically | Final report is verifiable and deterministic |
| Track what you've done, present clearly | Users understand progress and rationale |

---

## Testing & Validation

### Interactive Test Results
The agent successfully:
1. **Phase 1**: Gathered machine ID (HX-200) and symptom (high oil temperature)
2. **Phase 2**: Retrieved recent sensor data, component health, and recent interventions (7 days)
3. **Phase 3**: Presented troubleshooting procedures with diagnostic steps
4. **Phase 4**: Accepted user feedback on procedure attempt
5. **Phase 5**: Synthesized RCA report with:
   - Summary grounded in evidence
   - Evidence with specific case numbers (1 of 2, 3 of 5, etc.)
   - Top hypotheses ranked HIGH/MEDIUM/LOW with citations
   - Recommended actions in decision tree format
   - Data gaps explicitly listed
   - References section with INT-XXXX IDs

---

## Key Takeaways

1. **Phase boundaries are guardrails, not restrictions.** They prevent the agent from confusing different types of analysis (recent vs. historical).

2. **Intent routing is better than regex.** Semantic understanding of user input enables natural conversation flow.

3. **Conversational tone > checklist tone.** Removing forced confirmations made the agent feel like a partner, not a script.

4. **Structured citations are deterministic.** UsedReference model ensures final reports are verifiable and reproducible.

5. **Explicit phase instructions matter.** Clear tool restrictions and examples prevent scope violations.

6. **Accumulating state enables multi-turn workflows.** PostgreSQL checkpointing + LangGraph reducers make long conversations tractable.

---

## Future Enhancements (Potential)

- [ ] Real-time reference extraction from tool outputs (not just agent responses)
- [ ] Confidence scores per hypothesis based on case frequency
- [ ] Automated data gap detection and tool recommendation
- [ ] Multi-machine comparison in Phase 5 (model variants, sensor correlations)
- [ ] Feedback loop: users validate/reject hypotheses → reweight future searches
- [ ] Export reports to structured formats (JSON, PDF with citations)

---

## Conclusion

The RCA agent evolved from a rigid, script-driven system into a flexible, conversational, and citation-aware assistant through:

1. **Explicit phase boundaries** with tool restrictions
2. **Semantic intent routing** for natural conversation
3. **Removal of confirmation prompts** for organic flow
4. **Structured reference tracking** for deterministic citations
5. **Clear phase scope** definitions
6. **Mandatory report formats** for consistency
7. **State accumulation** for multi-turn context

These improvements collectively create an agent that feels natural to use, produces verifiable outputs, and guides users through structured diagnosis without feeling scripted.
