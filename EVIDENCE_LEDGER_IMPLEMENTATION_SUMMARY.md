# Evidence Ledger Implementation Summary

## What Was Fixed

The RCA agent's evidence ledger was not properly displaying source references, making it impossible for users to trace back from a hypothesis to the actual evidence that supported it.

### Problems
1. **Opaque Sources** — Graph output showed "GRAPH: High Oil Temperature Fault" instead of actual intervention IDs
2. **Missing Evidence Details** — No descriptions/explanations of physical mechanisms
3. **No Traceability** — Procedures returned filenames only; CM interventions weren't referenced by ID
4. **Inconsistent Format** — Different tools used different source reference styles

---

## Changes Made

### 1. **notebooks/tools/tools.py** — Tool Output Enhancement

#### `_format_proc_context()`
```python
# BEFORE
f"File: {payload.get('file_name', 'N/A')}\n"

# AFTER
file_name = payload.get('file_name', 'N/A')
page_number = payload.get('page_number', payload.get('page', 'N/A'))
chunk_number = payload.get('chunk_number', 'N/A')
source_ref = f"PROC_REF:{file_name}:{page_number}:chunk#{chunk_number}"
f"Source: {source_ref}\n"
```

**Result:** Procedures now output `PROC_REF:cooling-system.pdf:3:chunk#1`

---

#### `_format_cm_context()`
```python
# BEFORE
f"ID: {payload.get('id', 'N/A')}\n"

# AFTER
intervention_id = payload.get('id', 'N/A')
f"Source: INT-{intervention_id}\n"
```

**Result:** CM results now output `Source: INT-089` making them linkable

---

#### `query_known_issues_graph()`
```python
# BEFORE
res = f"GRAPH: {p.get('symptom_name', 'N/A')}\n"
res += f"Representative interventions: {', '.join(rep_ids)}\n"

# AFTER
rep_ids = p.get("representative_intervention_ids", [])
source_ref = f"GRAPH: {', '.join(rep_ids)}" if rep_ids else "GRAPH: (no interventions)"
res = f"Source: {source_ref}\n"
res += f"Symptom: {p.get('symptom_name', 'N/A')}\n"
```

**Result:** Graph output now shows `Source: GRAPH: INT-2025-0072, INT-2025-0398, INT-2025-0185`

---

### 2. **notebooks/prompts/rca_react_system_prompt.yml** — Specification Clarification

#### Updated Source Format Documentation
```yaml
# PROC_REF — must include page and chunk
PROC_REF:filename:page:chunk#number

# GRAPH — lists actual intervention IDs (not fault name)
GRAPH:INT-XXX, INT-YYY...
```

#### Fixed Ledger Display Examples
- Updated from `GRAPH:INT-001` to `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185`
- Shows how multiple INT sources can appear in one hypothesis
- Clarified when explanations should be included

---

### 3. **notebooks/20-rca-react-agent.ipynb** — Display Enhancement

#### `format_ledger_for_context()` Improvement
- Now properly formats delta/reason in parentheses: `(+0.30 KG match)`
- Shows explanation only for ACTIVE/LIKELY/CONFIRMED with evidence
- Formats sources as comma-separated IDs for clarity

**Example Output:**
```
✓ H1 [CONFIRMED] Cooler fouling — conf=0.85 (+0.30 KG match)
  Explanation: Cooler fins fouled by dust restricts cooling fluid flow, causing temperature rise.
  Sources: PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#2, GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185
```

---

## Source Format Specification

### PROC_REF — Procedure References
**Format:** `PROC_REF:filename:page:chunk#number`

**Components:**
- `filename` — Procedure document name (e.g., `cooling-system.pdf`)
- `page` — Page number in document
- `chunk#` — Section/chunk identifier
- `number` — Specific chunk reference

**Example:** `PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#1`

**Usage:** Users can find the exact procedure step referenced

---

### GRAPH — Knowledge Graph (Patterns)
**Format:** `GRAPH:INT-XXXX` (one or more IDs)

**Components:**
- `INT-XXXX` — Historical intervention IDs that match the pattern
- Can list multiple: `INT-001, INT-042, INT-089`

**Example:** `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185`

**Usage:** Users can review how similar cases were resolved in the past

---

### INT — Direct Intervention
**Format:** `INT-XXXX`

**Components:**
- `XXXX` — Unique intervention identifier

**Example:** `INT-089`, `INT-156`

**Usage:** Users can pull full intervention details (parts, duration, cost, outcome)

---

### USER — Technician Confirmation
**Format:** `USER`

**Meaning:** The technician (user) explicitly confirmed or rejected this hypothesis

**Examples:**
- `+0.25 USER` — technician confirmed ("yes, I checked that")
- `−0.25 USER` — technician ruled out ("no, we verified that's not it")

---

## How It Works Together

### Agent Tool Flow
1. **Agent calls tool** → `get_formatted_procedure_context("high oil temperature")`
2. **Tool returns** with `Source: PROC_REF:cooling-system.pdf:3:chunk#1`
3. **Agent extracts** sources from tool output
4. **Agent creates hypothesis** with sources: `[PROC_REF:cooling-system.pdf:3:chunk#1]`
5. **Ledger displays** the source for user traceability

### User Benefit
```
See hypothesis: "Cooler fouling"
↓
See source: PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#2
↓
Look up: Page 3, chunk 2 of troubleshooting procedures
↓
Find: "Check 2.1: Inspect cooler fins for fouling"
↓
Follow: Step-by-step diagnostic instructions
```

---

## Validation & Testing

The agent's source parsing logic correctly identifies:
- ✓ `PROC_REF:cooling-system.pdf:3:chunk#1` → PROC_REF type
- ✓ `GRAPH:INT-2025-0072, INT-2025-0398` → GRAPH type  
- ✓ `INT-089` → INT type
- ✓ `USER` → USER type

Source deduplication ensures:
- Each source appears only once across all hypotheses
- When multiple hypotheses share a source, it's assigned to the highest-confidence one
- This prevents double-counting evidence

---

## Documentation

Two user-facing guides created:

### [EVIDENCE_LEDGER_USAGE.md](EVIDENCE_LEDGER_USAGE.md)
- How to read the ledger format
- What each source type means
- How to use sources to investigate further
- Common questions and scenarios

### [EVIDENCE_LEDGER_FIXES.md](EVIDENCE_LEDGER_FIXES.md)
- Technical overview of changes
- Before/after examples
- Traceability chain explanation
- Implementation details

---

## Impact

### Before
- Sources were opaque and unmappable to actual evidence
- Users couldn't verify where a hypothesis came from
- No way to look up procedures or past cases referenced in the ledger

### After
- Complete source traceability
- Users can follow evidence chain: Hypothesis → Source → Original evidence
- Procedures, graphs, and interventions are now explicitly linkable
- Confidence-building through visible evidence accumulation

---

## Files Modified

1. `notebooks/tools/tools.py` — Tool output formatting
2. `notebooks/prompts/rca_react_system_prompt.yml` — Source format specification
3. `notebooks/20-rca-react-agent.ipynb` — Ledger display logic
4. **New:** `EVIDENCE_LEDGER_FIXES.md` — Technical reference
5. **New:** `EVIDENCE_LEDGER_USAGE.md` — User guide
