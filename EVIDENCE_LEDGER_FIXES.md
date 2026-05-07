# Evidence Ledger Format Fixes

## Problems Identified

1. **Source References Were Incorrect**
   - Procedure sources showed just the filename, not structured `PROC_REF:filename:page:chunk#N`
   - Graph sources showed the fault name (e.g., "High Oil Temperature Fault") instead of intervention IDs
   - CM intervention sources weren't being included at all

2. **Evidence Descriptions Missing**
   - Tool outputs didn't include descriptions/explanations that could be shown to the user
   - Ledger display wasn't showing physical mechanism explanations

3. **Source Traceability Lost**
   - Users couldn't trace back from a hypothesis to the actual document/intervention that supported it
   - Graph output was opaque about which specific historical cases matched

## Solutions Implemented

### 1. Enhanced Tool Output Formatting (`notebooks/tools/tools.py`)

#### `_format_proc_context()` — Procedures
- **Before:** Returned file name only
- **After:** Includes structured reference `PROC_REF:filename:page:chunk#number` at the top of each result
- Extracts `page_number` and `chunk_number` from payload
- Tools: `get_formatted_procedure_context()`

#### `_format_cm_context()` — Intervention History
- **Before:** Returned internal ID only
- **After:** Includes source reference `INT-{id}` so agent can link back to intervention
- Tools: `get_recent_formatted_cm_context()`, `get_long_formatted_cm_context()`, `get_formatted_cm_context()`

#### `query_known_issues_graph()` — Knowledge Graph
- **Before:** Output showed symptom name and generic "Representative interventions: INT-XXX, INT-YYY"
- **After:** Explicitly formatted as `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185` (concrete IDs)
- Moved interventions to the source line so agent can extract them as GRAPH sources
- Added action plan to output (root causes with recommended actions)

### 2. Updated System Prompt (`notebooks/prompts/rca_react_system_prompt.yml`)

#### Clarified Source Format Specification
- **PROC_REF:** Explicitly requires `filename:page:chunk#number` (was already correct, now reinforced)
- **GRAPH:** Now uses format `GRAPH:INT-XXX, INT-YYY` (one or more intervention IDs) instead of fault name
- **INT:** Direct intervention ID reference
- **USER:** Technician confirmation/rejection

#### Updated Ledger Display Examples
- Fixed the "RIGHT ledger" example to show proper GRAPH format with actual intervention IDs
- Demonstrated how multiple INT sources appear in a single hypothesis
- Showed explanation lines only appear for ACTIVE/LIKELY/CONFIRMED with evidence

### 3. Ledger Display Enhancement

#### `format_ledger_for_context()` — Better Evidence Display
- Shows delta/reason in parentheses: `(+0.30 KG match)`, `(-0.25 procedure attempted)`
- Includes explanation only when confidence > 0.20 AND evidence exists
- Displays all sources in comma-separated format
- Properly handles mixed source types (PROC_REF, GRAPH, INT, USER)

## Result Format

### Before
```
○ H1 [ACTIVE] High oil temperature — conf=0.30
  Sources: High Oil Temperature Fault

○ H2 [ACTIVE] Cooler fouling — conf=0.30 (+0.30 KG match)
  Sources: GRAPH: High Oil Temperature Fault
```

### After
```
○ H1 [ACTIVE] High oil temperature — conf=0.30 (+0.20 procedure baseline)
  Explanation: High oil temperature impacts machine operation and indicates cooling system issues.
  Sources: PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#1

○ H2 [ACTIVE] Cooler fouling — conf=0.30 (+0.30 KG match)
  Explanation: Cooler fins fouled by dust/debris restrict cooling fluid flow, causing temperature rise.
  Sources: PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#2, GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185
```

## Traceability Chain

Now users can follow the evidence chain:

1. **Hypothesis** → "Cooler fouling"
2. **Sources:**
   - `PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#2` → Look up procedures document, page 3, chunk 2
   - `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185` → Check these 3 past cases that had the same pattern
3. **Actions:** From procedure output, get specific diagnostic steps
4. **History:** From intervention details, see how similar cases were resolved

## Validation

The agent's `infer_source_type()` function correctly identifies:
- `PROC_REF:cooling-system.pdf:3:chunk#1` → PROC_REF type
- `GRAPH:INT-2025-0072, INT-2025-0398` → GRAPH type
- `INT-089` → INT type
- `USER` → USER type

The `extract_hypotheses()` parser handles the new formatted tool output and maintains:
- Source deduplication (each source appears once, in highest-confidence hypothesis)
- Confidence thresholds (PRIOR, ACTIVE, LIKELY, CONFIRMED, REJECTED)
- Proper delta tracking (+0.20, +0.30, -0.25, etc.)
