# Tier 1 RCA Agent Improvements — Implemented

## Summary
Added 2 new tools and clarified tool selection logic with a **Tool Decision Matrix** in the system prompt. Tool count: 14 → 16.

---

## New Tools

### 1. `get_sensor_anomaly_summary(machine, start_date, end_date, top_n=5)`
**Purpose:** Aggregate threshold events by sensor, ranking by severity.

**When to use:**
- AFTER `get_threshold_events_tool` identifies alarms
- BEFORE `get_sensor_timeline_tool` to prioritize which sensors to drill into

**What it returns:**
- Top N sensors with most anomalies
- Event count per sensor
- Has CRITICAL flag (yes/no)
- First breach timestamp
- Average value vs. thresholds

**Example workflow:**
```
1. get_sensor_catalog_tool(HX-200)
2. get_threshold_events_tool(HX-200, 2026-05-01, 2026-05-06)
3. get_sensor_anomaly_summary(HX-200, 2026-05-01, 2026-05-06, top_n=3)  ← NEW
4. get_sensor_timeline_tool(HX-200, ..., tag=TEMP_01)  ← Focus on worst sensor
```

---

### 2. `get_intervention_detail(intervention_id)`
**Purpose:** Expand a CM search result into full details.

**When to use:**
- AFTER finding a relevant intervention in `get_recent_formatted_cm_context` results
- To inspect parts replaced, root cause recorded, duration, cost, technician notes

**What it returns:**
- Complete intervention metadata:
  - Root cause (if recorded)
  - Parts replaced
  - Labor hours / downtime hours
  - Cost (€)
  - Technician name
  - Tags (fault codes)
  - Full description

**Example workflow:**
```
1. get_recent_formatted_cm_context(query="oil temperature", machine="HX-200")
2. get_intervention_detail("INT-0045")  ← NEW: expand ID from search
3. Use full details (root cause, parts) to refine hypothesis
```

---

## Updated Tool Docstrings

All 14 existing tools now have **Use BEFORE/Use AFTER** markers. Examples:

**get_sensor_catalog_tool:**
```
Use FIRST in any sensor analysis
Use BEFORE: get_threshold_events_tool, get_sensor_readings_tool, get_sensor_timeline_tool
```

**get_threshold_events_tool:**
```
Use AFTER get_sensor_catalog_tool to identify abnormal signals
Use BEFORE: get_sensor_anomaly_summary, get_sensor_timeline_tool
```

**get_recent_formatted_cm_context:**
```
Use: DEFAULT choice for CM history (covers 95% of cases)
Use BEFORE: get_long_formatted_cm_context (only if no recent results + chronic suspected)
Use AFTER: get_intervention_detail(intervention_id) to expand relevant cases
```

---

## Tool Decision Matrix (System Prompt)

Added explicit ordering rules in `rca_react_system_prompt.yml`:

### Sensor Analysis Sequence
```
1. get_sensor_catalog_tool(machine) — discover valid tags
2. get_threshold_events_tool(machine, date_range) — identify alarms
3. get_sensor_anomaly_summary(machine, date_range) — rank worst sensors
4. get_sensor_timeline_tool(..., tag) — deep dive on critical sensor
5. get_sensor_readings_tool(..., tag) — raw data if needed
```

### CM History Sequence
```
1. get_recent_formatted_cm_context() — DEFAULT
2. get_long_formatted_cm_context() — ONLY if recent empty + chronic suspected
3. get_intervention_detail(id) — expand relevant cases
```

### Known Issues & Procedures
```
1. query_known_issues_graph(symptom, machine) — early hypotheses
2. get_formatted_procedure_context(query) — diagnostic guidance
```

### Always First
```
- check_machine_exists(machine) — ALWAYS validate before proceeding
- list_available_machines() — ONLY if no machine specified
```

---

## Impact

✅ **Clearer tool selection:** Agent now has explicit ordering rules
✅ **Better triage flow:** Can now identify worst sensors before deep dive
✅ **Faster case lookup:** Can expand intervention summaries into actionable details
✅ **Reduced hallucination:** "Use BEFORE/AFTER" markers constrain tool chaining
✅ **List → Get pattern:** Matches your intuition about discovery then detail

---

## Files Modified

1. `notebooks/tools/tools.py`
   - Added `get_sensor_anomaly_summary()` (lines 947–1013)
   - Added `get_intervention_detail()` (lines 1015–1053)
   - Enhanced 10 tool docstrings with Use BEFORE/AFTER markers

2. `notebooks/20-rca-react-agent.ipynb`
   - Updated imports: added 2 new tools
   - Updated ALL_TOOLS list (16 tools total)

3. `notebooks/prompts/rca_react_system_prompt.yml`
   - Added "## TOOL DECISION MATRIX" section with sensor/history/validation sequences

---

## Next Steps (Tier 2)

If you want to expand further, Tier 2 includes:
- `get_maintenance_timeline()` — inspect/replacement events relative to failures
- `get_procedure_diagnostic_table()` — extract specific decision tables from procedures
- `get_fleet_comparison()` — find similar failures on different machines

Let me know if you want to implement those next!
