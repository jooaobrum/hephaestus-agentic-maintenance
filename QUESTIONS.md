# Multi-Intent Questions for Agentic Systems

**Objective:** Test multi-intent questions for agentic systems by mixing 2 symptoms per question, requiring the agent to retrieve and consolidate multiple fault histories.

---

## Simple (single machine / one symptom pair)

### Q1 — Two symptoms, same machine

> "The CB-200 conveyor is showing a belt misalignment alarm and the operator also reports audible grinding noise from the rollers. What are the likely root causes and what steps should the technician follow?"

- **Mixes:** `Belt misalignment >15mm` (B-001) + `Carry roller bearing alarm` (B-006)
- **Machine:** CB-200, mechanical subsystem
- **Agent challenge:** Retrieve two separate fault histories and consolidate the answer

---

### Q2 — Two symptoms, same subsystem but different fault codes

> "On the HX-200 hydraulic press, we have low hydraulic pressure below 150 bar and the pressure relief valve keeps opening. What does maintenance history say about this combination?"

- **Mixes:** `Low hydraulic pressure (<150 bar)` (E-001) + `Pressure relief valve open` (H-004 area)
- **Machine:** HX-200, hydraulic subsystem
- **Agent challenge:** Differentiate root causes (pump wear vs. contamination) for each fault

---

## Complex (cross-machine / cross-subsystem reasoning)

### Q3 — Two symptoms requiring temporal and relational reasoning

> "A cold rolling mill is simultaneously showing roll bearing high temperature above 80°C and a drive motor overcurrent trip. Both started within the same shift. What are the past interventions that dealt with both symptoms together, and what was the typical repair sequence?"

- **Mixes:** `Roll bearing high temperature (>80C)` (R-003) + `Drive motor overcurrent` (R-002)
- **Machines:** CR-100 / CR-150 — comments mention "vibration also elevated"
- **Agent challenge:** Retrieve co-occurring events, infer causality (bearing wear → motor overload), and order the repair steps

---

### Q4 — Two symptoms across subsystems with a follow-up diagnostic step

> "On the induction hardening system, we are seeing frequency deviation over 2% and at the same time the quench temperature is out of range. Could these two faults be related? What do past interventions say about their root causes, which fault should be prioritized, and has any technician documented a combined fix procedure?"

- **Mixes:** `Frequency deviation >2%` (H-004, Electrical) + `Quench temperature out of range` (C-005, Process)
- **Subsystems:** Electrical + Process (cross-subsystem)
- **Agent challenge:** Reason about cross-subsystem coupling, prioritize faults, and search `related_intervention` links and `comments` for a combined procedure

---

## Summary Table

| # | Question Type | Intents | Key Challenge |
|---|---|---|---|
| Q1 | Simple — same machine | Fault lookup × 2 | Merge two retrieval results |
| Q2 | Simple — same subsystem | Fault lookup × 2 + history scan | Distinguish root causes for co-located faults |
| Q3 | Complex — cross-fault causal chain | Co-occurrence retrieval + temporal reasoning + sequence inference | Infer causality and repair order |
| Q4 | Complex — cross-subsystem | Cross-subsystem retrieval + causal reasoning + prioritization + procedure search | 4 distinct sub-intents in one query |
