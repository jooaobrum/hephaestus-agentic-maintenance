# Evidence Ledger Format — Usage Guide

## Overview

The evidence ledger now provides **complete source traceability** for every hypothesis. When the agent displays a hypothesis, you can follow the trail back to the original evidence.

## Source Format Reference

### PROC_REF — Procedure References
Format: `PROC_REF:filename:page:chunk#number`

**Example:** `PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#1`

**Meaning:** 
- File: HX-200_Troubleshooting_Procedures
- Page: 3
- Chunk: 1

**Where to find:** In procedure documents stored in Qdrant, section 3.2, diagnostic step 1

---

### GRAPH — Knowledge Graph (Consolidated Patterns)
Format: `GRAPH:INT-XXXX` (one or more intervention IDs)

**Example:** `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185`

**Meaning:** This hypothesis appears in 3 historical interventions. The graph has consolidated these as a known failure pattern.

**Where to find:** Check these intervention IDs in your maintenance history to see how similar cases were resolved.

---

### INT — Direct Intervention Reference
Format: `INT-XXXX`

**Example:** `INT-089`, `INT-156`

**Meaning:** A specific past maintenance case that matches this hypothesis.

**Where to find:** In your corrective maintenance database, look up intervention INT-089 for full details (parts replaced, duration, technician notes, cost).

---

### USER — Technician Confirmation/Rejection
Format: `USER`

**Example:** Source shows `USER` with delta `+0.25` → you confirmed this hypothesis

**Meaning:** The technician (you) explicitly confirmed or ruled out this hypothesis through direct observation or testing.

---

## Reading the Evidence Ledger

### Structure of a Ledger Entry

```
✓ H1 [CONFIRMED] Cooler fouling — conf=0.85 (+0.30 KG match)
  Explanation: Cooler fins fouled by dust/debris restrict cooling fluid flow, causing temperature rise.
  Sources: PROC_REF:HX-200_Troubleshooting_Procedures:3:chunk#2, GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185
```

**Breaking it down:**
- **✓** — Status emoji (CONFIRMED means conf ≥ 0.80)
- **H1** — Hypothesis ID (for reference in conversation)
- **[CONFIRMED]** — Status label
- **Cooler fouling** — The hypothesis statement (what might be wrong)
- **conf=0.85** — Confidence score from 0.0 (impossible) to 1.0 (certain)
- **(+0.30 KG match)** — Last confidence change: +0.30 points from Knowledge Graph match
- **Explanation** — Physical mechanism: why this would cause the symptom
- **Sources** — Where the evidence comes from (how we know this)

---

## Using Sources to Investigate Further

### Scenario 1: You Want to See the Procedure
**Hypothesis shows:** `PROC_REF:cooling-system.pdf:3:chunk#1`

**You do:** 
1. Open the procedure document: cooling-system.pdf
2. Jump to page 3, section chunk#1
3. Follow the diagnostic steps outlined there

---

### Scenario 2: You Want to Learn From Past Cases
**Hypothesis shows:** `GRAPH:INT-2025-0072, INT-2025-0398, INT-2025-0185`

**You do:**
1. Pull up your maintenance database
2. Look at interventions INT-2025-0072, INT-2025-0398, INT-2025-0185
3. See what root causes were found, what was replaced, how long it took
4. Use that to estimate time/parts for your current repair

---

### Scenario 3: You Want to Double-Check Recent History
**Hypothesis shows:** `INT-089, INT-156`

**You do:**
1. Retrieve the full details of INT-089 (parts, cost, outcome)
2. Check if the same technician worked both cases — can you ask them directly?
3. See if the current symptom matches exactly or if there's a twist

---

## Confidence Thresholds

| Score | Status | Meaning |
|-------|--------|---------|
| ≥ 0.80 | ✓ CONFIRMED | Very high confidence — investigate immediately |
| 0.75–0.79 | ◎ LIKELY | Strong confidence — high priority |
| 0.25–0.74 | ○ ACTIVE | Moderate confidence — viable candidate |
| 0.21–0.24 | · PRIOR | Baseline (procedure only) — check if others fail |
| ≤ 0.20 | ✗ REJECTED | Ruled out — technician or evidence contradicts it |

---

## Confidence Delta Meanings

| Delta | Meaning | Example |
|-------|---------|---------|
| +0.20 | Procedure mentions this | Found in troubleshooting guide baseline |
| +0.30 | Known Graph pattern | 3 historical cases match |
| +0.15 | Intervention match | Similar case in recent history |
| +0.25 | Technician confirmed | "Yes, I checked—it's definitely that" |
| −0.25 | Technician ruled out | "No, we verified that's not it" |
| −0.25 | Procedure failed | "I followed the procedure and it didn't help" |

---

## Common Questions

### Q: Why does a hypothesis have multiple sources?
**A:** Multiple sources strengthen the hypothesis. For example:
- Procedure says "check the cooler" ← procedure baseline
- Knowledge graph shows 3 past cases where cooler was the issue ← pattern confirmation
- Recent maintenance shows INT-089 was a cooler problem ← specific precedent

Together, these add up to a high-confidence hypothesis.

---

### Q: What if all sources point to different root causes?
**A:** That's rare. The agent deduplicates sources — each source appears in only one hypothesis. If the knowledge graph points to "cooler fouling" and a procedure points to "low oil," those are separate hypotheses (H1, H2), not mixed.

---

### Q: Can I trust the sources?
**A:** Yes. Sources are:
- **Procedures** — documented standards
- **Knowledge graph** — consolidated from multiple past cases
- **Interventions** — your actual maintenance history
- **USER** — your own confirmation

They're all traceable and verifiable.

---

## Next Steps

When you see a high-confidence hypothesis:

1. **Read the explanation** — understand the physical mechanism
2. **Check the sources** — follow the links to original evidence
3. **Review procedures** — if PROC_REF is listed, that's your action plan
4. **Look at history** — INT and GRAPH sources tell you what was done before
5. **Ask the agent** — "What should I check first?" or "Walk me through the procedure steps"
