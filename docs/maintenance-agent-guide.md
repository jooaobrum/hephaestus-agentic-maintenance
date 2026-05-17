# Agent — Maintenance Investigation Assistant
## Practical Guide for Technicians and Engineers

**Who this is for:**  
Technicians, operators, and engineers using Agent in daily maintenance work.

No AI or technical background is required.

---

# Part 1 — What Agent Actually Does

Agent is an AI-powered maintenance investigation assistant.

Think of it like an experienced maintenance engineer who:
- knows the procedures by heart
- remembers past failures across the fleet
- compares similar breakdowns automatically
- guides investigations step by step

Agent does NOT simply guess a fix.

Its job is to:
1. gather evidence
2. build competing hypotheses
3. eliminate wrong possibilities
4. converge toward the most likely root cause

---

# Part 2 — How Agent Investigates Problems

## Agent works like a real investigation

When you report a problem, Agent:
1. gathers machine history
2. checks procedures and known failures
3. compares similar machines
4. builds multiple hypotheses
5. asks the fastest checks that eliminate the wrong branches

The goal is to:
- find the MOST likely cause
- avoid unnecessary repairs
- reduce downtime
- prevent repeated failures

---

## Why Agent Asks Questions

Every question Agent asks has a purpose.

A good question should:
- confirm one hypothesis
OR
- eliminate another

Examples:

| Question | Why Agent asks it |
|---|---|
| “Is the cooling fan spinning?” | Separates fan failure from cooler fouling |
| “Did this start after maintenance?” | Identifies maintenance-induced issues |
| “Does the vibration happen continuously?” | Separates mechanical wear from intermittent process instability |

Agent is trying to REMOVE possibilities, not just guess one.

---

# Part 3 — Best Practices for Faster Diagnosis

## Best Practice #1 — Always Include the Machine ID

### GOOD
```text
HX-200 hydraulic press — oil temperature alarm at 83°C.
```

### BAD
```text
The press is overheating.
```

### Why it matters
Without the machine ID, Agent cannot retrieve:
- maintenance history
- procedures
- known issues
- fleet comparisons

---

## Best Practice #2 — Describe Observations, Not Your Diagnosis

### GOOD
```text
Grinding noise near the pump housing during startup.
```

### BAD
```text
The pump bearings are bad.
```

### Why it matters
Your observations help Agent investigate objectively.
Starting with a diagnosis can bias the investigation too early.

---

## Best Practice #3 — Include Timing Information

### GOOD
```text
Started yesterday after hydraulic oil replacement.
```

### BAD
```text
It's been happening recently.
```

### Why it matters
Timing helps Agent distinguish between:
- sudden failures
- gradual degradation
- maintenance-related issues
- intermittent faults

---

## Best Practice #4 — Tell Agent What You Already Checked

### GOOD
```text
Fan is running normally. Oil level is at midpoint on sight glass.
```

### BAD
```text
I checked some things already.
```

### Why it matters
Every completed check helps Agent eliminate possibilities faster.

---

## Best Practice #5 — Give Observational Answers

### GOOD
```text
Yes, fan is spinning but airflow feels weaker than usual.
```

### BAD
```text
Yes.
```

### Why it matters
Small details often completely change the diagnosis.

---

## Best Practice #6 — Explain Why You Cannot Perform a Check

### GOOD
```text
Cannot inspect cooler — machine is still running production.
```

### BAD
```text
Didn't check.
```

### Why it matters
Agent adjusts the investigation based on accessibility and operational constraints.

---

## Best Practice #7 — Keep One Session Per Problem

### GOOD
```text
HX-200 — oil temperature alarm since this morning.
```

### BAD
```text
HX-200 overheating and CB-100 vibrating.
```

### Why it matters
Mixing problems confuses the investigation and contaminates the evidence trail.

---

# Part 4 — GOOD vs BAD Investigation Examples

## Example 1 — Weak vs Strong Opening Prompt

### BAD
```text
Machine not working.
```

### Problems
- no machine ID
- no symptom details
- no timing
- no prior checks

---

### GOOD
```text
CNC-500 on line 3 — fault E-07 during drilling cycle.
Started this morning during second shift.
Machine stops about 20 minutes into operation.
No prior checks performed yet.
```

### Why this works
- machine identified
- fault code included
- timing included
- operational context included
- current status included

---

## Example 2 — Weak vs Strong Follow-Up Answer

### Agent asks
```text
Is the cooling fan running?
```

### BAD
```text
Yes.
```

### GOOD
```text
Yes, fan is running normally but sounds louder than usual.
```

### Why this works
Additional observations often provide hidden clues.

---

## Example 3 — Weak vs Strong Constraint Reporting

### BAD
```text
I can't check that.
```

### GOOD
```text
Cannot access the cooler because the safety guard is locked during production.
```

### Why this works
Agent can adapt the investigation based on operational limitations.

---

## Example 4 — Weak vs Strong Fault Reporting

### BAD
```text
E-04 fault.
```

### GOOD
```text
Fault E-04 appears 30 seconds after spindle startup during milling cycle.
Machine stops completely after the alarm.
```

### Why this works
Context matters more than the fault code alone.

---

# Part 5 — How to Accelerate Diagnosis

The fastest investigations usually include:

## Prior checks
```text
Filter replaced yesterday. Fan confirmed operational.
```

## Timing
```text
Started immediately after maintenance shutdown.
```

## Frequency
```text
Occurs twice per shift under high load.
```

## Environmental conditions
```text
Ambient temperature in shop is around 38°C today.
```

## Operator observations
```text
Operator noticed burning smell before shutdown.
```

## Constraints
```text
Cannot stop machine until shift change.
```

---

# Part 6 — Asking for Immediate Recommendations

You can skip detailed investigation at any time.

### Examples
```text
Just give me the action plan.
```

```text
What should I check first?
```

```text
I need the fastest likely fix.
```

Agent will then provide:
- most likely causes
- recommended first checks
- prioritized action plans

based on available evidence.

---

# Part 7 — Real Investigation Example

```text
You:
HX-200 hydraulic press — oil temperature alarm at 83°C.
Started this morning. Oil level looks normal.

Agent:
Based on history and procedures, the leading hypotheses are:
1. Cooler fouling (60%)
2. Cooling fan failure (35%)

The fastest discriminator is fan operation.

Is the fan currently spinning?
- YES → reduces likelihood of fan failure
- NO → strongly supports fan failure

You:
Yes, fan is spinning normally.

Agent:
Good — that significantly reduces fan failure likelihood.

Current ranking:
1. Cooler fouling (80%)
2. Fan failure (10%)

Next check:
Can you inspect whether the cooler fins are clogged?

You:
Yes, fins are heavily clogged.

Agent:
That confirms the primary cause.
Restricted airflow through the cooler is causing overheating.

Recommended actions:
1. Shut down and cool machine
2. Clean cooler fins
3. Restart and monitor oil temperature
```

---

# Part 8 — Common Mistakes to Avoid

| Mistake | Better Approach |
|---|---|
| Starting with your diagnosis | Describe observations instead |
| One-word answers | Add details and observations |
| Hiding prior checks | Report everything already checked |
| Mixing multiple machines | Use one session per problem |
| Giving only fault codes | Add timing and behavior context |
| Saying “didn't check” | Explain WHY you couldn't check |

---

# Part 9 — Quick Reference Card

| What you want | What to type |
|---|---|
| Start investigation | “HX-200 oil temp alarm showing 83°C” |
| Report prior checks | “Fan checked — operating normally” |
| Skip investigation | “Just give me the action plan” |
| Ask how to do a check | “How do I measure spindle run-out?” |
| Explain constraints | “Cannot stop machine until next shift” |
| End session | “Got it” / “Done” / “Thanks” |

---

# Final Tips

Agent works best when:
- observations are specific
- machine IDs are included
- prior checks are reported
- timing is described
- answers are factual and concise

You do NOT need perfect technical language.

Just describe:
- what you see
- what you hear
- what changed
- what you already checked

Agent will handle the investigation logic.
