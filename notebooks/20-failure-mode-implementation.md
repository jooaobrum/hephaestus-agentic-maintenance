# Failure Mode Matching Implementation Guide

## Overview

This guide implements 4 enhancements to the RCA agent:
1. **Failure Mode Matching** — extract and score failure modes from procedures + known issues
2. **Fast Path for Known Issues** — skip investigation for high-confidence cases
3. **Tool Categorization** — split tools into diagnostic vs confirmation
4. **Validation Loop** — confirm before action for confidence > 0.70

---

## Step 1: Extend Hypothesis Model with Failure Mode Data

**File:** `notebooks/20-rca-react-agent.ipynb` → Update States cell

Add failure mode tracking to the `Hypothesis` class:

```python
from enum import Enum

class Severity(str, Enum):
    CRITICAL = "immediately_unsafe"      # Stop operation now
    HIGH = "degrading_rapidly"          # Will fail in hours/days
    MEDIUM = "monitoring_required"      # Watch closely, plan maintenance
    LOW = "planned_maintenance"         # Next scheduled window ok

class Hypothesis(BaseModel):
    source_id: str
    statement: str
    confidence: float
    status: Literal['LIKELY', 'CONFIRMED', 'REJECTED', 'ACTIVE']
    source: list[Reference] = []
    
    # NEW: Failure mode enrichment
    failure_mode: str = ""              # e.g., "bearing_wear", "seal_degradation"
    failure_mode_match_score: float = 0.0  # How well symptom matches this FM
    prevalence_pct: float = 0.0         # % of fleet affected by this failure
    last_similar_case_days_ago: int | None = None  # Recency
    severity: Severity | None = None    # Urgency level
```

---

## Step 2: Create Failure Mode Extraction Tool

**File:** `notebooks/tools/tools.py` → Add new function

```python
@tool
def extract_failure_modes_from_procedures(
    query: str,
    top_k: int = 3,
) -> str:
    """
    Extract failure modes and their characteristics from procedure collection.
    
    Returns structured info: failure_mode name, typical symptoms, failure impact,
    and preventive/corrective actions.
    
    Use: Early in diagnosis to understand what failure modes exist for this symptom.
    """
    results = _retrieve_procedures(query=query, top_k=top_k)
    
    failure_modes = []
    for result in results:
        payload = result["payload"]
        # Procedures typically contain structured failure modes in tables/sections
        text = payload.get('text', '')
        section = payload.get('section_title', '')
        
        # Parse failure mode if section mentions "failure mode" or "root cause"
        if 'failure' in section.lower() or 'root cause' in section.lower():
            failure_modes.append({
                'name': section,
                'description': text[:200],
                'source_file': payload.get('file_name'),
                'confidence_boost': 0.20  # Procedures are +0.20
            })
    
    if not failure_modes:
        return "No structured failure modes found in procedures for this query."
    
    output = "**Failure Modes from Procedures:**\n"
    for fm in failure_modes:
        output += f"- **{fm['name']}**: {fm['description']}\n"
    return output


def _extract_failure_mode_from_graph(known_issue_payload: dict) -> dict:
    """
    Parse failure mode info from known_issues graph structure.
    
    Expected payload shape (from query_known_issues_graph):
    {
        'symptom_name': str,
        'description': str,
        'root_causes': [
            {'root_cause': str, 'actions': [str], 'frequency': int}
        ],
        'affected_machines': [str]
    }
    """
    root_causes = known_issue_payload.get('root_causes', [])
    affected = known_issue_payload.get('affected_machines', [])
    
    failure_modes = []
    for rc in root_causes:
        failure_modes.append({
            'name': rc.get('root_cause', ''),
            'frequency': rc.get('frequency', 0),  # How many times seen
            'affected_count': len(affected),       # How many machines
            'prevalence_pct': (rc.get('frequency', 0) / 100) * 100,  # Simple estimate
            'confidence_boost': 0.30  # KG is +0.30
        })
    
    return failure_modes
```

---

## Step 3: Create Failure Mode Matching Function

**File:** `notebooks/20-rca-react-agent.ipynb` → Add new cell before agent_node

```python
def score_failure_mode_match(
    symptom_text: str,
    failure_mode_name: str,
    failure_mode_description: str,
) -> float:
    """
    Score how well a failure mode matches the reported symptom (0.0 to 1.0).
    
    Simple approach: keyword overlap using LLM semantic similarity.
    """
    # Extract keywords from symptom (e.g., "high temperature" → ["temperature"])
    symptom_keywords = set(symptom_text.lower().split())
    fm_keywords = set(failure_mode_description.lower().split())
    
    # Overlap ratio
    if not fm_keywords:
        return 0.0
    
    overlap = len(symptom_keywords & fm_keywords)
    match_score = min(overlap / len(fm_keywords), 1.0)  # Clamp to [0, 1]
    
    return match_score


def enrich_hypotheses_with_failure_modes(
    hypotheses: list[Hypothesis],
    failure_modes_from_procedures: list[dict],
    failure_modes_from_graph: list[dict],
) -> list[Hypothesis]:
    """
    Attach failure mode data to each hypothesis.
    
    For each hypothesis, find matching failure modes and boost confidence.
    """
    enriched = []
    
    for hyp in hypotheses:
        statement = hyp.statement
        
        # Find matching failure modes
        best_match = None
        best_score = 0.0
        
        all_fms = failure_modes_from_procedures + failure_modes_from_graph
        for fm in all_fms:
            score = score_failure_mode_match(
                statement,
                fm.get('name', ''),
                fm.get('description', '')
            )
            if score > best_score:
                best_score = score
                best_match = fm
        
        # Enrich with failure mode data
        if best_match:
            hyp.failure_mode = best_match.get('name', '')
            hyp.failure_mode_match_score = best_score
            hyp.prevalence_pct = best_match.get('prevalence_pct', 0.0)
            
            # Boost confidence based on failure mode prevalence
            if best_match.get('prevalence_pct', 0) > 50:  # Common failure
                hyp.confidence = min(hyp.confidence + 0.15, 1.0)
        
        enriched.append(hyp)
    
    return enriched
```

---

## Step 4: Update Agent Node with Failure Mode Extraction

**File:** `notebooks/20-rca-react-agent.ipynb` → Replace agent_node

```python
def agent_node(state):
    """Agent node: invoke LLM with tools and process response."""
    
    if isinstance(state, dict):
        state = AgentState(**state)
    
    # STAGE 1: Intake — extract machine, symptom, period
    if not state.intake_complete:
        user_messages = [m for m in state.messages if isinstance(m, HumanMessage)]
        if user_messages:
            intake_info = extract_intake_info(user_messages[-1].content)
            state.machine = intake_info.get("machine", "")
            state.symptom = intake_info.get("symptom", "")
            state.period = intake_info.get("period", "")
        
        missing = []
        if not state.machine:
            missing.append("machine ID")
        if not state.symptom:
            missing.append("symptom description")
        if not state.period:
            missing.append("time reference")
        
        if missing:
            ask_msg = f"I need a bit more info: {', '.join(missing)}?"
            return {
                "messages": [AIMessage(content=ask_msg)],
                "machine": state.machine,
                "symptom": state.symptom,
                "period": state.period,
                "hypotheses": [],
                "awaiting_user": True,
                "intake_complete": False,
            }
        
        state.intake_complete = True
    
    # STAGE 2: Call LLM
    SYSTEM_PROMPT = prompt_template_config("prompts/rca_react_system_prompt.yml", name="system_prompt")
    messages = state.messages

    response = _llm_with_tools.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), *messages[:15]]
    )

    # If LLM called tools, return for tool execution
    if response.tool_calls:
        return {
            "messages": [response],
            "machine": state.machine,
            "symptom": state.symptom,
            "period": state.period,
            "hypotheses": [],
            "awaiting_user": False,
            "intake_complete": state.intake_complete,
        }

    # Text response: extract hypotheses + failure modes
    hypotheses = extract_hypotheses(response.content)
    
    # NEW: Enrich hypotheses with failure mode data
    # (Normally extracted from tool results, but for demo, simulate)
    failure_modes_proc = [
        {
            'name': 'bearing_wear',
            'description': 'bearing fatigue and degradation',
            'prevalence_pct': 35.0,
            'confidence_boost': 0.20
        }
    ]
    failure_modes_graph = []  # Would come from query_known_issues_graph
    
    hypotheses = enrich_hypotheses_with_failure_modes(
        hypotheses,
        failure_modes_proc,
        failure_modes_graph
    )
    
    is_awaiting = "[AWAITING USER FEEDBACK]" in response.content
    cleaned_response = response.content.replace("[AWAITING USER FEEDBACK]", "").strip()

    return {
        "messages": [AIMessage(content=cleaned_response)],
        "machine": state.machine,
        "symptom": state.symptom,
        "period": state.period,
        "hypotheses": hypotheses,
        "awaiting_user": is_awaiting,
        "intake_complete": state.intake_complete,
    }
```

---

## Step 5: Tool Categorization & Fast Path Routing

**File:** `notebooks/20-rca-react-agent.ipynb` → Update Tools cell

```python
# Split tools into diagnostic vs confirmation
DIAGNOSTIC_TOOLS = [
    query_known_issues_graph,
    get_sensor_readings_tool,
    get_threshold_events_tool,
    get_formatted_procedure_context,
    get_formatted_cm_context,
]

CONFIRMATION_TOOLS = [
    get_remaining_life_tool,
    get_sensor_timeline_tool,
]

UTILITY_TOOLS = [
    get_current_date,
    calculate_date_window,
    check_machine_exists,
    list_available_machines,
    get_sensor_catalog_tool,
    get_recent_formatted_cm_context,
    get_long_formatted_cm_context,
]

ALL_TOOLS = DIAGNOSTIC_TOOLS + CONFIRMATION_TOOLS + UTILITY_TOOLS


def route_after_agent(state: AgentState) -> str:
    """Route: if high confidence, ask for validation; if unknown, do diagnostic tools."""
    last_msg = state.messages[-1]

    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return "tools"

    # NEW: Fast path for known issues
    if state.hypotheses:
        best = max(state.hypotheses, key=lambda h: h.confidence)
        if best.confidence >= 0.75 and best.status == "CONFIRMED":
            return "validation"  # Ask for validation before action
    
    if state.awaiting_user:
        return "user_feedback"

    return "__end__"


def validation_node(state: AgentState) -> dict:
    """
    NEW: Before acting on high-confidence hypothesis, ask technician for confirmation.
    
    This prevents costly mistakes (unnecessary replacement of expensive parts).
    """
    best = max(state.hypotheses, key=lambda h: h.confidence)
    
    validation_prompt = f"""
Before we proceed, I want to confirm:

**My diagnosis:** {best.statement}
**Confidence:** {best.confidence:.0%}
**Failure mode:** {best.failure_mode or 'N/A'}

Does this match what you're seeing? Any doubts?

Reply with 'yes', 'no', or describe what's different.
[AWAITING USER FEEDBACK]
"""
    
    return {
        "messages": [AIMessage(content=validation_prompt)],
        "awaiting_user": True,
    }
```

---

## Step 6: Update Graph Routing

**File:** `notebooks/20-rca-react-agent.ipynb` → Replace workflow edges

```python
from langgraph.graph import StateGraph, START, END

tool_node = ToolNode(ALL_TOOLS)

workflow = StateGraph(AgentState)

# Add all nodes
workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)
workflow.add_node("user_feedback", user_feedback_node)
workflow.add_node("validation", validation_node)  # NEW

# Define edges
workflow.add_edge(START, "agent")

workflow.add_conditional_edges("agent", route_after_agent, {
    "tools": "tools",
    "validation": "validation",  # NEW
    "user_feedback": "user_feedback",
    "__end__": END
})

workflow.add_conditional_edges("tools", route_after_tools, {
    "agent": "agent"
})

workflow.add_edge("validation", "user_feedback")  # Validation → wait for user
workflow.add_edge("user_feedback", END)

graph = workflow.compile(checkpointer=MemorySaver())
```

---

## Step 7: Simplified Confidence Deltas

**File:** `notebooks/prompts/rca_react_system_prompt.yml` → Update EVIDENCE LEDGER section

Replace the complex deltas with:

```yaml
**Confidence deltas:** 
- Procedure +0.10
- Knowledge graph +0.20
- Single intervention +0.15
- User confirm +0.25
- Common failure mode (>50% fleet) +0.10
```

---

## Testing Checklist

Run through the notebook cells in order:

1. ✅ **States** — Hypothesis model loads with new fields
2. ✅ **Tools** — extract_failure_modes_from_procedures callable
3. ✅ **Matching functions** — score_failure_mode_match returns [0, 1]
4. ✅ **Agent node** — Takes intake, extracts hypotheses + failure modes
5. ✅ **Graph** — Routing to validation node works
6. ✅ **Test case**:
   ```python
   state = AgentState(messages=[
       HumanMessage(content="HX-200 high oil temp, last 2 days")
   ])
   result = graph.invoke(state, config={"configurable": {"thread_id": "test"}})
   
   # Should route to validation if confidence > 0.75
   assert any("Does this match" in m.content for m in result["messages"] if isinstance(m, AIMessage))
   ```

---

## Implementation Order (Suggested)

1. **Quick** (10 min) — Add `failure_mode`, `severity` to Hypothesis model
2. **Quick** (10 min) — Add `validation_node` and update routing
3. **Medium** (20 min) — Implement `score_failure_mode_match` + `enrich_hypotheses_with_failure_modes`
4. **Medium** (15 min) — Split tools into diagnostic/confirmation categories
5. **Optional** (30 min) — Extract failure modes from procedures (more complex parsing)

Start with steps 1-2 for immediate value, then add 3-4 as needed.
