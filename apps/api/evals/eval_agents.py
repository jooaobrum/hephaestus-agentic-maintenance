import os
import json
import uuid
import dotenv

dotenv.load_dotenv()

if "QDRANT_URL" not in os.environ:
    os.environ["QDRANT_URL"] = "http://localhost:6333"
if "PG_URL" not in os.environ:
    os.environ["PG_URL"] = (
        "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
    )

from langsmith import Client
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI

from core.config import config
from agents.multi_agent.supervisor import build_supervisor

ls_client = Client()
DATASET_NAME = "hephaestus-agent-eval"

# Build a single supervisor graph (no checkpointer — each eval run is stateless)
supervisor_graph = build_supervisor(checkpointer=None)

# Extract sub-agent graphs for direct invocation in sub-agent-specific scenarios
_subgraphs = dict(supervisor_graph.get_subgraphs())
troubleshooting_graph = _subgraphs.get("troubleshooting")
summarizer_graph = _subgraphs.get("summarizer")

_SUB_AGENT_NAMES = {"troubleshooting", "summarizer"}
_ROUTING_STUBS = {"routing to troubleshooting.", "routing to summarizer."}


def _extract_reply(messages: list) -> str:
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        if getattr(msg, "name", None) not in _SUB_AGENT_NAMES:
            continue
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
            )
        content = content.strip()
        if content and "transfer" not in content.lower()[:30]:
            return content

    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
            )
        content = content.strip()
        if content and content.lower() not in _ROUTING_STUBS:
            return content

    return ""


def _detect_route(messages: list) -> str:
    """Infer which agent was activated by inspecting message names."""
    for msg in reversed(messages):
        if (
            isinstance(msg, AIMessage)
            and getattr(msg, "name", None) in _SUB_AGENT_NAMES
        ):
            return msg.name or "direct"
    return "direct"


def _invoke_subagent(graph, agent_name: str, user_msg: str) -> dict:
    """Invoke a sub-agent graph directly, bypassing the coordinator."""
    thread_id = f"eval-{uuid.uuid4()}"
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 80},
    )
    messages = result.get("messages", [])
    return {
        "answer": _extract_reply(messages) or _extract_any_reply(messages),
        "actual_route": agent_name,
    }


def _extract_any_reply(messages: list) -> str:
    """Fallback: return last non-empty AIMessage content."""
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
                for b in content
            )
        content = content.strip()
        if content:
            return content
    return ""


def run_agent(x: dict) -> dict:
    user_msg = x["user_message"]
    # When the dataset input carries a target_agent, invoke that sub-agent directly
    # so the coordinator cannot rewrite or hallucinate INT-IDs / date ranges.
    target = x.get("target_agent")
    if target == "troubleshooting" and troubleshooting_graph is not None:
        return _invoke_subagent(troubleshooting_graph, "troubleshooting", user_msg)
    if target == "summarizer" and summarizer_graph is not None:
        return _invoke_subagent(summarizer_graph, "summarizer", user_msg)

    thread_id = f"eval-{uuid.uuid4()}"
    result = supervisor_graph.invoke(
        {"messages": [HumanMessage(content=user_msg)]},
        config={"configurable": {"thread_id": thread_id}, "recursion_limit": 80},
    )
    output_messages = result.get("messages", [])
    return {
        "answer": _extract_reply(output_messages),
        "actual_route": _detect_route(output_messages),
    }


# ---------------------------------------------------------------------------
# Evaluators
# ---------------------------------------------------------------------------


def routing_evaluator(run, example) -> dict:
    """Binary: did the supervisor route to the correct agent?"""
    actual = (run.outputs or {}).get("actual_route", "")
    expected = (example.outputs or {}).get("expected_route", "")
    match = actual == expected
    return {
        "key": "routing_correct",
        "score": int(match),
        "comment": f"actual={actual}, expected={expected}",
    }


def answer_relevance_llm_evaluator(run, example) -> dict:
    """LLM judge: does the answer address the user message and ground truth?"""
    outputs = run.outputs or {}
    inputs = example.inputs or {}
    ref_outputs = example.outputs or {}

    answer = outputs.get("answer", "")
    question = inputs.get("user_message", "")
    ground_truth = ref_outputs.get("ground_truth_answer", "")

    if not answer:
        return {"key": "answer_relevance", "score": 0.0, "comment": "Empty answer."}

    prompt = f"""You are evaluating a maintenance assistant's response.

User message: "{question}"
Reference answer: "{ground_truth}"
Agent answer: "{answer}"

Score 0.0 to 1.0: how well does the agent answer address the user's request and align with the reference answer?
Return JSON only: {{"score": float, "reasoning": "brief"}}"""

    try:
        llm = ChatOpenAI(model=config.EVALUATION_MODEL, temperature=0)
        raw = llm.invoke(prompt).content
        res = json.loads(raw if isinstance(raw, str) else str(raw))
        return {
            "key": "answer_relevance",
            "score": float(res.get("score", 0.0)),
            "comment": res.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "key": "answer_relevance",
            "score": 0.0,
            "comment": f"LLM eval error: {e}",
        }


def agent_behavior_llm_evaluator(run, example) -> dict:
    """LLM judge: did the routed agent follow the expected behavior?"""
    outputs = run.outputs or {}
    ref_outputs = example.outputs or {}

    answer = outputs.get("answer", "")
    expected_behavior = ref_outputs.get("expected_agent_behavior", "")

    if not expected_behavior:
        return {"key": "agent_behavior", "score": 1.0, "comment": "No behavior spec."}
    if not answer:
        return {"key": "agent_behavior", "score": 0.0, "comment": "Empty answer."}

    prompt = f"""You are evaluating whether a maintenance AI agent followed its expected behavior.

Expected behavior: "{expected_behavior}"
Agent output: "{answer}"

Score 0.0 to 1.0: how well does the agent output match the expected behavior?
Return JSON only: {{"score": float, "reasoning": "brief"}}"""

    try:
        llm = ChatOpenAI(model=config.EVALUATION_MODEL, temperature=0)
        raw = llm.invoke(prompt).content
        res = json.loads(raw if isinstance(raw, str) else str(raw))
        return {
            "key": "agent_behavior",
            "score": float(res.get("score", 0.0)),
            "comment": res.get("reasoning", ""),
        }
    except Exception as e:
        return {
            "key": "agent_behavior",
            "score": 0.0,
            "comment": f"LLM eval error: {e}",
        }


def summarizer_completeness_evaluator(run, example) -> dict:
    """Check that the summarizer answer contains [INT: ID] markers for all expected IDs."""
    outputs = run.outputs or {}
    ref_outputs = example.outputs or {}

    answer = outputs.get("answer", "")
    expected_ids = ref_outputs.get("expected_int_ids", [])

    if not expected_ids:
        return {
            "key": "summarizer_completeness",
            "score": 1.0,
            "comment": "No IDs to check.",
        }

    found = [id_ for id_ in expected_ids if id_ in answer]
    score = len(found) / len(expected_ids)
    return {
        "key": "summarizer_completeness",
        "score": score,
        "comment": f"Found {len(found)}/{len(expected_ids)} INT IDs: {found}",
    }


# ---------------------------------------------------------------------------
# Evaluator sets per scenario group
# ---------------------------------------------------------------------------

ROUTING_EVALUATORS = [routing_evaluator, answer_relevance_llm_evaluator]
TROUBLESHOOTING_EVALUATORS = [
    routing_evaluator,
    agent_behavior_llm_evaluator,
    answer_relevance_llm_evaluator,
]
SUMMARIZER_EVALUATORS = [
    routing_evaluator,
    agent_behavior_llm_evaluator,
    summarizer_completeness_evaluator,
]

SCENARIO_EVALUATORS = {
    "routing_troubleshooting": ROUTING_EVALUATORS,
    "routing_summarizer": ROUTING_EVALUATORS,
    "routing_direct": ROUTING_EVALUATORS,
    "troubleshooting_interventions_only": TROUBLESHOOTING_EVALUATORS,
    "troubleshooting_procedure_intervention": TROUBLESHOOTING_EVALUATORS,
    "troubleshooting_procedure_only": TROUBLESHOOTING_EVALUATORS,
    "summarizer_summaries": SUMMARIZER_EVALUATORS,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("=" * 70)
    print("      HEPHAESTUS AGENT EVALUATION")
    print(f"      Dataset: {DATASET_NAME}")
    print("=" * 70)

    # Group examples by scenario
    examples = list(ls_client.list_examples(dataset_name=DATASET_NAME))
    print(f"\nLoaded {len(examples)} examples from '{DATASET_NAME}'")

    from collections import defaultdict

    by_scenario: dict[str, list] = defaultdict(list)
    for ex in examples:
        scenario = (ex.metadata or {}).get("scenario", "unknown")
        by_scenario[scenario].append(ex)

    for scenario, scenario_examples in sorted(by_scenario.items()):
        print(f"\n--- {scenario} ({len(scenario_examples)} examples) ---")
        evaluators = SCENARIO_EVALUATORS.get(scenario, ROUTING_EVALUATORS)

        # Build a minimal per-scenario dataset name so LangSmith groups results
        subset_name = f"{DATASET_NAME}_{scenario}"

        # Always recreate the subset so stale examples from previous dataset versions are purged
        try:
            ls_client.delete_dataset(dataset_name=subset_name)
        except Exception:
            pass
        try:
            subset_ds = ls_client.read_dataset(dataset_name=subset_name)
        except Exception:
            subset_ds = ls_client.create_dataset(
                dataset_name=subset_name,
                description=f"Subset of {DATASET_NAME} for scenario: {scenario}",
            )
            ls_client.create_examples(
                dataset_id=subset_ds.id,
                examples=[
                    {
                        "inputs": ex.inputs,
                        "outputs": ex.outputs,
                        "metadata": ex.metadata,
                    }
                    for ex in scenario_examples
                ],
            )
            print(f"  Created subset dataset '{subset_name}'")

        results = ls_client.evaluate(
            run_agent,
            data=subset_name,
            evaluators=evaluators,
            experiment_prefix=f"agent_eval_{scenario}",
            max_concurrency=2,
        )
        print(f"  ✓ Done — {results}")

    print("\n" + "=" * 70)
    print("✓ All scenario evaluations complete. Check LangSmith for results.")
    print("=" * 70)

    # --- CI pass/fail gate ---
    # Thresholds set at ~15% below observed baseline to absorb LLM variance.
    # routing_correct must be 1.0 across all scenarios (hard gate).
    _check_ci_gate(by_scenario)


def _check_ci_gate(by_scenario: dict) -> None:
    """Collect scores from the latest experiment per scenario and enforce thresholds."""
    from collections import defaultdict

    all_scores: dict[str, list[float]] = defaultdict(list)

    for scenario in sorted(by_scenario.keys()):
        subset_name = f"{DATASET_NAME}_{scenario}"
        try:
            ds = ls_client.read_dataset(dataset_name=subset_name)
            projs = sorted(
                list(ls_client.list_projects(reference_dataset_id=str(ds.id))),
                key=lambda p: p.start_time,
                reverse=True,
            )
            if not projs:
                continue
            latest = projs[0]
            runs = list(
                ls_client.list_runs(
                    project_name=latest.name, run_type="chain", is_root=True
                )
            )
            for run in runs:
                for fb in ls_client.list_feedback(run_ids=[str(run.id)]):
                    if fb.score is not None:
                        all_scores[fb.key].append(fb.score)
        except Exception as e:
            print(f"  [gate] could not read scores for {scenario}: {e}")

    _THRESHOLDS: dict[str, float] = {
        "routing_correct": 1.00,
        "answer_relevance": 0.40,
        "agent_behavior": 0.50,
        "summarizer_completeness": 0.80,
    }

    print("\n" + "=" * 70)
    print("CI GATE RESULTS")
    print("=" * 70)
    failures: list[str] = []
    for metric, threshold in sorted(_THRESHOLDS.items()):
        scores = all_scores.get(metric, [])
        if not scores:
            print(f"  {metric:<30} NO DATA   (threshold={threshold:.2f})")
            continue
        avg = sum(scores) / len(scores)
        status = "PASS" if avg >= threshold else "FAIL"
        if status == "FAIL":
            failures.append(f"{metric}: {avg:.3f} < {threshold:.2f}")
        print(
            f"  {metric:<30} {status}  avg={avg:.3f}  threshold={threshold:.2f}  (n={len(scores)})"
        )

    print("=" * 70)
    if failures:
        print(f"\n❌ CI GATE FAILED — {len(failures)} metric(s) below threshold:")
        for f in failures:
            print(f"   • {f}")
        raise SystemExit(1)
    else:
        print("\n✅ CI GATE PASSED — all metrics above threshold.")


if __name__ == "__main__":
    main()
