# -*- coding: utf-8 -*-
"""Evaluation pipeline for Hephaestus Multi-Agent System (Coordinator, Summarizer, and Troubleshooting).

Automatically seeds evaluation datasets in LangSmith if they do not already exist,
defines custom metadata and correctness evaluators, and runs evaluations for each agent.
"""

import os
import uuid
import json
import dotenv

# Load environment variables from root .env
dotenv.load_dotenv()

# Set local default configurations if running on the host machine outside Docker
if "PG_URL" not in os.environ:
    os.environ["PG_URL"] = (
        "postgresql://langgraph_user:langgraph_password@localhost:5433/langgraph_db"
    )
if "QDRANT_URL" not in os.environ:
    os.environ["QDRANT_URL"] = "http://localhost:6333"

from langsmith import Client
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from core.config import config
from agents.multi_agent.nodes import (
    coordinator_node,
    troubleshooting_node,
    summarizer_node,
)

# Initialize clients
ls_client = Client()

# --- Auto-Seeding Configurations & Data ---

COORDINATOR_SEED_EXAMPLES = [
    {
        "inputs": {
            "query": "Analyze the high vibration sensor warning on CNC-500 from yesterday"
        },
        "outputs": {
            "next_agent": "troubleshooting",
            "coordinator_final_answer": False,
            "machine_id": "CNC-500",
            "symptom": "high vibration sensor warning",
            "task": "troubleshooting",
        },
    },
    {
        "inputs": {
            "query": "IH-300 cooling flow has collapsed. Let's diagnose it immediately."
        },
        "outputs": {
            "next_agent": "troubleshooting",
            "coordinator_final_answer": False,
            "machine_id": "IH-300",
            "symptom": "cooling flow has collapsed",
            "task": "troubleshooting",
        },
    },
    {
        "inputs": {
            "query": "Summarize all corrective maintenance interventions on Heat Exchanger HX-200 between 2025-11-01 and 2025-12-01 and create a known case template."
        },
        "outputs": {
            "next_agent": "summarizer",
            "coordinator_final_answer": False,
            "machine_id": "HX-200",
            "symptom": None,
            "task": "summarize and template",
            "period": "2025-11-01 to 2025-12-01",
        },
    },
    {
        "inputs": {
            "query": "Generate a known issue template for conveyor belt CB-100 bearing wear issues recorded this quarter."
        },
        "outputs": {
            "next_agent": "summarizer",
            "coordinator_final_answer": False,
            "machine_id": "CB-100",
            "symptom": "bearing wear issues",
            "task": "build known issue template",
        },
    },
    {
        "inputs": {"query": "Hello! What is your name and what can you help me with?"},
        "outputs": {
            "next_agent": "",
            "coordinator_final_answer": True,
            "machine_id": None,
            "symptom": None,
            "task": None,
        },
    },
]

TROUBLESHOOTING_SEED_EXAMPLES = [
    {
        "inputs": {
            "query": "Machine IH-300 has high coil temperature. Let's diagnose.",
            "machine_id": "IH-300",
            "symptom": "high coil temperature",
            "task": "diagnose heating issue",
        },
        "outputs": {
            "expected_keywords": ["cooling", "pump", "flow", "coil", "heater"],
            "expected_root_cause": "coil cooling flow fault",
        },
    },
    {
        "inputs": {
            "query": "CNC-500 is reporting severe spindle vibration. What's the plan?",
            "machine_id": "CNC-500",
            "symptom": "severe spindle vibration",
            "task": "spindle bearing analysis",
        },
        "outputs": {
            "expected_keywords": [
                "bearing",
                "vibration",
                "spindle",
                "lubrication",
                "wear",
            ],
            "expected_root_cause": "bearing wear",
        },
    },
    {
        "inputs": {
            "query": "HX-200 heat exchanger has a low oil pressure alarm and hot oil. What does the procedure suggest?",
            "machine_id": "HX-200",
            "symptom": "low oil pressure and hot oil",
            "task": "diagnose oil leak/wear",
        },
        "outputs": {
            "expected_keywords": ["seal", "leak", "oil", "fouling", "cavitation"],
            "expected_root_cause": "seal wear",
        },
    },
]

SUMMARIZER_SEED_EXAMPLES = [
    {
        "inputs": {
            "query": "Summarize all corrective maintenance interventions on machine HX-200 from November 2025 to December 2025.",
            "machine_id": "HX-200",
            "symptom": "seal wear",
            "task": "summarize CM interventions",
            "period": "2025-11-01 to 2025-12-01",
        },
        "outputs": {
            "expected_keywords": [
                "summary",
                "HX-200",
                "intervention",
                "template",
                "root cause",
            ],
            "expected_template_elements": [
                "symptom_name",
                "description",
                "root_causes",
            ],
        },
    },
    {
        "inputs": {
            "query": "Build a known case template for conveyor belt CB-100 bearing wear issues recorded between 2025-10-01 and 2025-12-31.",
            "machine_id": "CB-100",
            "symptom": "bearing wear",
            "task": "build case template",
            "period": "2025-10-01 to 2025-12-31",
        },
        "outputs": {
            "expected_keywords": ["bearing", "conveyor", "CB-100", "template", "idler"],
            "expected_template_elements": [
                "symptom_name",
                "description",
                "root_causes",
                "affected_machines",
            ],
        },
    },
]


def seed_dataset_if_missing(name: str, examples: list):
    """Seed evaluation datasets in LangSmith if they are missing."""
    try:
        ls_client.read_dataset(dataset_name=name)
        print(f"✓ LangSmith Dataset '{name}' already exists.")
    except Exception:
        print(
            f"⚠ LangSmith Dataset '{name}' not found. Creating and seeding default examples..."
        )
        dataset = ls_client.create_dataset(
            dataset_name=name,
            description=f"Auto-generated evaluation dataset for Hephaestus Multi-Agent System: {name}",
        )
        for ex in examples:
            ls_client.create_example(
                inputs=ex["inputs"], outputs=ex["outputs"], dataset_id=dataset.id
            )
        print(f"✓ Created and seeded dataset '{name}' with {len(examples)} examples.")


# --- Execution Wrappers ---


def run_coordinator(x: dict) -> dict:
    """Wrapper to run the Coordinator node with mock SupervisorState."""
    query = x.get("query")
    if not query and "messages" in x:
        for msg in x["messages"]:
            if isinstance(msg, dict) and msg.get("type") == "human":
                query = msg.get("content")
                break
    if not query:
        query = "Hello"

    state = {
        "messages": [HumanMessage(content=query)],
        "active_agent": None,
        "coordinator_next": "",
        "coordinator_final": False,
        "troubleshooting_thread_id": None,
        "summarizer_thread_id": None,
        "machine_id": None,
        "symptom": None,
        "task": None,
        "period": None,
        "answer": "",
    }

    return coordinator_node(state)


def run_troubleshooting(x: dict) -> dict:
    """Wrapper to run the Troubleshooting node in a sandbox session."""
    query = x.get("query", "Analyze machine vibration issues")

    state = {
        "messages": [HumanMessage(content=query)],
        "active_agent": "troubleshooting",
        "coordinator_next": "troubleshooting",
        "coordinator_final": False,
        "troubleshooting_thread_id": f"eval_tb_{uuid.uuid4()}",
        "summarizer_thread_id": None,
        "machine_id": x.get("machine_id"),
        "symptom": x.get("symptom"),
        "task": x.get("task"),
        "period": None,
        "answer": "",
    }

    config = {"configurable": {"thread_id": f"eval_tb_thread_{uuid.uuid4()}"}}
    return troubleshooting_node(state, config)


def run_summarizer(x: dict) -> dict:
    """Wrapper to run the Summarizer node in a sandbox session."""
    query = x.get("query", "Summarize CM history and build template")

    state = {
        "messages": [HumanMessage(content=query)],
        "active_agent": "summarizer",
        "coordinator_next": "summarizer",
        "coordinator_final": False,
        "troubleshooting_thread_id": None,
        "summarizer_thread_id": f"eval_sum_{uuid.uuid4()}",
        "machine_id": x.get("machine_id"),
        "symptom": x.get("symptom"),
        "task": x.get("task"),
        "period": x.get("period"),
        "answer": "",
    }

    config = {"configurable": {"thread_id": f"eval_sum_thread_{uuid.uuid4()}"}}
    return summarizer_node(state, config)


# --- Custom Evaluators ---


def next_agent_evaluator(run, example) -> dict:
    """Evaluate if the coordinator routes to the correct sub-agent and final action."""
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}

    run_next = run_outputs.get("coordinator_next", "")
    example_next = example_outputs.get("next_agent", "")

    run_final = run_outputs.get("coordinator_final", False)
    example_final = example_outputs.get("coordinator_final_answer", False)

    next_agent_match = run_next == example_next
    final_answer_match = run_final == example_final

    score = int(next_agent_match and final_answer_match)

    return {
        "score": score,
        "key": "routing_accuracy",
        "comment": f"Match details: next_agent={next_agent_match} (run: {run_next}, expected: {example_next}), final_answer={final_answer_match} (run: {run_final}, expected: {example_final})",
    }


def metadata_extraction_evaluator(run, example) -> dict:
    """Evaluate if the coordinator correctly parses entities like machine_id and symptom."""
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}

    run_machine = (run_outputs.get("machine_id") or "").strip().lower()
    example_machine = (example_outputs.get("machine_id") or "").strip().lower()

    run_symptom = (run_outputs.get("symptom") or "").strip().lower()
    example_symptom = (example_outputs.get("symptom") or "").strip().lower()

    machine_match = (run_machine == example_machine) if example_machine else True
    symptom_match = (
        (example_symptom in run_symptom or run_symptom in example_symptom)
        if example_symptom
        else True
    )

    score = int(machine_match and symptom_match)

    return {
        "score": score,
        "key": "metadata_extraction_accuracy",
        "comment": f"Machine Match: {machine_match} ({run_machine} vs {example_machine}). Symptom Match: {symptom_match} ({run_symptom} vs {example_symptom}).",
    }


def keyword_overlap_evaluator(run, example) -> dict:
    """Evaluate if the agent's final answer contains key technical terms."""
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}

    answer = (run_outputs.get("answer") or "").lower()
    expected_keywords = example_outputs.get("expected_keywords", [])

    if not expected_keywords:
        return {
            "score": 1.0,
            "key": "keyword_match_rate",
            "comment": "No expected keywords defined.",
        }

    matched = [kw for kw in expected_keywords if kw.lower() in answer]
    match_rate = len(matched) / len(expected_keywords)

    return {
        "score": match_rate,
        "key": "keyword_match_rate",
        "comment": f"Matched {len(matched)} of {len(expected_keywords)} keywords: {matched}",
    }


def diagnostic_llm_evaluator(run, example) -> dict:
    """Evaluate if the diagnostic agent accurately identifies the expected root cause."""
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}

    answer = run_outputs.get("answer", "")
    expected_root_cause = example_outputs.get("expected_root_cause", "")

    if not expected_root_cause:
        return {
            "score": 1.0,
            "key": "diagnostic_correctness",
            "comment": "No expected root cause defined.",
        }

    prompt = f"""You are evaluating an AI engineering diagnostic troubleshooting agent's output.
Expected diagnosed issue / root cause to be covered: "{expected_root_cause}"

Agent's output:
"{answer}"

Determine if the agent's output correctly identifies, discusses, or addresses the expected root cause/issue.
Provide a score between 0.0 (totally missed) and 1.0 (perfectly correct).
Return your response in this JSON format:
{{"score": float, "reasoning": "brief explanation"}}
"""
    try:
        eval_llm = ChatOpenAI(
            model=config.EVALUATION_MODEL, temperature=0, api_key=config.OPENAI_API_KEY
        )
        response = eval_llm.invoke(prompt)
        res_json = json.loads(response.content)
        return {
            "score": float(res_json.get("score", 0.0)),
            "key": "diagnostic_correctness",
            "comment": res_json.get("reasoning", ""),
        }
    except Exception as e:
        # Fallback to string matching if LLM fails
        has_cause = int(expected_root_cause.lower() in answer.lower())
        return {
            "score": float(has_cause),
            "key": "diagnostic_correctness",
            "comment": f"Fallback search: Match={has_cause}. Error during LLM eval: {e}",
        }


def summarizer_completeness_evaluator(run, example) -> dict:
    """Evaluate if the summarizer successfully generates all expected case template metadata elements."""
    run_outputs = run.outputs or {}
    example_outputs = example.outputs or {}

    answer = run_outputs.get("answer", "")
    expected_elements = example_outputs.get("expected_template_elements", [])

    if not expected_elements:
        return {
            "score": 1.0,
            "key": "summary_completeness",
            "comment": "No expected template elements defined.",
        }

    prompt = f"""You are evaluating a maintenance summarization agent's compiled template report.
Expected elements to be present or structured in the report: {expected_elements}

Agent's output:
"{answer}"

Determine if the agent's output successfully provides or structures each of the expected elements.
Provide a completeness score between 0.0 (none present) and 1.0 (all beautifully presented).
Return your response in this JSON format:
{{"score": float, "reasoning": "brief explanation"}}
"""
    try:
        eval_llm = ChatOpenAI(
            model=config.EVALUATION_MODEL, temperature=0, api_key=config.OPENAI_API_KEY
        )
        response = eval_llm.invoke(prompt)
        res_json = json.loads(response.content)
        return {
            "score": float(res_json.get("score", 0.0)),
            "key": "summary_completeness",
            "comment": res_json.get("reasoning", ""),
        }
    except Exception as e:
        matched = [el for el in expected_elements if el.lower() in answer.lower()]
        return {
            "score": len(matched) / len(expected_elements),
            "key": "summary_completeness",
            "comment": f"Fallback keyword search: matched={matched}. Error during LLM eval: {e}",
        }


# --- Main Runner ---


def main():
    print("=" * 70)
    print("      HEPHAESTUS MULTI-AGENT EVALUATION RUNNER")
    print("=" * 70)

    # 1. Seed Datasets
    print("\n[STEP 1] Seeding datasets in LangSmith...")
    seed_dataset_if_missing("coordinator_eval_dataset", COORDINATOR_SEED_EXAMPLES)
    seed_dataset_if_missing(
        "troubleshooting_eval_dataset", TROUBLESHOOTING_SEED_EXAMPLES
    )
    seed_dataset_if_missing("summarizer_eval_dataset", SUMMARIZER_SEED_EXAMPLES)

    print("\n[STEP 2] Running Coordinator Agent Evaluation...")
    coordinator_results = ls_client.evaluate(
        run_coordinator,
        data="coordinator_eval_dataset",
        evaluators=[next_agent_evaluator, metadata_extraction_evaluator],
        experiment_prefix="coordinator_eval",
        max_concurrency=4,
    )
    print(f"✓ Coordinator Evaluation complete. Run ID: {coordinator_results}")

    print("\n[STEP 3] Running Troubleshooting Agent Evaluation...")
    troubleshooting_results = ls_client.evaluate(
        run_troubleshooting,
        data="troubleshooting_eval_dataset",
        evaluators=[keyword_overlap_evaluator, diagnostic_llm_evaluator],
        experiment_prefix="troubleshooting_eval",
        max_concurrency=2,
    )
    print(f"✓ Troubleshooting Evaluation complete. Run ID: {troubleshooting_results}")

    print("\n[STEP 4] Running Summarizer Agent Evaluation...")
    summarizer_results = ls_client.evaluate(
        run_summarizer,
        data="summarizer_eval_dataset",
        evaluators=[keyword_overlap_evaluator, summarizer_completeness_evaluator],
        experiment_prefix="summarizer_eval",
        max_concurrency=2,
    )
    print(f"✓ Summarizer Evaluation complete. Run ID: {summarizer_results}")

    print("\n" + "=" * 70)
    print("✓ All evaluations triggered successfully on LangSmith!")
    print("Check your LangSmith account to view the detailed dashboards.")
    print("=" * 70)


if __name__ == "__main__":
    main()
