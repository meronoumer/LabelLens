"""Run LabelLens v0.1's ReAct-style Actor loop against three scripted scenarios.

    python -m labellens.demo_react

Uses a MockLLMClient by default — no API key, no network call, no cost.
If LABELLENS_LLM_API_URL (and the matching API key / model env vars) are
set, this switches to the real HTTPLLMClient instead, so the same control
loop can be inspected against a real model once one is configured.
"""

from __future__ import annotations

import os

from labellens.agent.llm_client import HTTPLLMClient, LLMClient, MockLLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.react_runtime import ReActRuntime
from labellens.trajectory.models import Trajectory

QUERY_TEMPLATE = "Does {product} contain milk?"

# Each script is a fixed sequence of raw decision dicts, standing in for
# what a real model would have returned at each turn. These mirror the
# three v0 fixture outcomes: clear evidence, no evidence found, and
# evidence unavailable.
SCENARIOS: list[tuple[str, str]] = [
    ("product_001", "clear milk evidence -> finish"),
    ("product_002", "no milk evidence -> finish (hedged)"),
    ("product_003", "no ingredient data -> abstain"),
]


def build_mock_client(product_id: str) -> MockLLMClient:
    if product_id == "product_001":
        script = [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": product_id},
            },
            {
                "decision_summary": "Need ingredient evidence before answering.",
                "action": "get_ingredient_evidence",
                "arguments": {"product_id": product_id},
            },
            {
                "decision_summary": "Ingredient evidence shows a milk-derived ingredient; evidence is sufficient.",
                "action": "finish",
                "arguments": {
                    "answer": "Yes, evidence of a milk-derived ingredient was found: 'whey protein concentrate'."
                },
            },
        ]
    elif product_id == "product_002":
        script = [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": product_id},
            },
            {
                "decision_summary": "Need ingredient evidence before answering.",
                "action": "get_ingredient_evidence",
                "arguments": {"product_id": product_id},
            },
            {
                "decision_summary": "No milk-derived ingredients found; answering with available evidence.",
                "action": "finish",
                "arguments": {
                    "answer": "Based on the available evidence, no milk-derived ingredients were found. "
                    "This is not a guarantee the product is free of milk."
                },
            },
        ]
    else:
        script = [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": product_id},
            },
            {
                "decision_summary": "Need ingredient evidence before answering.",
                "action": "get_ingredient_evidence",
                "arguments": {"product_id": product_id},
            },
            {
                "decision_summary": "Required evidence is unavailable.",
                "action": "abstain",
                "arguments": {"reason": "Ingredient and allergen information could not be retrieved."},
            },
        ]
    return MockLLMClient(script)


def build_llm_client(product_id: str) -> LLMClient:
    required_env = ("LABELLENS_LLM_API_URL", "LABELLENS_LLM_API_KEY", "LABELLENS_LLM_MODEL")
    if all(os.environ.get(var) for var in required_env):
        return HTTPLLMClient()
    return build_mock_client(product_id)


def print_trajectory(trajectory: Trajectory) -> None:
    step_number = 0
    for step in trajectory.steps:
        if step.action.startswith("actor_decision:"):
            step_number += 1
            decision_summary = step.action.split(":", 1)[1].strip()
            action = step.tool_input["action"]
            arguments = step.tool_input["arguments"]
            print(f"STEP {step_number}")
            print(f"  Decision: {decision_summary}")
            print(f"  Action: {action}({arguments})")
        elif step.action.startswith("tool_call:"):
            print(f"  Observation: {step.tool_output}")
        elif step.action in ("finish", "abstain"):
            print(f"  Observation: {step.action} -> {trajectory.final_answer}")
        elif step.action.startswith("terminated:"):
            print(f"  Observation: {step.action}")


def run_scenario(product_id: str, label: str) -> None:
    query = QUERY_TEMPLATE.format(product=product_id)
    llm_client = build_llm_client(product_id)
    actor = ReActActor(llm_client)
    runtime = ReActRuntime(actor)

    print("QUERY")
    print(f"  {query}  ({label})")
    state, trajectory, saved_path = runtime.run(query)
    print_trajectory(trajectory)
    print("FINAL STATUS")
    print(f"  {state.status.value.upper()}")
    print("TRAJECTORY SAVED")
    print(f"  {saved_path}")
    print("-" * 60)


def main() -> None:
    for product_id, label in SCENARIOS:
        run_scenario(product_id, label)


if __name__ == "__main__":
    main()
