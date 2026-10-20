"""Run LabelLens v0.2's ReAct-style Actor loop against six scripted scenarios.

    python -m labellens.demo_react

Always uses a MockLLMClient — no API key, no network call, no cost. Each
scenario is a fixed script standing in for what a real model would have
decided at each turn; the runtime, tools, evidence, and trajectory
logging underneath are the real, unmodified v0.2 code. See
demo_react_live.py to run an arbitrary query through the real OpenAI
Actor instead.
"""

from __future__ import annotations

from labellens.agent.llm_client import MockLLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.react_runtime import ReActRuntime
from labellens.trajectory.models import Trajectory

# Each scenario: (query, label, script). The "action" sequence a real
# model would choose might differ (e.g. evidence tool order for the
# compound question) -- the runtime never enforces an order, only these
# demo scripts happen to pick one reasonable path.
SCENARIOS: list[tuple[str, str, list[dict]]] = [
    (
        "Does Example Protein Bar contain milk?",
        "1. allergen question -> finish (milk found)",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "product_001"},
            },
            {
                "decision_summary": "Need ingredient evidence before answering.",
                "action": "get_ingredient_evidence",
                "arguments": {"product_id": "product_001"},
            },
            {
                "decision_summary": "Ingredient evidence shows a milk-derived ingredient; evidence is sufficient.",
                "action": "finish",
                "arguments": {
                    "answer": "Yes, evidence of a milk-derived ingredient was found: 'whey protein concentrate'."
                },
            },
        ],
    ),
    (
        "Is Example Protein Bar kosher?",
        "2. certification question -> finish (kosher found)",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "product_001"},
            },
            {
                "decision_summary": "Need certification evidence before answering.",
                "action": "get_certification_evidence",
                "arguments": {"product_id": "product_001"},
            },
            {
                "decision_summary": "Certification evidence confirms kosher status; evidence is sufficient.",
                "action": "finish",
                "arguments": {"answer": "Yes, certified kosher by Example Kosher Council."},
            },
        ],
    ),
    (
        "Has Example Fruit Snack been recalled?",
        "3. recall question -> finish (recall found)",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "product_002"},
            },
            {
                "decision_summary": "Need recall evidence before answering.",
                "action": "get_recall_evidence",
                "arguments": {"product_id": "product_002"},
            },
            {
                "decision_summary": "A matching recall was found; evidence is sufficient.",
                "action": "finish",
                "arguments": {
                    "answer": "Yes, recall RECALL-2022-0099 (undeclared tree nut traces, 2022-11-03) applies."
                },
            },
        ],
    ),
    (
        "Is Example Protein Bar dairy-free and kosher?",
        "4. compound question -> two evidence tools -> finish",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "product_001"},
            },
            {
                "decision_summary": "Need ingredient evidence to assess dairy-free.",
                "action": "get_ingredient_evidence",
                "arguments": {"product_id": "product_001"},
            },
            {
                "decision_summary": "Need certification evidence to assess kosher status.",
                "action": "get_certification_evidence",
                "arguments": {"product_id": "product_001"},
            },
            {
                "decision_summary": "Both evidence types collected; sufficient to answer the compound question.",
                "action": "finish",
                "arguments": {
                    "answer": "Not dairy-free (contains whey protein concentrate), but is certified kosher."
                },
            },
        ],
    ),
    (
        "Is Example Imported Snack kosher?",
        "5. missing-certification case -> abstain",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "product_003"},
            },
            {
                "decision_summary": "Need certification evidence before answering.",
                "action": "get_certification_evidence",
                "arguments": {"product_id": "product_003"},
            },
            {
                "decision_summary": "Certification status is unknown and cannot be confirmed; abstaining.",
                "action": "abstain",
                "arguments": {"reason": "Certification records are unavailable for this product."},
            },
        ],
    ),
    (
        "Does the Acme Mystery Snack contain peanuts?",
        "6. unknown product case -> abstain",
        [
            {
                "decision_summary": "Need to identify the product before anything else.",
                "action": "identify_product",
                "arguments": {"query": "Acme Mystery Snack"},
            },
            {
                "decision_summary": "Product could not be identified; cannot proceed to evidence retrieval.",
                "action": "abstain",
                "arguments": {"reason": "Could not identify a product matching 'Acme Mystery Snack'."},
            },
        ],
    ),
]


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


def run_scenario(query: str, label: str, script: list[dict]) -> None:
    actor = ReActActor(MockLLMClient(script))
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
    for query, label, script in SCENARIOS:
        run_scenario(query, label, script)


if __name__ == "__main__":
    main()
