"""Run one live query through the real OpenAI-backed ReAct Actor.

    python -m labellens.demo_react_live
    python -m labellens.demo_react_live "Does Example Protein Bar contain milk?"

Requires OPENAI_API_KEY. This makes real, billed calls to the OpenAI API —
it is a manual demo only and is never run by the automated test suite
(which uses MockLLMClient exclusively; see demo_react.py).
"""

from __future__ import annotations

import os
import sys

from labellens.agent.llm_client import DEFAULT_MODEL, OpenAILLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.react_runtime import ReActRuntime
from labellens.trajectory.models import Trajectory

DEFAULT_QUERY = "Does Example Protein Bar contain milk?"


def print_trajectory(trajectory: Trajectory) -> None:
    for step in trajectory.steps:
        if step.action.startswith("actor_decision:"):
            decision_summary = step.action.split(":", 1)[1].strip()
            print(f"Decision: {decision_summary}")
            print(f"Action: {step.tool_input['action']}")
            print(f"Arguments: {step.tool_input['arguments']}")
        elif step.action.startswith("tool_call:"):
            print(f"Observation: {step.tool_output}")
            print()
        elif step.action in ("finish", "abstain"):
            print(f"Observation: {step.action} -> {trajectory.final_answer}")
            print()
        elif step.action.startswith("terminated:"):
            print(f"Observation: {step.action}")
            print()


def read_query() -> str:
    if len(sys.argv) > 1:
        return " ".join(sys.argv[1:])
    typed = input(f"Query [{DEFAULT_QUERY}]: ").strip()
    return typed or DEFAULT_QUERY


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. This demo needs a real OpenAI API key to run.")
        print("Set it, optionally choose a model, then re-run:")
        print('  export OPENAI_API_KEY="sk-..."')
        print(f'  export LABELLENS_LLM_MODEL="{DEFAULT_MODEL}"   # optional, this is already the default')
        print("  python -m labellens.demo_react_live")
        sys.exit(1)

    query = read_query()
    actor = ReActActor(OpenAILLMClient())
    runtime = ReActRuntime(actor)

    print("QUERY")
    print(f"  {query}")
    print()

    state, trajectory, saved_path = runtime.run(query)

    print_trajectory(trajectory)
    print("FINAL STATUS")
    print(f"  {state.status.value.upper()}")
    print("FINAL ANSWER")
    print(f"  {state.final_answer}")
    print("TRAJECTORY SAVED")
    print(f"  {saved_path}")


if __name__ == "__main__":
    main()
