"""ReActActor: decides the next action. Never executes tools.

The Actor's only job is: given the query, a summary of current state, the
available tool descriptions, and prior observations, return exactly one
AgentDecision. Executing that decision (running a tool, updating state,
logging, enforcing the step limit) is entirely the runtime's job — see
react_runtime.py.
"""

from __future__ import annotations

from labellens.agent.decision import AgentDecision
from labellens.agent.llm_client import LLMClient
from labellens.agent.state import AgentState

TOOL_DESCRIPTIONS: dict[str, dict] = {
    "identify_product": {
        "purpose": "Resolve a product name, ID, or barcode into a Product.",
        "input": {"query": "string"},
        "output": "Product or no match.",
    },
    "get_ingredient_evidence": {
        "purpose": "Retrieve structured ingredient and allergen evidence for a resolved product.",
        "input": {"product_id": "string"},
        "output": "List of Evidence objects.",
    },
    "finish": {
        "purpose": "Give a final evidence-grounded answer and stop.",
        "input": {"answer": "string"},
        "output": "Ends the run with status ANSWERED.",
    },
    "abstain": {
        "purpose": "Stop without answering because the available evidence is insufficient.",
        "input": {"reason": "string"},
        "output": "Ends the run with status ABSTAINED.",
    },
}


class ReActActor:
    """Wraps an LLMClient and turns (query, state, observations) into one AgentDecision."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def decide(
        self,
        *,
        query: str,
        state: AgentState,
        observations: list[dict],
    ) -> AgentDecision:
        prompt_context = {
            "query": query,
            "state_summary": self._summarize_state(state),
            "tool_descriptions": TOOL_DESCRIPTIONS,
            "observations": observations,
        }
        return self.llm_client.generate_decision(prompt_context)

    @staticmethod
    def _summarize_state(state: AgentState) -> dict:
        """A compact, JSON-safe view of state for the prompt — no hidden fields."""
        return {
            "status": state.status.value,
            "resolved_product": state.resolved_product.to_dict() if state.resolved_product else None,
            "evidence_collected_count": len(state.evidence_collected),
            "tools_called": list(state.tools_called),
            "current_step": state.current_step,
        }
