"""ReActActor: decides the next action. Never executes tools.

The Actor's only job is: given the query, a summary of current state, the
available tool descriptions, and prior observations, return exactly one
AgentDecision. Executing that decision (running a tool, updating state,
logging, enforcing the step limit) is entirely the runtime's job — see
react_runtime.py.
"""

from __future__ import annotations

from typing import Optional

from labellens.agent.decision import AgentDecision
from labellens.agent.llm_client import LLMClient
from labellens.agent.state import AgentState

TOOL_DESCRIPTIONS: dict[str, dict] = {
    "identify_product": {
        "purpose": (
            "Resolve a product name, ID, or barcode into a Product. Usually the first action when the product "
            "is not yet known."
        ),
        "input": {"query": "string"},
        "output": "Product or no match.",
    },
    "get_ingredient_evidence": {
        "purpose": (
            "Retrieve structured ingredient and allergen evidence for a resolved product. Use this to answer "
            "ingredient/allergen questions (e.g. 'does this contain milk/peanuts?'). This evidence does not "
            "prove or disprove formal certification."
        ),
        "input": {"product_id": "string"},
        "output": "List of Evidence objects (claim_type: ingredient | allergen).",
    },
    "get_certification_evidence": {
        "purpose": (
            "Retrieve formal certification evidence (e.g. kosher, halal, certified gluten-free) for a resolved "
            "product. Use this to answer certification questions (e.g. 'is this kosher?'). Ingredient evidence "
            "alone is not proof of certification — use this tool for certification questions specifically. Each "
            "item's status is FOUND, NOT_FOUND, or UNKNOWN; NOT_FOUND and UNKNOWN are both absence of proof, "
            "not proof of absence."
        ),
        "input": {"product_id": "string"},
        "output": "List of Evidence objects (claim_type: certification, status: FOUND | NOT_FOUND | UNKNOWN).",
    },
    "get_recall_evidence": {
        "purpose": (
            "Retrieve recall/safety-alert evidence for a resolved product. Use this to answer recall questions "
            "(e.g. 'has this been recalled?'). Not implied by ingredient or certification evidence."
        ),
        "input": {"product_id": "string"},
        "output": "List of Evidence objects (claim_type: recall, status: FOUND | NOT_FOUND | UNKNOWN).",
    },
    "finish": {
        "purpose": (
            "Give a final evidence-grounded answer and stop. Only finish once the evidence actually collected "
            "is sufficient to answer the question asked — a compound question may require more than one "
            "evidence tool before this is appropriate."
        ),
        "input": {"answer": "string"},
        "output": "Ends the run with status ANSWERED.",
    },
    "abstain": {
        "purpose": (
            "Stop without answering because the required evidence is unavailable or was not found. Prefer "
            "this over guessing when the relevant evidence is missing, UNKNOWN, or NOT_FOUND and the question "
            "needs a definite answer."
        ),
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
        max_steps: Optional[int] = None,
    ) -> AgentDecision:
        remaining_steps = (max_steps - state.current_step) if max_steps is not None else None
        prompt_context = {
            "query": query,
            "state_summary": self._summarize_state(state),
            "tool_descriptions": TOOL_DESCRIPTIONS,
            "observations": observations,
            "max_steps": max_steps,
            "remaining_steps": remaining_steps,
        }
        return self.llm_client.generate_decision(prompt_context)

    @staticmethod
    def _summarize_state(state: AgentState) -> dict:
        """A compact, JSON-safe view of state for the prompt — no hidden fields."""
        evidence_by_claim_type: dict[str, int] = {}
        for item in state.evidence_collected:
            evidence_by_claim_type[item.claim_type] = evidence_by_claim_type.get(item.claim_type, 0) + 1
        return {
            "status": state.status.value,
            "resolved_product": state.resolved_product.to_dict() if state.resolved_product else None,
            "evidence_collected_count": len(state.evidence_collected),
            "evidence_by_claim_type": evidence_by_claim_type,
            "tools_called": list(state.tools_called),
            "missing_evidence": list(state.missing_evidence),
            "conflicts": list(state.conflicts),
            "current_step": state.current_step,
        }
