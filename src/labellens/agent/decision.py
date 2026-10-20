"""AgentDecision: the one structured thing a ReAct Actor is allowed to produce.

No hidden chain-of-thought is modeled or stored anywhere in this file —
`decision_summary` is a short, explicit, research-safe description of the
next step, not a private reasoning trace.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from labellens.serialization import Serializable

ALLOWED_ACTIONS = {
    "identify_product",
    "get_ingredient_evidence",
    "get_certification_evidence",
    "get_recall_evidence",
    "finish",
    "abstain",
}


class ActorOutputError(Exception):
    """Raised when an Actor's raw output cannot be parsed into a well-formed AgentDecision.

    This is a *shape* failure (not valid JSON / not an object / missing a
    required key) — distinct from a *semantic* validation failure (e.g. an
    unknown action name), which is the runtime's responsibility, not this
    module's. Callers must not attempt to silently repair the input.
    """


@dataclass
class AgentDecision(Serializable):
    """Exactly one next action, as reported by a ReAct Actor."""

    decision_summary: str
    action: str
    arguments: dict = field(default_factory=dict)


def parse_decision(raw: object) -> AgentDecision:
    """Validate that `raw` is shaped like a decision and build an AgentDecision.

    Only checks structure (right keys, right types) — does not check
    whether `action` is one of the allowed actions or whether the
    action-specific required arguments are present. That's semantic
    validation, done by the runtime once it knows which tools exist.
    """
    if not isinstance(raw, dict):
        raise ActorOutputError(f"actor output must be a JSON object, got {type(raw).__name__}")

    for key in ("decision_summary", "action", "arguments"):
        if key not in raw:
            raise ActorOutputError(f"actor output is missing required field '{key}'")

    decision_summary = raw["decision_summary"]
    action = raw["action"]
    arguments = raw["arguments"]

    if not isinstance(decision_summary, str) or not decision_summary.strip():
        raise ActorOutputError("'decision_summary' must be a non-empty string")
    if not isinstance(action, str) or not action.strip():
        raise ActorOutputError("'action' must be a non-empty string")
    if not isinstance(arguments, dict):
        raise ActorOutputError("'arguments' must be a JSON object")

    return AgentDecision(decision_summary=decision_summary, action=action, arguments=arguments)
