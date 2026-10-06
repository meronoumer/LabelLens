from __future__ import annotations

import json

import pytest

from labellens.agent.decision import ActorOutputError, AgentDecision, parse_decision


def test_parse_decision_builds_agent_decision_from_well_formed_dict():
    raw = {
        "decision_summary": "Need ingredient evidence before answering.",
        "action": "get_ingredient_evidence",
        "arguments": {"product_id": "product_001"},
    }
    decision = parse_decision(raw)
    assert isinstance(decision, AgentDecision)
    assert decision.decision_summary == raw["decision_summary"]
    assert decision.action == "get_ingredient_evidence"
    assert decision.arguments == {"product_id": "product_001"}


def test_agent_decision_serializes_cleanly():
    decision = AgentDecision(decision_summary="Evidence is sufficient.", action="finish", arguments={"answer": "x"})
    assert json.dumps(decision.to_dict())


@pytest.mark.parametrize(
    "raw",
    [
        "not a dict",
        {},
        {"decision_summary": "x"},
        {"decision_summary": "", "action": "finish", "arguments": {}},
        {"decision_summary": "x", "action": "", "arguments": {}},
        {"decision_summary": "x", "action": "finish", "arguments": "not a dict"},
        {"decision_summary": "x", "action": "finish"},  # missing "arguments" entirely
    ],
)
def test_parse_decision_rejects_malformed_input(raw):
    with pytest.raises(ActorOutputError):
        parse_decision(raw)
