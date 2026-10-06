from __future__ import annotations

import pytest

from labellens.agent.decision import ActorOutputError
from labellens.agent.llm_client import MockLLMClient


def test_mock_llm_client_replays_script_in_order():
    client = MockLLMClient(
        [
            {"decision_summary": "first", "action": "identify_product", "arguments": {"query": "product_001"}},
            {"decision_summary": "second", "action": "finish", "arguments": {"answer": "done"}},
        ]
    )

    first = client.generate_decision(prompt_context={})
    assert first.action == "identify_product"

    second = client.generate_decision(prompt_context={})
    assert second.action == "finish"


def test_mock_llm_client_raises_on_malformed_script_entry():
    client = MockLLMClient([{"decision_summary": "x"}])  # missing "action" and "arguments"
    with pytest.raises(ActorOutputError):
        client.generate_decision(prompt_context={})


def test_mock_llm_client_raises_when_script_exhausted():
    client = MockLLMClient([{"decision_summary": "x", "action": "finish", "arguments": {"answer": "done"}}])
    client.generate_decision(prompt_context={})
    with pytest.raises(ActorOutputError):
        client.generate_decision(prompt_context={})
