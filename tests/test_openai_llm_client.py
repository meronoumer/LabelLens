"""Unit tests for OpenAILLMClient.

Every test here injects a fake stand-in for the OpenAI SDK client via the
`client=` constructor argument — none of them construct a real `openai.OpenAI`
instance or make a network call. No OPENAI_API_KEY is required to run these;
the fake key set via monkeypatch below is just a string the constructor
checks for presence, never sent anywhere.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from labellens.agent.decision import ActorOutputError
from labellens.agent.llm_client import DEFAULT_MODEL, SYSTEM_INSTRUCTIONS, OpenAILLMClient


class FakeResponses:
    """Stands in for `openai.OpenAI().responses`, recording every call."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


class FakeOpenAI:
    """Stands in for an `openai.OpenAI` client instance."""

    def __init__(self, response: object) -> None:
        self.responses = FakeResponses(response)


def fake_response(raw: dict | None, *, text: str | None = "__unset__", refusal: str | None = None):
    if refusal is not None:
        block = SimpleNamespace(type="refusal", refusal=refusal)
        return SimpleNamespace(output_text=None, output=[SimpleNamespace(content=[block])])
    output_text = text if text != "__unset__" else json.dumps(raw)
    return SimpleNamespace(output_text=output_text, output=[])


# -- configuration / model selection --------------------------------------


def test_default_model_used_when_env_var_is_unset(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LABELLENS_LLM_MODEL", raising=False)

    client = OpenAILLMClient(client=FakeOpenAI(fake_response({})))

    assert client.model == DEFAULT_MODEL == "gpt-6-luna"


def test_env_var_model_overrides_the_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LABELLENS_LLM_MODEL", "gpt-9-strong")

    client = OpenAILLMClient(client=FakeOpenAI(fake_response({})))

    assert client.model == "gpt-9-strong"


def test_explicit_model_argument_overrides_the_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("LABELLENS_LLM_MODEL", "gpt-9-strong")

    client = OpenAILLMClient(model="gpt-override", client=FakeOpenAI(fake_response({})))

    assert client.model == "gpt-override"


def test_missing_api_key_raises_without_touching_the_network(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAILLMClient(client=FakeOpenAI(fake_response({})))


# -- request construction ---------------------------------------------------


def test_generate_decision_sends_expected_request_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    raw = {
        "decision_summary": "Need ingredient evidence before answering.",
        "action": "get_ingredient_evidence",
        "arguments": {"query": None, "product_id": "product_001", "answer": None, "reason": None},
    }
    fake_client = FakeOpenAI(fake_response(raw))
    client = OpenAILLMClient(client=fake_client)
    prompt_context = {"query": "Does this contain milk?", "max_steps": 6, "observations": []}

    client.generate_decision(prompt_context)

    assert len(fake_client.responses.calls) == 1
    call = fake_client.responses.calls[0]
    assert call["model"] == DEFAULT_MODEL
    assert call["instructions"] == SYSTEM_INSTRUCTIONS
    assert json.loads(call["input"]) == prompt_context  # the full context, nothing hidden/added
    assert call["text"]["format"]["type"] == "json_schema"
    assert call["text"]["format"]["strict"] is True
    assert set(call["text"]["format"]["schema"]["properties"]) == {"decision_summary", "action", "arguments"}


# -- structured response parsing ---------------------------------------------


def test_generate_decision_parses_a_well_formed_response_and_strips_null_arguments(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    raw = {
        "decision_summary": "Evidence is sufficient to answer.",
        "action": "finish",
        "arguments": {"query": None, "product_id": None, "answer": "Yes, milk found.", "reason": None},
    }
    client = OpenAILLMClient(client=FakeOpenAI(fake_response(raw)))

    decision = client.generate_decision({"query": "q"})

    assert decision.decision_summary == "Evidence is sufficient to answer."
    assert decision.action == "finish"
    assert decision.arguments == {"answer": "Yes, milk found."}  # nulls stripped


def test_generate_decision_raises_actor_output_error_on_invalid_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAILLMClient(client=FakeOpenAI(fake_response(None, text="not valid json")))

    with pytest.raises(ActorOutputError):
        client.generate_decision({"query": "q"})


def test_generate_decision_raises_actor_output_error_on_empty_output_text(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAILLMClient(client=FakeOpenAI(fake_response(None, text=None)))

    with pytest.raises(ActorOutputError, match="no output text"):
        client.generate_decision({"query": "q"})


def test_generate_decision_raises_actor_output_error_on_model_refusal(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAILLMClient(client=FakeOpenAI(fake_response(None, refusal="I can't determine medical safety.")))

    with pytest.raises(ActorOutputError, match="refused"):
        client.generate_decision({"query": "q"})


def test_generate_decision_raises_actor_output_error_on_missing_required_field(monkeypatch):
    # Defense in depth: parse_decision is the single source of truth for
    # validity, even though the schema is supposed to guarantee these keys.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = OpenAILLMClient(client=FakeOpenAI(fake_response({"decision_summary": ""})))

    with pytest.raises(ActorOutputError):
        client.generate_decision({"query": "q"})
