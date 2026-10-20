"""LLMClient: the one seam between the ReAct Actor and an actual model call.

Keeping this abstraction small and provider-agnostic means swapping which
model answers `generate_decision` never touches react_actor.py or
react_runtime.py. Nothing here executes tools or reads fixture data.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from openai import OpenAI

from labellens.agent.decision import ALLOWED_ACTIONS, ActorOutputError, AgentDecision, parse_decision

DEFAULT_MODEL = "gpt-6-luna"

SYSTEM_INSTRUCTIONS = (
    "You are the LabelLens Actor. Your job is to decide exactly one next action. "
    "Use external tools rather than guessing facts. Do not claim information that "
    "has not been observed. Evidence from one kind of tool must not be treated as "
    "proof for a different kind of claim — ingredient evidence does not prove or "
    "disprove formal certification, and neither proves or disproves a recall. A "
    "compound question may require more than one evidence tool before you have "
    "enough to finish. Normally resolve product identity with identify_product "
    "before requesting evidence, unless the product is already known. If required "
    "evidence is unavailable, abstain. Finish only when the available evidence is "
    "sufficient. "
    "Respond with exactly one JSON object: `decision_summary` is a short, "
    "high-level explanation of what information or action is needed next, "
    "suitable for research logging — it is not a chain-of-thought and must not "
    "contain private step-by-step reasoning. `action` is exactly one of: "
    f"{', '.join(sorted(ALLOWED_ACTIONS))}. `arguments` holds only the argument(s) "
    "that action needs; leave the others null."
)

# Flattened union of every action's arguments, so one strict JSON schema can
# cover all four actions. Unused fields for the chosen action are left null
# and stripped before building an AgentDecision.
_DECISION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "decision_summary": {"type": "string"},
        "action": {"type": "string", "enum": sorted(ALLOWED_ACTIONS)},
        "arguments": {
            "type": "object",
            "properties": {
                "query": {"type": ["string", "null"]},
                "product_id": {"type": ["string", "null"]},
                "answer": {"type": ["string", "null"]},
                "reason": {"type": ["string", "null"]},
            },
            "required": ["query", "product_id", "answer", "reason"],
            "additionalProperties": False,
        },
    },
    "required": ["decision_summary", "action", "arguments"],
    "additionalProperties": False,
}

_DECISION_RESPONSE_FORMAT = {
    "format": {
        "type": "json_schema",
        "name": "agent_decision",
        "schema": _DECISION_JSON_SCHEMA,
        "strict": True,
    }
}


class LLMClient(ABC):
    """Produces exactly one AgentDecision from a prompt context, or raises ActorOutputError."""

    @abstractmethod
    def generate_decision(self, prompt_context: dict) -> AgentDecision:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic, test-safe LLMClient: replays a fixed, pre-written script.

    Never makes a network call, costs nothing, and needs no API key. This
    is what every automated test and the default demo use. Each entry in
    `script` is a raw dict exactly as a real model's JSON output would
    look — including, deliberately, malformed entries in tests that need
    to exercise the error path.
    """

    def __init__(self, script: list[dict]) -> None:
        self._script = list(script)
        self._index = 0

    def generate_decision(self, prompt_context: dict) -> AgentDecision:
        if self._index >= len(self._script):
            raise ActorOutputError("mock script exhausted before reaching finish/abstain")
        raw = self._script[self._index]
        self._index += 1
        return parse_decision(raw)


class OpenAILLMClient(LLMClient):
    """Real LLMClient backed by the OpenAI Responses API (not the Assistants API).

    Configuration is entirely environment-driven:
      - OPENAI_API_KEY: required. Read directly; never hard-coded, never logged.
      - LABELLENS_LLM_MODEL: optional, defaults to DEFAULT_MODEL ("gpt-6-luna").

    Uses the Responses API's structured-output support (`text.format` with a
    strict json_schema) so the model's reply is already JSON-schema-valid by
    construction. `generate_decision` still re-validates it through the same
    `parse_decision` the mock client uses, for one shared validation path.

    Only raises ActorOutputError for problems with the *model's output*
    (a refusal, missing output text, invalid JSON, a schema-shaped-but-still
    invalid decision). Network, auth, and rate-limit errors from the OpenAI
    SDK are intentionally left to propagate as-is — those are a different
    failure class than "the actor's decision was malformed," and the
    runtime does not catch them.
    """

    def __init__(self, model: Optional[str] = None, client: Optional[OpenAI] = None) -> None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OpenAILLMClient requires the OPENAI_API_KEY environment variable to be set.")
        self.model = model or os.environ.get("LABELLENS_LLM_MODEL") or DEFAULT_MODEL
        self.client = client if client is not None else OpenAI(api_key=api_key)

    def generate_decision(self, prompt_context: dict) -> AgentDecision:
        response = self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTIONS,
            input=json.dumps(prompt_context),
            text=_DECISION_RESPONSE_FORMAT,
        )

        refusal = _extract_refusal(response)
        if refusal is not None:
            raise ActorOutputError(f"model refused to produce a decision: {refusal}")

        output_text = getattr(response, "output_text", None)
        if not output_text:
            raise ActorOutputError("model response contained no output text")

        try:
            raw = json.loads(output_text)
        except json.JSONDecodeError as exc:
            raise ActorOutputError(f"model output was not valid JSON: {exc}") from exc

        if isinstance(raw, dict) and isinstance(raw.get("arguments"), dict):
            raw["arguments"] = {key: value for key, value in raw["arguments"].items() if value is not None}

        return parse_decision(raw)


def _extract_refusal(response: object) -> Optional[str]:
    """Pull a refusal message out of a Responses API result, if the model gave one."""
    for item in getattr(response, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "refusal":
                return getattr(block, "refusal", None) or "model declined to respond"
    return None
