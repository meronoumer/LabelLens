"""LLMClient: the one seam between the ReAct Actor and an actual model call.

Keeping this abstraction small and provider-agnostic means swapping which
model answers `generate_decision` never touches react_actor.py or
react_runtime.py. Nothing here executes tools or reads fixture data.
"""

from __future__ import annotations

import json
import os
import urllib.request
from abc import ABC, abstractmethod

from labellens.agent.decision import ActorOutputError, AgentDecision, parse_decision


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


class HTTPLLMClient(LLMClient):
    """Minimal, generic JSON-over-HTTP LLMClient, configured entirely from environment variables.

    No vendor SDK, no hard-coded key, no secret committed. Expects:
      - LABELLENS_LLM_API_URL: endpoint to POST to
      - LABELLENS_LLM_API_KEY: bearer token
      - LABELLENS_LLM_MODEL: model identifier to include in the request body

    POSTs {"model": ..., "input": prompt_context} and expects a JSON
    response body that either *is* the decision object, or has it under a
    "decision" key. This is a deliberately generic placeholder contract —
    adapt it to whatever real provider is actually used. No automated test
    exercises this class; it requires a live endpoint and real credentials.
    """

    def __init__(self) -> None:
        self.api_url = os.environ.get("LABELLENS_LLM_API_URL")
        self.api_key = os.environ.get("LABELLENS_LLM_API_KEY")
        self.model = os.environ.get("LABELLENS_LLM_MODEL")
        if not self.api_url or not self.api_key or not self.model:
            raise RuntimeError(
                "HTTPLLMClient requires LABELLENS_LLM_API_URL, LABELLENS_LLM_API_KEY, "
                "and LABELLENS_LLM_MODEL to be set."
            )

    def generate_decision(self, prompt_context: dict) -> AgentDecision:
        body = json.dumps({"model": self.model, "input": prompt_context}).encode("utf-8")
        request = urllib.request.Request(
            self.api_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = json.loads(response.read().decode("utf-8"))

        raw = response_body.get("decision", response_body)
        return parse_decision(raw)
