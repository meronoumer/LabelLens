from __future__ import annotations

from labellens.agent.decision import AgentDecision
from labellens.agent.llm_client import LLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.state import AgentState


class RecordingLLMClient(LLMClient):
    """Captures the prompt_context it was called with, for assertions."""

    def __init__(self, decision: AgentDecision) -> None:
        self.decision = decision
        self.received_context: dict | None = None

    def generate_decision(self, prompt_context: dict) -> AgentDecision:
        self.received_context = prompt_context
        return self.decision


def test_actor_passes_query_state_and_observations_to_the_client_and_returns_its_decision():
    decision = AgentDecision(
        decision_summary="Need to identify the product before anything else.",
        action="identify_product",
        arguments={"query": "product_001"},
    )
    client = RecordingLLMClient(decision)
    actor = ReActActor(client)
    state = AgentState(query="Does this contain milk?")
    observations = [{"tool_name": "identify_product", "success": True}]

    result = actor.decide(query=state.query, state=state, observations=observations)

    assert result is decision
    assert client.received_context["query"] == state.query
    assert client.received_context["observations"] == observations
    assert "tool_descriptions" in client.received_context
    assert set(client.received_context["tool_descriptions"].keys()) == {
        "identify_product",
        "get_ingredient_evidence",
        "finish",
        "abstain",
    }
    assert client.received_context["state_summary"]["status"] == "started"


def test_actor_never_touches_a_real_tool_or_fixture_data():
    # If the actor executed tools itself, this decision (asking for a
    # nonexistent product) would have to go fetch fixture data. It
    # doesn't — the actor only relays whatever the client returns.
    decision = AgentDecision(
        decision_summary="x", action="identify_product", arguments={"query": "no-such-product"}
    )
    client = RecordingLLMClient(decision)
    actor = ReActActor(client)
    state = AgentState(query="q")

    result = actor.decide(query="q", state=state, observations=[])

    assert result.action == "identify_product"
    assert state.resolved_product is None  # untouched by the actor
    assert state.tools_called == []
