from __future__ import annotations

from labellens.agent.decision import AgentDecision
from labellens.agent.llm_client import LLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.state import AgentState
from labellens.evidence.models import Evidence


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
        "get_certification_evidence",
        "get_recall_evidence",
        "finish",
        "abstain",
    }
    assert client.received_context["state_summary"]["status"] == "started"


def test_actor_state_summary_includes_evidence_breakdown_and_gaps():
    decision = AgentDecision(decision_summary="x", action="abstain", arguments={"reason": "no evidence"})
    client = RecordingLLMClient(decision)
    actor = ReActActor(client)
    state = AgentState(query="q")
    state.evidence_collected.append(Evidence(claim_type="ingredient", value="whey"))
    state.evidence_collected.append(Evidence(claim_type="certification", value="kosher: found", status="FOUND"))
    state.missing_evidence.append("recall evidence unavailable")

    actor.decide(query="q", state=state, observations=[])

    summary = client.received_context["state_summary"]
    assert summary["evidence_by_claim_type"] == {"ingredient": 1, "certification": 1}
    assert summary["missing_evidence"] == ["recall evidence unavailable"]
    assert summary["conflicts"] == []


def test_actor_reports_remaining_steps_when_max_steps_given():
    decision = AgentDecision(decision_summary="x", action="abstain", arguments={"reason": "no evidence"})
    client = RecordingLLMClient(decision)
    actor = ReActActor(client)
    state = AgentState(query="q")
    state.current_step = 2

    actor.decide(query="q", state=state, observations=[], max_steps=6)

    assert client.received_context["remaining_steps"] == 4


def test_actor_forwards_max_steps_into_the_prompt_context():
    decision = AgentDecision(decision_summary="x", action="abstain", arguments={"reason": "no evidence"})
    client = RecordingLLMClient(decision)
    actor = ReActActor(client)
    state = AgentState(query="q")

    actor.decide(query="q", state=state, observations=[], max_steps=6)

    assert client.received_context["max_steps"] == 6


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
