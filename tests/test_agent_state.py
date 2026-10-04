from __future__ import annotations

from labellens.agent.state import AgentState, QuestionType, RunStatus


def test_agent_state_creation_defaults():
    state = AgentState(query="Does this product contain milk?", product_identifier="product_001")

    assert state.run_id.startswith("run_")
    assert state.status == RunStatus.STARTED
    assert state.question_type == QuestionType.INGREDIENT_OR_ALLERGEN
    assert state.resolved_product is None
    assert state.evidence_collected == []
    assert state.conflicts == []
    assert state.missing_evidence == []
    assert state.final_answer is None


def test_agent_state_serializes_cleanly():
    state = AgentState(query="Does this product contain milk?", product_identifier="product_001")
    data = state.to_dict()
    assert data["status"] == "started"
    assert data["question_type"] == "ingredient_or_allergen"
