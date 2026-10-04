from __future__ import annotations

from labellens.agent.runtime import run_query
from labellens.agent.state import RunStatus

QUERY = "Does this product contain milk?"


def test_clear_milk_evidence_workflow_answers(tmp_path):
    state, trajectory, saved_path = run_query(QUERY, "product_001", output_dir=tmp_path)

    assert state.status == RunStatus.ANSWERED
    assert "milk-derived ingredient was found" in state.final_answer
    assert any("whey" in e.value for e in state.evidence_collected)
    assert saved_path.exists()
    assert trajectory.final_status == "answered"
    assert "identify_product" in trajectory.tools_used
    assert "get_ingredient_evidence" in trajectory.tools_used


def test_no_milk_evidence_workflow_answers_with_hedge(tmp_path):
    state, trajectory, saved_path = run_query(QUERY, "product_002", output_dir=tmp_path)

    assert state.status == RunStatus.ANSWERED
    assert "no milk-derived ingredients were found" in state.final_answer
    assert "not a guarantee" in state.final_answer
    assert saved_path.exists()


def test_missing_evidence_workflow_abstains(tmp_path):
    state, trajectory, saved_path = run_query(QUERY, "product_003", output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert state.final_answer == "I do not have enough ingredient or allergen evidence to determine this."
    assert "ingredient_or_allergen_evidence" in state.missing_evidence
    assert trajectory.final_status == "abstained"
    assert saved_path.exists()


def test_unidentifiable_product_abstains(tmp_path):
    state, trajectory, saved_path = run_query(QUERY, "no-such-product", output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert state.resolved_product is None
    assert "product_identification" in state.missing_evidence
    assert saved_path.exists()
