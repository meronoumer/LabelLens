from __future__ import annotations

import json

from labellens.agent.llm_client import MockLLMClient
from labellens.agent.react_actor import ReActActor
from labellens.agent.react_runtime import ReActRuntime, default_tools
from labellens.agent.state import RunStatus
from labellens.tools.base import Tool, ToolResult

QUERY = "Does this product contain milk?"


def make_runtime(script: list[dict], max_steps: int = 6, tools=None) -> ReActRuntime:
    actor = ReActActor(MockLLMClient(script))
    return ReActRuntime(actor, tools=tools, max_steps=max_steps)


# -- individual action selection -----------------------------------------


def test_actor_selects_identify_product(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {"decision_summary": "done", "action": "finish", "arguments": {"answer": "ok"}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert "identify_product" in state.tools_called
    assert state.resolved_product is not None
    assert state.resolved_product.product_id == "product_001"
    assert saved_path.exists()


def test_actor_selects_get_ingredient_evidence(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "done", "action": "finish", "arguments": {"answer": "ok"}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert "get_ingredient_evidence" in state.tools_called
    assert len(state.evidence_collected) > 0


def test_actor_chooses_finish(tmp_path):
    script = [{"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "Yes, milk found."}}]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ANSWERED
    assert state.final_answer == "Yes, milk found."
    assert state.final_decision_type == "finish"
    assert trajectory.final_status == "answered"


def test_actor_chooses_abstain(tmp_path):
    script = [{"decision_summary": "insufficient", "action": "abstain", "arguments": {"reason": "no evidence"}}]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert state.final_answer == "no evidence"
    assert state.final_decision_type == "abstain"
    assert trajectory.final_status == "abstained"


# -- validation failures --------------------------------------------------


def test_invalid_tool_name_terminates_with_error(tmp_path):
    # "get_price_evidence" names no tool that has ever existed in this
    # codebase — selecting it must fail validation, not silently run it.
    script = [
        {
            "decision_summary": "trying a nonexistent tool",
            "action": "get_price_evidence",
            "arguments": {"product_id": "product_001"},
        }
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ERROR
    assert state.final_decision_type == "error"
    error_steps = [s for s in trajectory.steps if s.error]
    assert len(error_steps) == 1
    assert "unknown action" in error_steps[0].error


def test_malformed_actor_decision_terminates_with_error(tmp_path):
    script = [{"decision_summary": "missing action and arguments"}]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ERROR
    assert state.final_decision_type == "error"
    error_steps = [s for s in trajectory.steps if s.error]
    assert len(error_steps) == 1
    assert "malformed actor decision" in error_steps[0].error


def test_missing_required_argument_terminates_with_error(tmp_path):
    script = [{"decision_summary": "forgot the answer", "action": "finish", "arguments": {}}]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ERROR
    error_steps = [s for s in trajectory.steps if s.error]
    assert "missing required argument" in error_steps[0].error


# -- tool execution failure is recoverable, not fatal ----------------------


class FailingTool(Tool):
    name = "get_ingredient_evidence"
    description = "A test double that always raises during execute()."

    def execute(self, product_id: str) -> ToolResult:
        raise RuntimeError("simulated tool crash")


def test_tool_execution_failure_is_captured_as_observation_and_loop_continues(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "give up after the failure", "action": "abstain", "arguments": {"reason": "tool failed"}},
    ]
    tools = default_tools()
    tools["get_ingredient_evidence"] = FailingTool()
    state, trajectory, saved_path = make_runtime(script, tools=tools).run(QUERY, output_dir=tmp_path)

    # the run did not crash and reached its own abstain decision afterward
    assert state.status == RunStatus.ABSTAINED
    failed_observation = state.observations[-1]
    assert failed_observation["success"] is False
    assert "simulated tool crash" in failed_observation["summary"]

    tool_call_steps = [s for s in trajectory.steps if s.tool_name == "get_ingredient_evidence"]
    assert len(tool_call_steps) == 1
    assert tool_call_steps[0].tool_output["success"] is False


# -- step limit -------------------------------------------------------------


def test_max_steps_reached_terminates_safely_without_inventing_an_answer(tmp_path):
    # scripted to call get_ingredient_evidence forever and never finish
    loop_decision = {
        "decision_summary": "still checking",
        "action": "get_ingredient_evidence",
        "arguments": {"product_id": "product_001"},
    }
    script = [loop_decision] * 10  # more than max_steps
    state, trajectory, saved_path = make_runtime(script, max_steps=2).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.MAX_STEPS_REACHED
    assert state.final_decision_type == "max_steps_reached"
    assert state.final_answer is None
    assert state.current_step == 2
    assert trajectory.steps[-1].action == "terminated: maximum step count reached"


# -- trajectory structure ---------------------------------------------------


def test_trajectory_records_decision_action_observation_in_order(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "Yes, milk found."}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    step_labels = [step.action for step in trajectory.steps]
    assert step_labels == [
        "run_started",
        "actor_decision: identify first",
        "tool_call: identify_product",
        "actor_decision: need evidence",
        "tool_call: get_ingredient_evidence",
        "actor_decision: sufficient",
        "finish",
    ]


def test_evidence_ids_are_preserved_from_observation_to_trajectory(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "Yes, milk found."}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    state_evidence_ids = {e.evidence_id for e in state.evidence_collected}
    observation_evidence_ids = set(state.observations[-1]["evidence_ids"])
    assert state_evidence_ids == observation_evidence_ids
    assert set(trajectory.evidence_ids) == state_evidence_ids
    assert len(state_evidence_ids) > 0


# -- end-to-end fixture runs --------------------------------------------------


def test_end_to_end_clear_evidence_fixture_run_answers(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "milk-derived ingredient found",
            "action": "finish",
            "arguments": {"answer": "Yes, evidence of a milk-derived ingredient was found: whey."},
        },
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ANSWERED
    assert state.resolved_product.product_id == "product_001"
    assert any("whey" in e.value for e in state.evidence_collected)
    assert saved_path.exists()


def test_end_to_end_abstention_fixture_run(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_003"}},
        {
            "decision_summary": "need evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_003"},
        },
        {
            "decision_summary": "evidence unavailable",
            "action": "abstain",
            "arguments": {"reason": "Ingredient and allergen information could not be retrieved."},
        },
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert state.evidence_collected == []
    assert state.final_answer == "Ingredient and allergen information could not be retrieved."
    assert saved_path.exists()


# -- v0.2: certification / recall tool routing -------------------------------


def test_actor_selects_get_certification_evidence(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need certification evidence",
            "action": "get_certification_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "Yes, kosher certified."}},
    ]
    state, trajectory, saved_path = make_runtime(script).run("Is this kosher?", output_dir=tmp_path)

    assert "get_certification_evidence" in state.tools_called
    assert any(e.claim_type == "certification" for e in state.evidence_collected)
    assert any(e.status == "FOUND" for e in state.evidence_collected)


def test_actor_selects_get_recall_evidence(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_002"}},
        {
            "decision_summary": "need recall evidence",
            "action": "get_recall_evidence",
            "arguments": {"product_id": "product_002"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "Yes, there is a recall."}},
    ]
    state, trajectory, saved_path = make_runtime(script).run("Has this been recalled?", output_dir=tmp_path)

    assert "get_recall_evidence" in state.tools_called
    recall_evidence = [e for e in state.evidence_collected if e.claim_type == "recall"]
    assert len(recall_evidence) == 1
    assert recall_evidence[0].status == "FOUND"


def test_certification_and_recall_tool_failures_are_recoverable_like_ingredient_failures(tmp_path):
    class FailingCertTool(Tool):
        name = "get_certification_evidence"
        description = "A test double that always raises during execute()."

        def execute(self, product_id: str) -> ToolResult:
            raise RuntimeError("certification backend unreachable")

    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need certification evidence",
            "action": "get_certification_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "give up", "action": "abstain", "arguments": {"reason": "certification tool failed"}},
    ]
    tools = default_tools()
    tools["get_certification_evidence"] = FailingCertTool()
    state, trajectory, saved_path = make_runtime(script, tools=tools).run("Is this kosher?", output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    failed_observation = state.observations[-1]
    assert failed_observation["success"] is False
    assert "certification backend unreachable" in failed_observation["summary"]


# -- v0.2: compound queries requiring more than one evidence tool ------------


def test_compound_query_collects_both_ingredient_and_certification_evidence(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need ingredient evidence for dairy-free",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "need certification evidence for kosher",
            "action": "get_certification_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "both evidence types collected; sufficient to answer",
            "action": "finish",
            "arguments": {"answer": "Not dairy-free (contains whey), but is kosher certified."},
        },
    ]
    state, trajectory, saved_path = make_runtime(script).run(
        "Is this dairy-free and kosher?", output_dir=tmp_path
    )

    assert state.status == RunStatus.ANSWERED
    claim_types = {e.claim_type for e in state.evidence_collected}
    assert "ingredient" in claim_types or "allergen" in claim_types
    assert "certification" in claim_types
    assert trajectory.tools_used == ["identify_product", "get_ingredient_evidence", "get_certification_evidence"]
    # order is whatever the Actor chose -- the runtime didn't dictate it
    tool_call_steps = [s.tool_name for s in trajectory.steps if s.tool_name]
    assert tool_call_steps == ["identify_product", "get_ingredient_evidence", "get_certification_evidence"]


def test_missing_certification_leads_to_abstention_not_a_guess(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_003"}},
        {
            "decision_summary": "need certification evidence",
            "action": "get_certification_evidence",
            "arguments": {"product_id": "product_003"},
        },
        {
            "decision_summary": "certification status is unknown; cannot confirm or deny",
            "action": "abstain",
            "arguments": {"reason": "Certification records are unavailable for this product."},
        },
    ]
    state, trajectory, saved_path = make_runtime(script).run("Is this kosher?", output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert len(state.evidence_collected) == 1
    assert state.evidence_collected[0].status == "UNKNOWN"
    assert "Certification records are unavailable for this product." in state.missing_evidence


def test_missing_product_case_actor_abstains_rather_than_guessing(tmp_path):
    script = [
        {
            "decision_summary": "identify first",
            "action": "identify_product",
            "arguments": {"query": "no-such-product-xyz"},
        },
        {
            "decision_summary": "product could not be identified; cannot proceed",
            "action": "abstain",
            "arguments": {"reason": "Could not identify a product matching the query."},
        },
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ABSTAINED
    assert state.resolved_product is None
    assert state.evidence_collected == []


def test_evidence_tool_can_be_called_without_a_prior_identify_product(tmp_path):
    # Intentional: the runtime does not gate evidence tools on product
    # identity being resolved first. The Actor decides that ordering; an
    # out-of-order call just gets whatever the tool returns for that id.
    script = [
        {
            "decision_summary": "jumping straight to ingredient evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "ok"}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.status == RunStatus.ANSWERED
    assert state.resolved_product is None  # identify_product was never called
    assert len(state.evidence_collected) > 0  # the tool still ran against the given product_id


# -- v0.2: repeated calls are allowed and logged, not silently collapsed -----


def test_repeated_identical_tool_call_is_logged_and_not_deduplicated(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "need ingredient evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "double-checking ingredient evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "ok"}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    ingredient_calls = [s for s in trajectory.steps if s.tool_name == "get_ingredient_evidence"]
    assert len(ingredient_calls) == 2  # both calls preserved, not collapsed into one
    assert state.tools_called.count("get_ingredient_evidence") == 2
    assert state.observations[2].get("repeated_call") is True
    assert "repeated" in ingredient_calls[1].action
    # duplicate evidence is preserved too, with distinct ids
    ingredient_evidence_ids = [e.evidence_id for e in state.evidence_collected]
    assert len(ingredient_evidence_ids) == len(set(ingredient_evidence_ids))  # all unique ids
    assert len(state.evidence_collected) == 10  # 5 items x 2 identical calls


# -- v0.2: structured provenance survives into AgentState/Trajectory --------


def test_provenance_fields_are_preserved_for_every_evidence_type(tmp_path):
    script = [
        {"decision_summary": "identify first", "action": "identify_product", "arguments": {"query": "product_001"}},
        {
            "decision_summary": "ingredient evidence",
            "action": "get_ingredient_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "certification evidence",
            "action": "get_certification_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {
            "decision_summary": "recall evidence",
            "action": "get_recall_evidence",
            "arguments": {"product_id": "product_001"},
        },
        {"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "ok"}},
    ]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert {e.claim_type for e in state.evidence_collected} == {"ingredient", "allergen", "certification", "recall"}
    for evidence in state.evidence_collected:
        assert evidence.source_name == "fixture_product_database"
        assert evidence.source_type == "product_database"
        assert evidence.source_reference
        assert evidence.tool_name in {"get_ingredient_evidence", "get_certification_evidence", "get_recall_evidence"}
        assert evidence.retrieved_at is not None


# -- v0.2: one canonical run_id for AgentState and Trajectory ----------------


def test_agent_state_and_trajectory_share_one_canonical_run_id(tmp_path):
    script = [{"decision_summary": "sufficient", "action": "finish", "arguments": {"answer": "ok"}}]
    state, trajectory, saved_path = make_runtime(script).run(QUERY, output_dir=tmp_path)

    assert state.run_id == trajectory.run_id
    assert state.run_id.startswith("run_")

    saved = json.loads(saved_path.read_text())
    assert saved["run_id"] == state.run_id
    # the state snapshots embedded in each step must also reflect the same id
    assert saved["steps"][0]["state_after"]["run_id"] == state.run_id
