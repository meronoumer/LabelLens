from __future__ import annotations

import json

from labellens.trajectory.logger import TrajectoryLogger


def test_steps_are_recorded_in_order():
    logger = TrajectoryLogger(query="Does this product contain milk?", product_identifier="product_001")
    logger.log_step("first")
    logger.log_step("second")
    logger.log_step("third")

    actions = [step.action for step in logger.trajectory.steps]
    assert actions == ["first", "second", "third"]


def test_log_step_tracks_tools_used_and_evidence_ids():
    logger = TrajectoryLogger(query="q", product_identifier="product_001")
    logger.log_step("called a tool", tool_name="identify_product")
    logger.log_step("called another tool", tool_name="get_ingredient_evidence", evidence_ids_created=["ev_1", "ev_2"])

    assert logger.trajectory.tools_used == ["identify_product", "get_ingredient_evidence"]
    assert logger.trajectory.evidence_ids == ["ev_1", "ev_2"]


def test_trajectory_json_export_roundtrip(tmp_path):
    logger = TrajectoryLogger(query="Does this product contain milk?", product_identifier="product_001")
    logger.log_step("run_started")
    logger.finalize(final_status="answered", final_answer="Yes, milk evidence found.")

    saved_path = logger.save(output_dir=tmp_path)
    assert saved_path.exists()

    with saved_path.open(encoding="utf-8") as f:
        loaded = json.load(f)

    assert loaded["run_id"] == logger.trajectory.run_id
    assert loaded["final_status"] == "answered"
    assert loaded["final_answer"] == "Yes, milk evidence found."
    assert loaded["steps"][0]["action"] == "run_started"
