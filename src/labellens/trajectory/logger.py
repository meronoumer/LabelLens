"""TrajectoryLogger: accumulates TrajectorySteps for one run and saves them."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from labellens.evaluation.models import EvaluationResult
from labellens.trajectory.models import Trajectory, TrajectoryStep

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data" / "trajectories"


class TrajectoryLogger:
    """One logger per run. The runtime calls `log_step` at each point in the
    workflow, then `finalize` once a status/answer is reached, then `save`.
    """

    def __init__(self, query: str, product_identifier: str, run_id: Optional[str] = None) -> None:
        """`run_id` is optional so existing callers (e.g. v0's deterministic
        runtime) keep getting an auto-generated id with no change. Pass it
        explicitly when a canonical id already exists elsewhere (e.g.
        ReActRuntime generates one run_id and shares it with AgentState)
        so the two don't drift apart.
        """
        if run_id is not None:
            self.trajectory = Trajectory(run_id=run_id, query=query, product_identifier=product_identifier)
        else:
            self.trajectory = Trajectory(query=query, product_identifier=product_identifier)

    def log_step(
        self,
        action: str,
        *,
        tool_name: Optional[str] = None,
        tool_input: Optional[dict] = None,
        tool_output: Optional[dict] = None,
        evidence_ids_created: Optional[list[str]] = None,
        state_before: Optional[dict] = None,
        state_after: Optional[dict] = None,
        error: Optional[str] = None,
    ) -> TrajectoryStep:
        step = TrajectoryStep(
            action=action,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            evidence_ids_created=evidence_ids_created or [],
            state_before=state_before,
            state_after=state_after,
            error=error,
        )
        self.trajectory.steps.append(step)

        if tool_name and tool_name not in self.trajectory.tools_used:
            self.trajectory.tools_used.append(tool_name)
        for evidence_id in step.evidence_ids_created:
            if evidence_id not in self.trajectory.evidence_ids:
                self.trajectory.evidence_ids.append(evidence_id)

        return step

    def finalize(
        self,
        final_status: str,
        final_answer: Optional[str],
        evaluation: Optional[EvaluationResult] = None,
    ) -> Trajectory:
        self.trajectory.final_status = final_status
        self.trajectory.final_answer = final_answer
        self.trajectory.evaluation = evaluation
        return self.trajectory

    def to_dict(self) -> dict:
        return self.trajectory.to_dict()

    def save(self, output_dir: Path = DEFAULT_OUTPUT_DIR) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"{self.trajectory.run_id}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.trajectory.to_dict(), f, indent=2)
        return path
