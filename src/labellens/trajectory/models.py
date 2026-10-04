"""TrajectoryStep and Trajectory: the immutable, serializable record of one run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from labellens.evaluation.models import EvaluationResult
from labellens.serialization import Serializable


@dataclass
class TrajectoryStep(Serializable):
    """One recorded event in a run: a tool call, or a non-tool decision.

    `state_before`/`state_after` are plain-dict snapshots (AgentState.to_dict())
    taken immediately before and after this step, so a reader can see
    exactly what changed at each point without re-deriving it.
    """

    step_id: str = field(default_factory=lambda: f"step_{uuid.uuid4().hex[:10]}")
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    action: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[dict] = None
    evidence_ids_created: list[str] = field(default_factory=list)
    state_before: Optional[dict] = None
    state_after: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class Trajectory(Serializable):
    """The full, ordered record of one agent run on one query.

    This is the unit a future AvaTaR-inspired Comparator will read: every
    tool call, every evidence id referenced, and the final decision.
    """

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    query: str = ""
    product_identifier: str = ""
    steps: list[TrajectoryStep] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    final_status: str = ""
    final_answer: Optional[str] = None
    evaluation: Optional[EvaluationResult] = None
