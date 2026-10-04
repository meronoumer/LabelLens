"""AgentState: the mutable working memory for one run."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from labellens.evidence.models import Evidence
from labellens.serialization import Serializable
from labellens.tools.product import Product


class RunStatus(str, Enum):
    STARTED = "started"
    GATHERING_EVIDENCE = "gathering_evidence"
    READY_TO_ANSWER = "ready_to_answer"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    ANSWERED = "answered"
    ABSTAINED = "abstained"
    ERROR = "error"


class QuestionType(str, Enum):
    """Only one value in v0 — the only question category the runtime supports.

    More values (e.g. CERTIFICATION, RECALL) will be added once those
    tools are implemented; adding them now with no behavior behind them
    would just be dead enum values.
    """

    INGREDIENT_OR_ALLERGEN = "ingredient_or_allergen"


@dataclass
class AgentState(Serializable):
    """One run's working memory. Mutated by the runtime as the workflow proceeds.

    `trajectory_step_ids` holds step IDs only, not full step bodies — the
    full TrajectoryStep objects live once, in Trajectory/TrajectoryLogger.
    `conflicts` is always empty in v0 (single evidence source; nothing to
    conflict yet) but the field is kept for v0.1.
    """

    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    query: str = ""
    product_identifier: str = ""
    resolved_product: Optional[Product] = None
    question_type: QuestionType = QuestionType.INGREDIENT_OR_ALLERGEN
    required_evidence: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    evidence_collected: list[Evidence] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    status: RunStatus = RunStatus.STARTED
    final_answer: Optional[str] = None
    trajectory_step_ids: list[str] = field(default_factory=list)
