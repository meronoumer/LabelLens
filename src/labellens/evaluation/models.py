"""EvaluationResult: a judgment over one completed Trajectory.

Defined but not automated in v0 — every run leaves this `None`. The field
types are chosen now so the shape is settled before an automatic grader or
human annotator fills them in:

- answer_correct: bool, ground truth correctness. Needs a labeled answer
  key that doesn't exist yet -> left for manual/future scoring.
- tool_selection_correct: bool, whether the tools called were the right
  ones for the question. Only meaningful once tool choice is no longer
  hardcoded by the runtime (i.e. once an LLM Actor exists).
- necessary_tool_missing: bool, a required tool was never called.
- unnecessary_tool_used: bool, a tool was called that didn't help.
- evidence_coverage: float in [0, 1], fraction of required evidence types
  that were actually collected. Numeric rather than bool since partial
  coverage is informative once multiple evidence types exist (v0.1+).
- conflict_detected: bool, whether a real evidence conflict was present
  and caught. Not meaningful in v0 (single evidence source, no conflicts
  possible yet).
- appropriate_abstention: bool, whether abstaining (or not abstaining) was
  the right call in hindsight. Requires human/offline judgment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from labellens.serialization import Serializable


@dataclass
class EvaluationResult(Serializable):
    answer_correct: Optional[bool] = None
    tool_selection_correct: Optional[bool] = None
    necessary_tool_missing: Optional[bool] = None
    unnecessary_tool_used: Optional[bool] = None
    evidence_coverage: Optional[float] = None
    conflict_detected: Optional[bool] = None
    appropriate_abstention: Optional[bool] = None
