"""The Evidence model: one piece of information obtained from a tool call."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from labellens.serialization import Serializable


EVIDENCE_STATUS_FOUND = "FOUND"
EVIDENCE_STATUS_NOT_FOUND = "NOT_FOUND"
EVIDENCE_STATUS_UNKNOWN = "UNKNOWN"


@dataclass
class Evidence(Serializable):
    """One structured fact relevant to a dietary question.

    Fields are flat (not nested under a separate provenance object) by
    design: source_name/source_type/source_reference/tool_name/retrieved_at
    already carry everything needed to trace a claim back to its source —
    see evidence/provenance.py, which reconstructs that trace from these
    fields rather than introducing a second data shape.

    `status` is the one v0.2 addition: FOUND | NOT_FOUND | UNKNOWN. It is
    only meaningful for claim_type "certification" and "recall", which are
    answers to an explicit yes/no/don't-know lookup against a registry.
    Ingredient/allergen evidence leaves it None — those are plain retrieved
    facts (an ingredient either is or isn't on the label), not the result
    of a found/not-found/unknown search.
    """

    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    claim_type: str = "ingredient"  # "ingredient" | "allergen" | "certification" | "recall"
    value: str = ""
    status: Optional[str] = None  # FOUND | NOT_FOUND | UNKNOWN (certification/recall only)
    product_id: str = ""
    source_name: str = ""
    source_type: str = ""
    source_reference: Optional[str] = None
    tool_name: str = ""
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_data: Optional[dict] = None
