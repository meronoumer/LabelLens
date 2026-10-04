"""The Evidence model: one piece of information obtained from a tool call."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from labellens.serialization import Serializable


@dataclass
class Evidence(Serializable):
    """One structured fact relevant to a dietary question.

    Fields are flat (not nested under a separate provenance object) by
    design: source_name/source_type/source_reference/tool_name/retrieved_at
    already carry everything needed to trace a claim back to its source —
    see evidence/provenance.py, which reconstructs that trace from these
    fields rather than introducing a second data shape.
    """

    evidence_id: str = field(default_factory=lambda: f"ev_{uuid.uuid4().hex[:10]}")
    claim_type: str = "ingredient"  # e.g. "ingredient" | "allergen"
    value: str = ""
    product_id: str = ""
    source_name: str = ""
    source_type: str = ""
    source_reference: Optional[str] = None
    tool_name: str = ""
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_data: Optional[dict] = None
