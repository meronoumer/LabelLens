"""Provenance tracing: final claim -> evidence -> tool call -> source -> raw data.

Not a new data model and not a graph database — v0 doesn't need either.
`Evidence` already carries everything needed to answer "where did this
claim come from?" (tool_name, source_name, source_type, source_reference,
retrieved_at, raw_data). This module just assembles that into one
human-readable trace on demand.
"""

from __future__ import annotations

from labellens.evidence.models import Evidence


def build_provenance_trace(evidence: Evidence) -> dict:
    """Return an ordered trace from claim to raw source data for one Evidence item."""
    return {
        "claim": evidence.value,
        "claim_type": evidence.claim_type,
        "evidence_id": evidence.evidence_id,
        "tool_call": evidence.tool_name,
        "source": {
            "name": evidence.source_name,
            "type": evidence.source_type,
            "reference": evidence.source_reference,
        },
        "retrieved_at": evidence.retrieved_at.isoformat(),
        "raw_data": evidence.raw_data,
    }
