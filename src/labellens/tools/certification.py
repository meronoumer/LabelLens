"""The get_certification_evidence tool."""

from __future__ import annotations

from pathlib import Path

from labellens.evidence.models import (
    EVIDENCE_STATUS_FOUND,
    EVIDENCE_STATUS_NOT_FOUND,
    EVIDENCE_STATUS_UNKNOWN,
    Evidence,
)
from labellens.tools.base import Tool, ToolResult
from labellens.tools.product import DEFAULT_FIXTURE_PATH, load_fixture_products

SOURCE_NAME = "fixture_product_database"
SOURCE_TYPE = "product_database"


class GetCertificationEvidenceTool(Tool):
    """Fetch formal certification evidence (kosher, halal, gluten-free, ...) for a product.

    Never infers certification from ingredients — this only reports what is
    explicitly on file for formal certifications. Retrieval stays
    dumb/complete, same philosophy as get_ingredient_evidence: returns one
    Evidence per certification record on file for the product, not just the
    one the question happens to be about. Deciding relevance is the
    Actor's job, not this tool's.

    If certification records are unavailable for the product entirely
    (`certifications_available: false`), returns exactly one Evidence item
    with status UNKNOWN rather than an empty list — "no data" and "checked,
    none found" are different claims and must stay distinguishable.
    """

    name = "get_certification_evidence"
    description = (
        "Return certification Evidence (e.g. kosher, halal, gluten-free) for a product_id using fixture data. "
        "Each item's status is FOUND, NOT_FOUND, or UNKNOWN."
    )

    def __init__(self, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self.fixture_path = fixture_path

    def execute(self, product_id: str) -> ToolResult:
        records = load_fixture_products(self.fixture_path)
        record = next((r for r in records if r["product_id"] == product_id), None)

        if record is None or not record.get("certifications_available", False):
            unknown = Evidence(
                claim_type="certification",
                value="Certification records are unavailable for this product.",
                status=EVIDENCE_STATUS_UNKNOWN,
                product_id=product_id,
                source_name=SOURCE_NAME,
                source_type=SOURCE_TYPE,
                source_reference=str(self.fixture_path),
                tool_name=self.name,
                raw_data=None,
            )
            return ToolResult(success=True, data={"evidence": [unknown.to_dict()]})

        evidence_items: list[Evidence] = []
        for entry in record.get("certifications", []):
            status = entry.get("status", EVIDENCE_STATUS_UNKNOWN)
            cert_type = entry.get("type", "unknown")
            if status == EVIDENCE_STATUS_FOUND:
                body = entry.get("certifying_body", "an unspecified certifying body")
                value = f"Certified {cert_type} by {body}."
            elif status == EVIDENCE_STATUS_NOT_FOUND:
                value = f"No {cert_type} certification found on file."
            else:
                value = f"{cert_type} certification status is unknown."
            evidence_items.append(
                Evidence(
                    claim_type="certification",
                    value=value,
                    status=status,
                    product_id=product_id,
                    source_name=SOURCE_NAME,
                    source_type=SOURCE_TYPE,
                    source_reference=str(self.fixture_path),
                    tool_name=self.name,
                    raw_data=entry,
                )
            )

        return ToolResult(success=True, data={"evidence": [e.to_dict() for e in evidence_items]})
