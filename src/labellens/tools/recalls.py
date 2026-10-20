"""The get_recall_evidence tool."""

from __future__ import annotations

from pathlib import Path

from labellens.evidence.models import EVIDENCE_STATUS_FOUND, EVIDENCE_STATUS_NOT_FOUND, EVIDENCE_STATUS_UNKNOWN, Evidence
from labellens.tools.base import Tool, ToolResult
from labellens.tools.product import DEFAULT_FIXTURE_PATH, load_fixture_products

SOURCE_NAME = "fixture_product_database"
SOURCE_TYPE = "product_database"


class GetRecallEvidenceTool(Tool):
    """Fetch recall/safety-alert evidence for a product.

    Retrieval is dumb/complete: returns one Evidence per recall record on
    file. If the dataset has no recalls for a product but recall tracking
    is available, returns exactly one Evidence item with status NOT_FOUND
    (checked, nothing there) — never an empty list, since an empty list
    can't be told apart from "didn't check." If recall tracking itself is
    unavailable for the product, returns one Evidence item with status
    UNKNOWN.
    """

    name = "get_recall_evidence"
    description = (
        "Return recall Evidence for a product_id using fixture data. "
        "Each item's status is FOUND, NOT_FOUND, or UNKNOWN."
    )

    def __init__(self, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self.fixture_path = fixture_path

    def execute(self, product_id: str) -> ToolResult:
        records = load_fixture_products(self.fixture_path)
        record = next((r for r in records if r["product_id"] == product_id), None)

        if record is None or not record.get("recalls_available", False):
            unknown = Evidence(
                claim_type="recall",
                value="Recall records are unavailable for this product.",
                status=EVIDENCE_STATUS_UNKNOWN,
                product_id=product_id,
                source_name=SOURCE_NAME,
                source_type=SOURCE_TYPE,
                source_reference=str(self.fixture_path),
                tool_name=self.name,
                raw_data=None,
            )
            return ToolResult(success=True, data={"evidence": [unknown.to_dict()]})

        recalls = record.get("recalls", [])
        if not recalls:
            not_found = Evidence(
                claim_type="recall",
                value="No recall found for this product in the available dataset.",
                status=EVIDENCE_STATUS_NOT_FOUND,
                product_id=product_id,
                source_name=SOURCE_NAME,
                source_type=SOURCE_TYPE,
                source_reference=str(self.fixture_path),
                tool_name=self.name,
                raw_data=None,
            )
            return ToolResult(success=True, data={"evidence": [not_found.to_dict()]})

        evidence_items: list[Evidence] = []
        for entry in recalls:
            date = entry.get("date")
            date_part = f" ({date})" if date else ""
            value = f"Recall {entry.get('recall_id', 'unknown')}: {entry.get('reason', 'no reason on file')}{date_part}."
            evidence_items.append(
                Evidence(
                    claim_type="recall",
                    value=value,
                    status=EVIDENCE_STATUS_FOUND,
                    product_id=product_id,
                    source_name=SOURCE_NAME,
                    source_type=SOURCE_TYPE,
                    source_reference=str(self.fixture_path),
                    tool_name=self.name,
                    raw_data=entry,
                )
            )

        return ToolResult(success=True, data={"evidence": [e.to_dict() for e in evidence_items]})
