"""The get_ingredient_evidence tool."""

from __future__ import annotations

from pathlib import Path

from labellens.evidence.models import Evidence
from labellens.tools.base import Tool, ToolResult
from labellens.tools.product import DEFAULT_FIXTURE_PATH, load_fixture_products

SOURCE_NAME = "fixture_product_database"
SOURCE_TYPE = "product_database"


class GetIngredientEvidenceTool(Tool):
    """Fetch ingredient/allergen evidence for an already-identified product.

    Emits one Evidence per ingredient-list entry (claim_type="ingredient"),
    plus one more for the allergen statement if the fixture has one
    (claim_type="allergen"). Retrieval stays dumb/complete on purpose —
    deciding which evidence is *relevant* to a given question is the
    runtime's job, not this tool's.

    If the fixture marks a product's ingredient data unavailable
    (`ingredients_available: false`), this returns an empty evidence list
    rather than guessing or fabricating data.
    """

    name = "get_ingredient_evidence"
    description = "Return ingredient/allergen Evidence for a product_id using fixture data."

    def __init__(self, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self.fixture_path = fixture_path

    def execute(self, product_id: str) -> ToolResult:
        records = load_fixture_products(self.fixture_path)
        record = next((r for r in records if r["product_id"] == product_id), None)

        if record is None or not record.get("ingredients_available", False):
            return ToolResult(success=True, data={"evidence": []})

        evidence_items: list[Evidence] = []
        for ingredient in record.get("ingredients", []):
            evidence_items.append(
                Evidence(
                    claim_type="ingredient",
                    value=ingredient,
                    product_id=product_id,
                    source_name=SOURCE_NAME,
                    source_type=SOURCE_TYPE,
                    source_reference=str(self.fixture_path),
                    tool_name=self.name,
                    raw_data={"ingredients": record.get("ingredients", [])},
                )
            )

        allergen_statement = record.get("allergen_statement")
        if allergen_statement:
            evidence_items.append(
                Evidence(
                    claim_type="allergen",
                    value=allergen_statement,
                    product_id=product_id,
                    source_name=SOURCE_NAME,
                    source_type=SOURCE_TYPE,
                    source_reference=str(self.fixture_path),
                    tool_name=self.name,
                    raw_data={"allergen_statement": allergen_statement},
                )
            )

        return ToolResult(success=True, data={"evidence": [e.to_dict() for e in evidence_items]})
