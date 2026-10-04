"""Product model, fixture loading, and the identify_product tool."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from labellens.serialization import Serializable
from labellens.tools.base import Tool, ToolResult

# src/labellens/tools/product.py -> tools -> labellens -> src -> repo root
DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parents[3] / "data" / "fixtures" / "products.json"


@dataclass
class Product(Serializable):
    """A packaged-food product resolved from a user's query."""

    product_id: str
    name: str
    brand: Optional[str] = None
    barcode: Optional[str] = None


def load_fixture_products(path: Path = DEFAULT_FIXTURE_PATH) -> list[dict]:
    """Load the raw fixture product records (dicts, not just Product fields).

    Returns the full fixture dict per product (including ingredients /
    allergen_statement / ingredients_available) since ingredients.py needs
    those fields too; `IdentifyProductTool` only reads the Product-shaped
    subset.
    """
    with path.open(encoding="utf-8") as f:
        return json.load(f)


class IdentifyProductTool(Tool):
    """Resolve a free-text name, barcode, or product_id to a Product.

    Matches in order: exact product_id -> exact barcode -> exact name
    (case-insensitive) -> substring of name (case-insensitive). Reads from
    the local fixture file for v0; no network calls.
    """

    name = "identify_product"
    description = "Resolve a product_id, barcode, or name to a Product using fixture data."

    def __init__(self, fixture_path: Path = DEFAULT_FIXTURE_PATH) -> None:
        self.fixture_path = fixture_path

    def execute(self, product_query: str) -> ToolResult:
        records = load_fixture_products(self.fixture_path)
        query_lower = product_query.strip().lower()

        match = None
        for record in records:
            if record["product_id"] == product_query:
                match = record
                break
        if match is None:
            for record in records:
                if record.get("barcode") and record["barcode"] == product_query:
                    match = record
                    break
        if match is None:
            for record in records:
                if record["name"].lower() == query_lower:
                    match = record
                    break
        if match is None:
            for record in records:
                if query_lower in record["name"].lower():
                    match = record
                    break

        if match is None:
            return ToolResult(success=True, data={"product": None}, error=None)

        product = Product(
            product_id=match["product_id"],
            name=match["name"],
            brand=match.get("brand"),
            barcode=match.get("barcode"),
        )
        return ToolResult(success=True, data={"product": product.to_dict()})
