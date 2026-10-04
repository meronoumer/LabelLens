from __future__ import annotations

from labellens.tools.ingredients import GetIngredientEvidenceTool


def test_get_ingredient_evidence_returns_ingredient_and_allergen_items():
    tool = GetIngredientEvidenceTool()
    result = tool.execute(product_id="product_001")

    assert result.success is True
    evidence = result.data["evidence"]
    claim_types = {item["claim_type"] for item in evidence}
    assert claim_types == {"ingredient", "allergen"}

    values = [item["value"] for item in evidence]
    assert "whey protein concentrate" in values
    assert any("Milk" in v for v in values)


def test_get_ingredient_evidence_missing_data_returns_empty_list():
    tool = GetIngredientEvidenceTool()
    result = tool.execute(product_id="product_003")

    assert result.success is True
    assert result.data["evidence"] == []
