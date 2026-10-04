from __future__ import annotations

import json

from labellens.tools.base import Tool, ToolResult
from labellens.tools.product import IdentifyProductTool, Product


def test_product_serialization_roundtrip():
    product = Product(product_id="product_001", name="Example Protein Bar", brand="FitFuel", barcode="0123456789012")
    data = product.to_dict()
    assert json.dumps(data)
    restored = Product.from_dict(data)
    assert restored == product


def test_identify_product_tool_satisfies_the_tool_interface():
    tool = IdentifyProductTool()
    assert isinstance(tool, Tool)
    assert tool.name == "identify_product"
    assert isinstance(tool.description, str) and tool.description

    result = tool.execute(product_query="product_001")
    assert isinstance(result, ToolResult)
    assert result.success is True


def test_identify_product_by_id_barcode_and_name():
    tool = IdentifyProductTool()

    by_id = tool.execute(product_query="product_001").data["product"]
    assert by_id["name"] == "Example Protein Bar"

    by_barcode = tool.execute(product_query="0123456789012").data["product"]
    assert by_barcode["product_id"] == "product_001"

    by_name = tool.execute(product_query="Example Protein Bar").data["product"]
    assert by_name["product_id"] == "product_001"

    by_partial_name = tool.execute(product_query="protein bar").data["product"]
    assert by_partial_name["product_id"] == "product_001"


def test_identify_product_not_found_returns_none_not_an_error():
    tool = IdentifyProductTool()
    result = tool.execute(product_query="nonexistent-product-xyz")
    assert result.success is True
    assert result.data["product"] is None
