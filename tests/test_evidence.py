from __future__ import annotations

import json

from labellens.evidence.models import Evidence
from labellens.evidence.provenance import build_provenance_trace


def test_evidence_serialization_roundtrip():
    evidence = Evidence(
        claim_type="ingredient",
        value="whey protein concentrate",
        product_id="product_001",
        source_name="fixture_product_database",
        source_type="product_database",
        tool_name="get_ingredient_evidence",
        raw_data={"ingredients": ["whey protein concentrate"]},
    )
    data = evidence.to_dict()
    assert json.dumps(data)  # must be JSON-serializable
    restored = Evidence.from_dict(data)
    assert restored == evidence


def test_provenance_trace_references_the_same_evidence():
    evidence = Evidence(
        claim_type="allergen",
        value="Contains: Milk, Soy, Tree Nuts",
        product_id="product_001",
        source_name="fixture_product_database",
        source_type="product_database",
        source_reference="data/fixtures/products.json",
        tool_name="get_ingredient_evidence",
    )
    trace = build_provenance_trace(evidence)

    assert trace["evidence_id"] == evidence.evidence_id
    assert trace["claim"] == evidence.value
    assert trace["tool_call"] == evidence.tool_name
    assert trace["source"]["name"] == evidence.source_name
    assert trace["source"]["reference"] == evidence.source_reference
    assert json.dumps(trace)
