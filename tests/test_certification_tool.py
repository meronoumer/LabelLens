from __future__ import annotations

from labellens.evidence.models import EVIDENCE_STATUS_FOUND, EVIDENCE_STATUS_NOT_FOUND, EVIDENCE_STATUS_UNKNOWN
from labellens.tools.certification import GetCertificationEvidenceTool


def test_certification_found_is_reported_explicitly():
    tool = GetCertificationEvidenceTool()
    result = tool.execute(product_id="product_001")

    assert result.success is True
    evidence = result.data["evidence"]
    assert len(evidence) == 1
    item = evidence[0]
    assert item["claim_type"] == "certification"
    assert item["status"] == EVIDENCE_STATUS_FOUND
    assert "kosher" in item["value"].lower()
    assert item["raw_data"]["certifying_body"] == "Example Kosher Council"


def test_certification_not_found_is_distinguished_from_unavailable():
    tool = GetCertificationEvidenceTool()
    result = tool.execute(product_id="product_002")

    evidence = result.data["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["status"] == EVIDENCE_STATUS_NOT_FOUND
    assert "no gluten_free certification" in evidence[0]["value"].lower()


def test_certification_unavailable_does_not_fabricate_a_negative():
    tool = GetCertificationEvidenceTool()
    result = tool.execute(product_id="product_003")

    evidence = result.data["evidence"]
    assert len(evidence) == 1
    item = evidence[0]
    assert item["status"] == EVIDENCE_STATUS_UNKNOWN
    # must not claim "not certified" / "no certification" -- only that it's unknown
    assert "not certified" not in item["value"].lower()
    assert "no " not in item["value"].lower()
    assert item["raw_data"] is None


def test_certification_evidence_preserves_provenance():
    tool = GetCertificationEvidenceTool()
    result = tool.execute(product_id="product_001")
    item = result.data["evidence"][0]

    assert item["source_name"] == "fixture_product_database"
    assert item["source_type"] == "product_database"
    assert item["source_reference"]
    assert item["tool_name"] == "get_certification_evidence"
    assert item["retrieved_at"]


def test_certification_unknown_product_is_treated_as_unavailable():
    tool = GetCertificationEvidenceTool()
    result = tool.execute(product_id="no-such-product")

    assert result.success is True
    assert result.data["evidence"][0]["status"] == EVIDENCE_STATUS_UNKNOWN
