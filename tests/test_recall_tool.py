from __future__ import annotations

from labellens.evidence.models import EVIDENCE_STATUS_FOUND, EVIDENCE_STATUS_NOT_FOUND, EVIDENCE_STATUS_UNKNOWN
from labellens.tools.recalls import GetRecallEvidenceTool


def test_recall_found_is_reported_with_identifier_reason_and_date():
    tool = GetRecallEvidenceTool()
    result = tool.execute(product_id="product_002")

    evidence = result.data["evidence"]
    assert len(evidence) == 1
    item = evidence[0]
    assert item["claim_type"] == "recall"
    assert item["status"] == EVIDENCE_STATUS_FOUND
    assert "RECALL-2022-0099" in item["value"]
    assert "2022-11-03" in item["value"]
    assert item["raw_data"]["reason"] == "undeclared tree nut traces"


def test_no_recall_found_is_distinguished_from_unavailable():
    tool = GetRecallEvidenceTool()
    result = tool.execute(product_id="product_001")  # recalls_available true, recalls empty

    evidence = result.data["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["status"] == EVIDENCE_STATUS_NOT_FOUND


def test_recall_data_unavailable_is_reported_explicitly():
    tool = GetRecallEvidenceTool()
    result = tool.execute(product_id="product_003")  # recalls_available: false

    evidence = result.data["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["status"] == EVIDENCE_STATUS_UNKNOWN
    assert evidence[0]["raw_data"] is None


def test_recall_evidence_preserves_provenance():
    tool = GetRecallEvidenceTool()
    result = tool.execute(product_id="product_002")
    item = result.data["evidence"][0]

    assert item["source_name"] == "fixture_product_database"
    assert item["source_type"] == "product_database"
    assert item["source_reference"]
    assert item["tool_name"] == "get_recall_evidence"
    assert item["retrieved_at"]
