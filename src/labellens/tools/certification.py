"""Placeholder interface for certification evidence. Not implemented in v0.

Defined now so the shape of a future `get_certification_evidence` tool is
visible, and so `agent/state.py`'s `required_evidence` concept generalizes
cleanly later. The runtime never calls this in v0.
"""

from __future__ import annotations

from labellens.tools.base import Tool, ToolResult


class GetCertificationEvidenceTool(Tool):
    name = "get_certification_evidence"
    description = "Not implemented in v0. Will return certification evidence (e.g. organic, kosher)."

    def execute(self, product_id: str) -> ToolResult:
        return ToolResult(success=False, data={}, error="get_certification_evidence is not implemented in v0")
