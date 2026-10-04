"""Placeholder interface for recall evidence. Not implemented in v0.

Defined now so the shape of a future `get_recall_evidence` tool is
visible. The runtime never calls this in v0.
"""

from __future__ import annotations

from labellens.tools.base import Tool, ToolResult


class GetRecallEvidenceTool(Tool):
    name = "get_recall_evidence"
    description = "Not implemented in v0. Will return active recall/safety-alert evidence."

    def execute(self, product_id: str) -> ToolResult:
        return ToolResult(success=False, data={}, error="get_recall_evidence is not implemented in v0")
