"""The shared Tool interface every LabelLens tool implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from labellens.serialization import Serializable


@dataclass
class ToolResult(Serializable):
    """Uniform envelope every tool returns, regardless of which tool it is.

    `success` reflects whether the tool call itself executed correctly
    (e.g. the fixture file was read), not whether anything useful was
    found — "no match" and "no evidence available" are both success=True
    with an empty/None payload. This lets the trajectory logger store any
    tool's output the same way.
    """

    success: bool
    data: dict = field(default_factory=dict)
    error: Optional[str] = None


class Tool(ABC):
    """Common interface every LabelLens tool implements.

    Each subclass defines its own explicit, typed `execute(...)` signature
    (inputs differ per tool) but always returns a `ToolResult`, so the
    trajectory logger can log any tool call uniformly.
    """

    name: str
    description: str

    @abstractmethod
    def execute(self, *args, **kwargs) -> ToolResult:
        raise NotImplementedError
