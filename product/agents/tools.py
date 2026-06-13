"""Agent tool framework.

Provides:
- ``Tool`` — a callable unit that agents can invoke.
- ``ToolResult`` — structured result from a tool execution.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ToolResult:
    """Result from executing a tool."""

    def __init__(
        self,
        success: bool,
        output: Any = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.success = success
        self.output = output
        self.error = error
        self.metadata = metadata or {}

    def __bool__(self) -> bool:
        return self.success

    def __str__(self) -> str:
        if self.success:
            return str(self.output)
        return f"Error: {self.error}"


class Tool(ABC):
    """Abstract base class for agent tools.

    Each tool has a name, description, and an ``execute()`` method.
    """

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, **kwargs: Any) -> ToolResult:
        """Execute the tool with the given parameters.

        Args:
            **kwargs: Tool-specific parameters.

        Returns:
            ToolResult with the execution result.
        """
        ...

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
        }
