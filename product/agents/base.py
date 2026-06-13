"""Base agent abstraction for the AI Platform.

Provides:
- ``BaseAgent`` — abstract agent with memory, tool support, and optional LLM.
- Execution tracing with telemetry instrumentation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from product.agents.memory import InMemoryMemory, Memory
from product.agents.tools import Tool
from product.providers.base import LLMProvider, Message


class BaseAgent(ABC):
    """Abstract base class for all agents.

    Agents can operate:
    - **With LLM**: Uses the provider for generation.
    - **Rule-based**: Implements ``process()`` logic directly.
    """

    def __init__(
        self,
        name: str,
        provider: LLMProvider | None = None,
        system_prompt: str = "",
        memory: Memory | None = None,
        tools: list[Tool] | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.system_prompt = system_prompt
        self.memory = memory or InMemoryMemory()
        self.tools = {t.name: t for t in (tools or [])}

        if system_prompt:
            self.memory.add_message(Message(role="system", content=system_prompt))

    @abstractmethod
    async def process(self, input_data: Any, **kwargs: Any) -> Any:
        """Process input data and return a result.

        This is the main entry point for agent execution.
        Subclasses must implement this with their specific logic.

        Args:
            input_data: The input to process.
            **kwargs: Additional context parameters.

        Returns:
            The processed result.
        """
        ...

    async def chat(self, user_message: str) -> str:
        """Chat with the agent.

        Uses the LLM provider if available, otherwise raises.

        Args:
            user_message: The user's message.

        Returns:
            The agent's response.

        Raises:
            ValueError: If no provider is configured.
        """
        if not self.provider:
            raise ValueError(f"Agent '{self.name}' has no LLM provider configured")

        self.memory.add_message(Message(role="user", content=user_message))
        response = await self.provider.generate(self.memory.get_history())
        self.memory.add_message(Message(role="assistant", content=response.content))
        return response.content

    def add_tool(self, tool: Tool) -> None:
        """Register a tool with this agent."""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self.tools.get(name)

    def clear_context(self) -> None:
        """Clear conversation history, preserving the system prompt."""
        self.memory.clear()
        if self.system_prompt:
            self.memory.add_message(Message(role="system", content=self.system_prompt))
