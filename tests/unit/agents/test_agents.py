"""Tests for agent framework components."""
from __future__ import annotations

import pytest

from product.agents.base import BaseAgent
from product.agents.tools import Tool, ToolResult
from product.agents.memory import InMemoryMemory, Memory
from product.providers.base import Message


# ── Memory Tests ──────────────────────────────────────────────────────────

class TestInMemoryMemory:
    def test_add_and_retrieve(self) -> None:
        mem = InMemoryMemory()
        msg = Message(role="user", content="hello")
        mem.add_message(msg)
        history = mem.get_history()
        assert len(history) == 1
        assert history[0].content == "hello"

    def test_max_messages(self) -> None:
        mem = InMemoryMemory(max_messages=2)
        mem.add_message(Message(role="user", content="a"))
        mem.add_message(Message(role="user", content="b"))
        mem.add_message(Message(role="user", content="c"))
        assert len(mem) == 2
        assert mem.get_history()[0].content == "b"

    def test_clear(self) -> None:
        mem = InMemoryMemory()
        mem.add_message(Message(role="user", content="test"))
        mem.clear()
        assert len(mem) == 0

    def test_len(self) -> None:
        mem = InMemoryMemory()
        assert len(mem) == 0
        mem.add_message(Message(role="user", content="x"))
        assert len(mem) == 1


# ── Tool Tests ────────────────────────────────────────────────────────────

class TestToolResult:
    def test_success(self) -> None:
        r = ToolResult(success=True, output="done")
        assert bool(r) is True
        assert str(r) == "done"

    def test_error(self) -> None:
        r = ToolResult(success=False, error="failed")
        assert bool(r) is False
        assert "failed" in str(r)

    def test_metadata(self) -> None:
        r = ToolResult(success=True, output=42, metadata={"key": "val"})
        assert r.metadata["key"] == "val"


class TestTool:
    def test_tool_abstract(self) -> None:
        with pytest.raises(TypeError):
            Tool()  # type: ignore

    def test_tool_implementation(self) -> None:
        class EchoTool(Tool):
            name = "echo"
            description = "Echoes input"

            async def execute(self, **kwargs) -> ToolResult:
                return ToolResult(success=True, output=kwargs.get("text", ""))

        tool = EchoTool()
        assert tool.name == "echo"
        assert tool.description == "Echoes input"
        assert tool.to_dict() == {"name": "echo", "description": "Echoes input"}


# ── BaseAgent Tests ──────────────────────────────────────────────────────

class TestBaseAgent:
    def test_create_agent(self) -> None:
        agent = _create_test_agent()
        assert agent.name == "TestAgent"
        assert agent.system_prompt == "You are a test agent."

    def test_agent_with_system_prompt_adds_message(self) -> None:
        agent = _create_test_agent()
        assert len(agent.memory) == 1
        assert agent.memory.get_history()[0].role == "system"

    def test_chat_raises_without_provider(self) -> None:
        agent = _create_test_agent()
        with pytest.raises(ValueError, match="has no LLM provider configured"):
            import asyncio
            asyncio.run(agent.chat("hello"))

    def test_add_tool(self) -> None:
        agent = _create_test_agent()
        tool = _create_echo_tool()
        agent.add_tool(tool)
        assert agent.get_tool("echo") is tool

    def test_clear_context(self) -> None:
        agent = _create_test_agent()
        agent.memory.add_message(Message(role="user", content="hello"))
        assert len(agent.memory) == 2
        agent.clear_context()
        assert len(agent.memory) == 1  # system prompt remains
        assert agent.memory.get_history()[0].role == "system"

    def test_clear_context_no_system_prompt(self) -> None:
        agent = _create_test_agent(system_prompt="")
        agent.memory.add_message(Message(role="user", content="hello"))
        agent.clear_context()
        assert len(agent.memory) == 0


# ── Helpers ───────────────────────────────────────────────────────────────

def _create_test_agent(system_prompt: str = "You are a test agent.") -> BaseAgent:
    class TestAgent(BaseAgent):
        async def process(self, input_data, **kwargs):
            return input_data

    return TestAgent(name="TestAgent", system_prompt=system_prompt)


def _create_echo_tool() -> Tool:
    class EchoTool(Tool):
        name = "echo"
        description = "Echoes input"

        async def execute(self, **kwargs) -> ToolResult:
            return ToolResult(success=True, output=kwargs.get("text", ""))

    return EchoTool()
