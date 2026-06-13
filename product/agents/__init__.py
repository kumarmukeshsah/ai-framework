"""Agent framework for the AI Platform.

Provides:
- ``BaseAgent`` — abstract agent with tool-calling and memory.
- ``Tool`` / ``ToolResult`` — framework for agent tools.
- ``Memory`` — conversation history backends.
- ``EvaluatorAgent`` — multi-stage candidate evaluation pipeline.
"""

from product.agents.base import BaseAgent
from product.agents.evaluator import EvaluationStage, EvaluatorAgent
from product.agents.memory import InMemoryMemory, Memory
from product.agents.tools import Tool, ToolResult

__all__ = [
    "BaseAgent",
    "Tool",
    "ToolResult",
    "Memory",
    "InMemoryMemory",
    "EvaluatorAgent",
    "EvaluationStage",
]
