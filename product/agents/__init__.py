"""Agent framework for the AI Platform.

Provides:
- ``BaseAgent`` — abstract agent with tool-calling and memory.
- ``Tool`` / ``ToolResult`` — framework for agent tools.
- ``Memory`` — conversation history backends.
- ``EvaluatorAgent`` — multi-stage candidate evaluation pipeline.
"""
from product.agents.base import BaseAgent
from product.agents.tools import Tool, ToolResult
from product.agents.memory import Memory, InMemoryMemory
from product.agents.evaluator import EvaluatorAgent, EvaluationStage

__all__ = [
    "BaseAgent",
    "Tool",
    "ToolResult",
    "Memory",
    "InMemoryMemory",
    "EvaluatorAgent",
    "EvaluationStage",
]