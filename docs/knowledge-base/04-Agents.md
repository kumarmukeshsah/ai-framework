# Agent Framework

## Overview

The agent framework provides a structured way to build LLM-powered agents with memory, tools, and multi-stage pipelines. Agents can operate in either **LLM mode** (using a provider) or **rule-based mode** (deterministic logic).

## BaseAgent

```python
class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(
        self,
        name: str,
        provider: LLMProvider | None = None,
        system_prompt: str = "",
        memory: Memory | None = None,
        tools: list[Tool] | None = None,
    ):
        self.name = name
        self.provider = provider
        self.system_prompt = system_prompt
        self.memory = memory or InMemoryMemory()
        self.tools = {t.name: t for t in (tools or [])}
```

### Key Methods

| Method | Description |
|--------|-------------|
| `process(input_data, **kwargs)` | Main entry point — override with agent logic |
| `chat(user_message)` | Chat using LLM provider |
| `add_tool(tool)` | Register a tool |
| `get_tool(name)` | Get a tool by name |
| `clear_context()` | Clear conversation history |

### Two Operating Modes

1. **LLM Mode**: Uses `self.provider` for generation, with memory and tools
2. **Rule-based Mode**: Implements `process()` with hardcoded logic (no LLM dependency)

## Memory System

### InMemoryMemory

Default memory implementation storing messages in-memory:

```python
class InMemoryMemory(Memory):
    def add_message(self, message: Message) -> None: ...
    def get_history(self) -> list[Message]: ...
    def clear(self) -> None: ...
    def get_recent(self, n: int) -> list[Message]: ...
```

## Tool System

```python
class Tool(BaseModel):
    name: str
    description: str
    func: Callable[..., Any]
    parameters: dict[str, Any] = Field(default_factory=dict)
```

Tools allow agents to perform actions beyond text generation:

```python
# Create a tool
def search_database(query: str) -> list[dict]:
    # Execute search
    return results

tool = Tool(
    name="search",
    description="Search the database",
    func=search_database,
    parameters={"query": {"type": "string"}}
)

# Register with agent
agent.add_tool(tool)
```

## EvaluatorAgent (Reference Implementation)

The `EvaluatorAgent` is a multi-stage agent for candidate evaluation:

### Pipeline Stages

```
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Parser   │───▶│ Evaluate │───▶│  Report  │
│  Stage 1  │    │  Stage 2 │    │  Stage 3 │
└──────────┘    └──────────┘    └──────────┘
```

### Stage 1: Parse

Extracts structured information from transcript:
- Skill keywords detection (30+ patterns)
- Years of experience extraction
- Seniority level hints

### Stage 2: Evaluate

Scores against rubric (LLM or rule-based):
```python
rubric = RubricBreakdown(
    technical_depth=0-3,
    problem_solving=0-3,
    communication=0-2,
    experience_relevance=0-2,
)
```

### Stage 3: Report

Generates final `CandidateEvaluation`:
```python
class CandidateEvaluation(BaseModel):
    candidate_level: str         # Junior / Mid / Senior / Lead
    score: float                 # 0-10
    recommendation: str          # Strong Hire / Hire / Consider / Reject
    skills: list[str]
    experience_years: float | None
    rubric: RubricBreakdown
    feedback: str
    strengths: list[str]
    weaknesses: list[str]
    chain_of_thought: str | None
```

### Score Calculation

```python
RECOMMENDATION_THRESHOLDS = [
    (7.5, "Strong Hire", "Strong candidate..."),
    (5.0, "Consider", "Candidate has potential..."),
    (0.0, "Reject", "Does not meet minimum requirements."),
]
```

## Creating a Custom Agent

```python
from product.agents.base import BaseAgent

class MyCustomAgent(BaseAgent):
    def __init__(self, **kwargs):
        super().__init__(
            name="MyCustomAgent",
            system_prompt="You are a helpful assistant that...",
            **kwargs
        )

    async def process(self, input_data: str, **kwargs) -> MyResult:
        # Stage 1: Parse input
        parsed = self._parse(input_data)

        # Stage 2: Evaluate (LLM or rule-based)
        if self._use_llm and self.provider:
            result = await self._llm_evaluate(parsed)
        else:
            result = self._rule_evaluate(parsed)

        # Stage 3: Build result
        return MyResult(
            success=True,
            data=result,
            duration_ms=...,  # track manually
        )

    def _parse(self, text: str) -> dict:
        # Implement parsing logic
        ...

    def _rule_evaluate(self, parsed: dict) -> dict:
        # Implement rule-based evaluation
        ...
```

## Telemetry & Tracing

Agents are automatically instrumented via decorators:

```python
@track_agent_execution(agent_name="EvaluatorAgent")
async def process(self, transcript: str, context: str | None = None):
    ...

# Stage-level tracing
with span("evaluator.parse"):
    parse_result = await self._run_parse_stage(...)
```

## Best Practices

1. **Dual-mode operation**: Implement both LLM and rule-based paths
2. **Stage isolation**: Each stage should be independently testable
3. **Error recovery**: Return error results rather than raising exceptions
4. **Duration tracking**: Measure and report stage durations
5. **Structured output**: Use Pydantic models for all results