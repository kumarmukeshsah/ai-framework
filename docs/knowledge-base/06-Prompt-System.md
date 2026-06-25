# Prompt Management System

## Overview

The prompt system provides versioned, YAML-based prompt templates with LRU caching. Every prompt has an immutable version, enabling traceability and rollback.

## Prompt Structure

Prompts are stored as YAML files organized by version:

```
product/prompts/
├── v1/
│   └── candidate_evaluation.yaml
├── v2/
│   └── candidate_evaluation.yaml
└── v3/
    └── candidate_evaluation.yaml
```

### YAML Template Format

```yaml
# product/prompts/v3/candidate_evaluation.yaml
version: "v3"
description: "Evaluates candidate transcripts for technical roles"

system_prompt: |
  You are an expert technical interviewer evaluating candidates.
  Analyze transcripts carefully and provide structured, fair evaluations.

user_template: |
  Evaluate this candidate transcript:

  TRANSCRIPT:
  {transcript}

  CONTEXT: {context}

  Provide a JSON response with:
  - candidate_level: one of Junior, Mid, Senior, Lead
  - score: 0-10
  - recommendation: one of Strong Hire, Hire, Consider, Reject
  - skills: key technical skills identified
  - strengths: list of strengths
  - weaknesses: list of areas for improvement

output_schema:
  type: object
  properties:
    candidate_level:
      type: string
      enum: [Junior, Mid, Senior, Lead]
    score:
      type: number
      minimum: 0
      maximum: 10
```

## PromptManager

```python
class PromptManager:
    def __init__(self, prompts_dir: str = "product/prompts", max_cache: int = 128):
        self._prompts_dir = Path(prompts_dir)
        self._max_cache = max_cache
        self._cache: dict[str, PromptTemplate] = {}
        self._load_prompts()
```

### Key Methods

| Method | Description |
|--------|-------------|
| `get_prompt(name, version=None)` | Get prompt by name (latest or specific version) |
| `render_prompt(name, variables, version=None)` | Render template with variables |
| `get_latest_version(name)` | Get highest version number |
| `list_prompts()` | List all available prompts |

### Usage Examples

```python
from product.services.prompt_service import PromptManager

pm = PromptManager()

# Get latest version
template = pm.get_prompt("candidate_evaluation")
print(f"Version: {template.version}")  # "v3"

# Get specific version
template = pm.get_prompt("candidate_evaluation", version="v2")

# Render with variables
system, user, version = pm.render_prompt(
    "candidate_evaluation",
    variables={
        "transcript": "I have 5 years of Python experience...",
        "context": "Senior Python Developer",
    },
)
print(f"System: {system}")
print(f"User: {user}")
print(f"Version: {version}")  # "v3"

# List all prompts
for p in pm.list_prompts():
    print(f"{p['name']}@{p['version']}: {p['description']}")
```

## PromptTemplate Model

```python
class PromptTemplate(BaseModel):
    version: str
    description: str
    system_prompt: str
    user_template: str
    output_schema: dict[str, Any] | None = None
```

## Versioning Strategy

1. **Immutable versions**: Once created, a version is never modified
2. **Semantic versioning**: `v1`, `v2`, `v3`, etc.
3. **Backward compatible**: New versions should maintain compatibility
4. **Traceability**: Every evaluation records the prompt version used

### Version Resolution

```python
def get_prompt(self, prompt_name: str, version: str | None = None) -> PromptTemplate:
    if version:
        key = f"{prompt_name}@{version}"
        if key in self._cache:
            return self._cache[key]
        raise PromptNotFoundError(f"Prompt '{key}' not found")

    # Find latest by sorting version keys
    matching = sorted(
        (k for k in self._cache if k.startswith(f"{prompt_name}@")),
        key=self._version_sort_key,
        reverse=True,
    )
    if not matching:
        raise PromptNotFoundError(f"No prompts found for '{prompt_name}'")
    return self._cache[matching[0]]
```

## Caching

- LRU cache with configurable size (default: 128 entries)
- All prompts loaded at startup from YAML files
- No runtime I/O after initialization
- Thread-safe for concurrent access

## Adding a New Prompt

1. Create YAML file in appropriate version directory:

```yaml
# product/prompts/v1/skill_assessment.yaml
version: "v1"
description: "Assesses technical skills from transcript"

system_prompt: |
  You are a technical skill assessor.

user_template: |
  Assess skills from: {transcript}
```

2. Restart the application — prompts are auto-discovered

## Errors

| Exception | When Raised |
|-----------|-------------|
| `PromptNotFoundError` | Prompt name or version not found |
| `PromptRenderError` | Template rendering fails (missing variable, etc.) |