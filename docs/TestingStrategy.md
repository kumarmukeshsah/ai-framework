# Testing Strategy

## Test Pyramid

The framework implements a comprehensive test pyramid:

```
          ╱  E2E  ╲          ← Few, high-value, slow
         ╱ Security ╲        ← Attack vectors, edge cases
        ╱ Integration╲       ← Provider, DB, API contract
       ╱   Contract   ╲      ← Schema validation (must pass)
      ╱     Unit       ╲     ← Many, fast, isolated
     ╱   Performance    ╲    ← Throughput, latency, benchmarks
    ╱   Evaluation      ╲   ← Golden dataset, LLM-as-Judge
```

## Unit Tests

**Location**: `tests/unit/`
**Focus**: Isolated component testing
**Coverage Targets**: 80%+ (enforced in CI via `--cov-fail-under=80`)

| Test | Description |
|------|-------------|
| `test_prompt_service.py` | Prompt loading, rendering, versioning |
| `test_chunker.py` | Chunking strategies, edge cases |
| `test_interview_agent.py` | Interview agent evaluation logic |
| `test_security_middleware.py` | Security middleware, rate limiting, injection detection |
| `test_models.py` | Pydantic model validation |
| `core/test_config.py` | Configuration loading and env overrides |
| `core/test_errors.py` | Error hierarchy and serialization |
| `core/test_di.py` | Dependency injection container |
| `agents/test_agents.py` | Agent framework (memory, tools, base) |
| `providers/test_registry.py` | Provider registration and creation |

## Contract Tests

**Location**: `tests/contract/`
**Focus**: 
- Pydantic schema validation (V1/V2/V3)
- Provider interface conformance (signature matching)
**Critical**: Build must fail if contracts change

Tests ensure:
- All required fields are validated
- Field constraints (min, max, enum) are enforced
- Optional fields have correct defaults
- Old schemas remain backward compatible
- All 6 LLM providers implement the full interface contract
- Method signatures match the base class contract

## Integration Tests

**Location**: `tests/integration/`
**Focus**: Component interaction
**Requires**: Running infrastructure (Qdrant, Ollama)

## E2E Tests

**Location**: `tests/e2e/`
**Focus**: Complete request-response flow
**Covered**:
- Health check
- Evaluate endpoint with various inputs (senior, junior, context)
- Prompt listing and retrieval
- Metrics endpoint
- Request ID headers
- Security injection detection
- Large input validation
- Missing field validation

## Smoke Tests

**Location**: `tests/e2e/` (subset via `-k` filter)
**Focus**: Pre-deployment sanity checks
**Run in PR Pipeline**: 
- Health check
- Senior candidate evaluation
- Junior candidate evaluation
- Prompt listing
**Timeout**: 30 seconds

## Security Tests

**Location**: `tests/security/`
**Focus**: Attack vector detection
**Covered**:
- 23 known prompt injection patterns
- API key/token output filtering (OpenAI, GitHub, AWS, Bearer)
- Benign input false positive prevention
- Severity scaling (1-5 matched patterns)
- Rate limiting (per-IP tracking, window expiry)

## Performance Tests

**Location**: `tests/performance/`
**Focus**: Throughput and latency benchmarks
**Covered**:
- Chunker throughput (fixed, paragraph, recursive, sentence strategies)
- Merge chunks throughput
- API endpoint response times (health, evaluate, prompts)
- Empty text edge case handling

## LLM-as-Judge

**Location**: `evaluation/judges/`
**Focus**: Quality assessment of generated outputs
**Judges**:
- **CorrectnessJudge**: Factual accuracy via keyword overlap and key fact presence
- **RelevanceJudge**: Response relevance via input keyword coverage
- **CompletenessJudge**: Output length/diversity and expected element coverage

Each judge produces:
- Score (0.0 to 1.0)
- Reasoning (explanation of the score)
- Specific feedback for improvement
- Metadata for detailed analysis

## Evaluation Metrics

**Location**: `evaluation/metrics/`
**Focus**: Retrieval quality measurement
**Metrics**:
- **Recall@k**: Fraction of relevant documents in top-k results
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant document
- **nDCG (Normalized Discounted Cumulative Gain)**: Ranking quality with position discounting

## CI/CD Integration

### PR Pipeline (.github/workflows/pr.yml)
- Code quality (ruff, black, mypy)
- **Unit tests with coverage threshold (80% minimum)**
- Contract tests with strict markers
- Integration tests
- **Smoke tests (fast pre-merge sanity)**
- Security tests
- Coverage upload as artifact

### Nightly Pipeline (.github/workflows/nightly.yml)
- Full evaluation suite (E2E tests)
- **Golden dataset evaluation with 70% pass rate threshold**
- Performance benchmarks (5+ rounds minimum)
- Security scan
- Regression warning notification

### Release Pipeline (.github/workflows/release.yml)
- Full quality gates
- **Full test suite with 80% coverage threshold (fails on Codecov error)**
- E2E tests
- **Golden dataset evaluation with 80% pass rate threshold**
- Benchmark suite (detailed columns: min, max, mean, stddev)
- Docker build and push to GHCR

## Golden Dataset

**Location**: `evaluation/datasets/golden_dataset.json`
**Size**: 12 items (expanded from 5)

### Scenarios Covered
| Scenario | Count | Description |
|----------|-------|-------------|
| Strong Hire | 3 | Senior/Lead with 8-15+ years |
| Consider | 3 | Mid-level, Junior with 0-5 years |
| Reject | 2 | Junior QA, empty/gibberish |
| Edge Cases | 4 | Empty transcript, non-English, prompt injection, gibberish |

### Tags
- `edge_case`, `empty_transcript`, `no_professional_experience`
- `strong_hire`, `experienced`, `senior`
- `non_english`, `prompt_injection`, `adversarial`, `gibberish`

## Promptfoo Multi-Track Evaluation

**Location**: `tests/prompts/`
**Orchestrator**: [Promptfoo](https://www.promptfoo.dev/) (v0.121+)
**Architecture**:

```
                    Git PR
                       │
                       ▼
              ┌─────────────────┐
              │  Promptfoo Eval  │
              └────────┬────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│  DeepEval   │ │   RAGAS     │ │   Custom    │
│  Track      │ │   Track     │ │   Track     │
├─────────────┤ ├─────────────┤ ├─────────────┤
│Hallucination│ │    MRR      │ │ JSON Schema │
│Faithfulness │ │    NDCG     │ │ Tool Trace  │
│ Toxicity    │ │   Recall    │ │ API Valid.  │
│    Bias     │ │   Context   │ │ Cost Cap    │
│             │ │   Relev.    │ │ Latency SLA │
│             │ │  Faithfuln. │ │ Biz Rules   │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────┼───────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   Quality Gate   │
              │  (Pass >= 0.8)  │
              └────────┬────────┘
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
       Deploy ✅              Reject ❌
```

### Track 1: DeepEval Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| Hallucination | Measures factual hallucination (lower=better) | ≥ 0.7 |
| Faithfulness | How well output aligns with context | ≥ 0.8 |
| Toxicity | Measures harmful content (lower=better) | ≥ 0.7 |
| Bias | Measures demographic bias (lower=better) | ≥ 0.7 |

### Track 2: RAGAS Metrics
| Metric | Description | Threshold |
|--------|-------------|-----------|
| MRR | Mean Reciprocal Rank for retrieval | ≥ 0.5 |
| NDCG | Normalized Discounted Cumulative Gain | ≥ 0.5 |
| Recall | Fraction of relevant docs retrieved | ≥ 0.5 |
| Context Precision | Precision of retrieved contexts | ≥ 0.6 |
| Context Recall | Recall of retrieved contexts | ≥ 0.6 |
| Faithfulness | Output grounded in context | ≥ 0.7 |
| Answer Relevancy | How relevant the answer is | ≥ 0.7 |

### Track 3: Custom Validators
| Validator | Source | Threshold |
|-----------|--------|-----------|
| Correctness | `CorrectnessJudge` | ≥ 0.7 |
| Relevance | `RelevanceJudge` | ≥ 0.7 |
| Completeness | `CompletenessJudge` | ≥ 0.7 |
| Hallucination | `HallucinationJudge` | ≥ 0.7 |
| Safety | `SafetyJudge` | ≥ 0.8 |
| Fairness | `FairnessJudge` | ≥ 0.8 |
| JSON Schema | Schema compliance checker | ≥ 0.8 |
| Cost Cap | Per-call cost under $0.01 | ≥ 0.8 |
| Latency SLA | Response under 10s | ≥ 0.8 |
| Tool Trace | Tool call validity | ≥ 0.8 |

### Files

| File | Purpose |
|------|---------|
| `tests/prompts/promptfooconfig.yaml` | Main orchestrator config |
| `tests/prompts/prompts/candidate_v{1,2,3}.txt` | Prompt templates (3 versions) |
| `tests/prompts/providers/deepeval_provider.py` | DeepEval metric wrapper |
| `tests/prompts/providers/ragas_provider.py` | RAGAS metric wrapper |
| `tests/prompts/providers/custom_validators_provider.py` | Custom judge wrappers |
| `tests/prompts/datasets/promptfoo_test_cases.csv` | 12 test scenarios |
| `tests/prompts/quality_gate.py` | Aggregation + pass/fail decision |
| `tests/prompts/redteam/redteam_config.yaml` | Red teaming config |

### Test Categories

| Category | Count | Scenarios |
|----------|-------|-----------|
| Senior/Strong Hire | 4 | Backend, VP Eng, DevOps, QA |
| Junior/Consider | 3 | Junior, Mid-level, Graduate |
| Edge Cases | 3 | Empty, Non-English, Gibberish |
| Adversarial | 1 | Prompt injection |
| Mixed | 1 | Customer-focused |

### Running

```bash
# Run Promptfoo evaluation
npx promptfoo eval \
  --config tests/prompts/promptfooconfig.yaml \
  --output tests/prompts/results/eval_results.json

# Run quality gate
python tests/prompts/quality_gate.py \
  --results tests/prompts/results/eval_results.json

# Run red teaming
npx promptfoo redteam \
  --config tests/prompts/redteam/redteam_config.yaml \
  --output tests/prompts/results/redteam_results.json

# View web UI
npx promptfoo view
```

### CI Integration
- **PR Pipeline** (`promptfoo-eval.yml`): Runs on PRs touching prompts
- **Quality Gate**: Enforces ≥ 0.8 overall score, posts PR comment
- **Red Teaming**: Runs on main branch only (nightly)

## Running Tests

```bash
# All tests
pytest

# With coverage (minimum 80%)
pytest --cov=product --cov-report=html --cov-fail-under=80

# Specific category
pytest tests/security/ -v

# Performance benchmarks
pytest tests/performance/ --benchmark-json=results.json

# Smoke tests
pytest tests/e2e/ -k "test_health or test_evaluate_senior"

# Parallel execution
pytest -n auto

# Failed first
pytest --ff

# Golden dataset evaluation
python -c "
import asyncio
from product.agents.interview_agent import InterviewAgent
from evaluation.runners.dataset_runner import DatasetRunner
agent = InterviewAgent()
runner = DatasetRunner(agent, threshold=0.7)
report = asyncio.run(runner.run_dataset('evaluation/datasets/golden_dataset.json'))
print(report.summary())
"

# Promptfoo multi-track evaluation
npx promptfoo eval --config tests/prompts/promptfooconfig.yaml

# Quality gate
python tests/prompts/quality_gate.py --results tests/prompts/results/eval_results.json
```

## Key Quality Gates

1. **Coverage**: `--cov-fail-under=80` enforced in PR and Release
2. **Evaluation Pass Rate**: 70% nightly, 80% release
3. **Promptfoo Quality Gate**: ≥ 0.8 overall across 3 evaluation tracks
4. **Contract Tests**: Strict markers, signature verification
5. **Security**: All 23 injection patterns must be detected
6. **Benchmarks**: Minimum 5 rounds for reliable measurements
