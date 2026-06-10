# Evaluation Framework

## Overview

The AI Platform includes a comprehensive evaluation framework for measuring agent performance, response quality, and system reliability.

## Components

### Golden Datasets
**Location**: `evaluation/datasets/`
**Format**: JSON array of `{input, expected}` objects

```json
{
  "input": "Interview transcript text...",
  "expected": {
    "candidate_level": "Senior",
    "score": 8.5,
    "recommendation": "Strong Hire",
    "skills": ["Python", "FastAPI"],
    "experience_years": 8.0
  }
}
```

### DatasetRunner
**Location**: `evaluation/runners/dataset_runner.py`

Features:
- Batch execution of agent against dataset
- Multi-metric scoring (accuracy, error, overlap)
- Pass/fail assessment with configurable threshold
- JSON report generation

### Scoring Metrics

| Metric | Weight | Description |
|--------|--------|-------------|
| Level Accuracy | 30% | Correct seniority assessment |
| Score Error | 30% | Closeness of score to expected (normalized) |
| Recommendation Match | 20% | Correct hire decision |
| Skill Overlap | 20% | Jaccard similarity of skills detected |

### Report Format

Reports are JSON with:
```json
{
  "summary": {
    "dataset": "golden_dataset",
    "total": 5,
    "passed": 4,
    "failed": 1,
    "pass_rate": "80.0%",
    "metrics": {
      "avg_score": 0.85,
      "level_accuracy": "80.0%",
      "recommendation_accuracy": "80.0%",
      "avg_score_error": 0.12,
      "avg_skill_overlap": "65.0%",
      "avg_duration_ms": 5.2
    }
  },
  "results": [...]
}
```

## Running Evaluations

### CLI Evaluation
```bash
python -c "
import asyncio
from product.agents.interview_agent import InterviewAgent
from evaluation.runners.dataset_runner import DatasetRunner

agent = InterviewAgent()
runner = DatasetRunner(agent, threshold=0.7)
report = asyncio.run(runner.run_dataset('evaluation/datasets/golden_dataset.json'))
runner.save_report(report, 'evaluation/reports/report.json')
print(report.summary())
"
```

### Continuous Evaluation
- Nightly CI pipeline runs evaluation against golden datasets
- Reports are archived as CI artifacts
- Pass rate regression triggers alerts

## RAG Evaluation Metrics

The framework supports these RAG evaluation metrics:

- **Recall@k**: Fraction of relevant documents in top-k results
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of relevant documents
- **nDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality

## LLM-as-Judge

The framework supports LLM-based evaluation using:

- **CorrectnessJudge**: Evaluates factual accuracy
- **RelevanceJudge**: Evaluates response relevance
- **CompletenessJudge**: Evaluates response completeness

Each judge produces:
- Score (0.0 to 1.0)
- Reasoning (explanation of the score)
- Specific feedback for improvement

## Extending

To add a new metric:
1. Create metric in `evaluation/metrics/`
2. Implement the evaluation logic
3. Register in the `DatasetRunner`

To add a new dataset:
1. Create JSON dataset file in `evaluation/datasets/`
2. Run `DatasetRunner` against it
3. Review the generated report