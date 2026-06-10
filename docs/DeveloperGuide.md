# Developer Guide

## Getting Started

### Prerequisites
- Python 3.12+
- Docker & Docker Compose (for local infrastructure)
- Ollama (for local LLM inference)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd ai-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install pre-commit hooks
pre-commit install
```

### Configuration

Copy and edit the environment file:

```bash
cp .env.example .env
```

Key configuration options:

```bash
# Provider selection
LLM__PROVIDER=ollama
LLM__MODEL_NAME=llama3.1

# Vector DB
VECTOR_DB__URL=http://localhost:6333

# Environment
ENV=development
```

## Running the Framework

### Development Server

```bash
# Start with hot reload
uvicorn product.api.app:app --reload --port 8000

# Or via module
python -m product.api.app
```

### Docker Deployment

```bash
# Start all services
docker compose -f infra/docker/docker-compose.yml up -d

# View logs
docker compose -f infra/docker/docker-compose.yml logs -f api
```

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Evaluate Candidate
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "transcript": "I have 8 years of Python experience...",
    "use_llm": false
  }'
```

### Chat with Agent
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Evaluate this candidate..."}'
```

### Index Document for RAG
```bash
curl -X POST http://localhost:8000/index \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc-1",
    "content": "...document content...",
    "metadata": {"source": "interview_notes"}
  }'
```

### Prometheus Metrics
```bash
curl http://localhost:8001/metrics
```

## Testing

### Test Pyramid

```bash
# Unit tests
pytest tests/unit/ -v

# Contract tests (must pass build)
pytest tests/contract/ -v

# Security tests
pytest tests/security/ -v

# E2E tests
pytest tests/e2e/ -v

# All tests with coverage
pytest tests/ -v --cov=product --cov-report=term-missing
```

### Evaluation

```bash
# Run golden dataset evaluation
python -c "
import asyncio
from product.agents.interview_agent import InterviewAgent
from evaluation.runners.dataset_runner import DatasetRunner

agent = InterviewAgent()
runner = DatasetRunner(agent)
report = asyncio.run(runner.run_dataset('evaluation/datasets/golden_dataset.json'))
runner.save_report(report, 'evaluation/reports/report.json')
print(report.summary())
"
```

## Adding a New Provider

1. Create `product/providers/your_provider.py`
2. Implement `LLMProvider` interface:

```python
class YourProvider(LLMProvider):
    provider_name = "your_provider"

    async def generate(self, messages, temperature, max_tokens, stop_sequences):
        ...
    
    async def structured_generate(self, messages, response_model, temperature, max_tokens):
        ...
    
    async def embeddings(self, texts, model):
        ...
    
    # ... implement all abstract methods
```

3. Register in `product/providers/factory.py`
4. Add to `config.py` provider types

## Adding a New Prompt Version

1. Create `product/prompts/v4/your_prompt.yaml`
2. Define version, system_prompt, user_template, output_schema
3. The `PromptManager` auto-loads on startup

## Adding a New Agent

1. Create `product/agents/your_agent.py`
2. Extend `BaseAgent`
3. Implement `process()` method
4. Register with API in `product/api/app.py`