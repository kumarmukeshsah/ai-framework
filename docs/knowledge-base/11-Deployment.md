# Deployment Guide

## Overview

This guide covers deployment options for the AI Platform Framework, including Docker, Docker Compose, and production considerations.

## Docker Deployment

### Building the Image

```bash
# Build from project root
docker build -t ai-framework:latest -f infra/docker/Dockerfile .

# Or use docker-compose for full stack
docker-compose -f infra/docker/docker-compose.yml build
```

### Dockerfile

```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY product/ product/
COPY evaluation/ evaluation/

# Expose ports
EXPOSE 8000 8001

# Run with production settings
CMD ["uvicorn", "product.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "4", \
     "--limit-max-requests", "10000"]
```

### Running with Docker

```bash
# Run standalone
docker run -p 8000:8000 \
  -e LLM__PROVIDER=openai \
  -e LLM__API_KEY=sk-... \
  ai-framework:latest

# With environment file
docker run -p 8000:8000 --env-file .env.prod ai-framework:latest
```

## Docker Compose (Full Stack)

```yaml
# infra/docker/docker-compose.yml
version: "3.8"

services:
  app:
    build:
      context: ../..
      dockerfile: infra/docker/Dockerfile
    ports:
      - "8000:8000"
      - "8001:8001"
    environment:
      - LLM__PROVIDER=${LLM__PROVIDER:-openai}
      - LLM__API_KEY=${LLM__API_KEY}
      - VECTOR_DB__URL=http://qdrant:6333
      - DATABASE__URL=postgresql+asyncpg://user:pass@postgres:5432/ai_framework
      - OBSERVABILITY__ENABLED=true
    depends_on:
      qdrant:
        condition: service_healthy
      postgres:
        condition: service_healthy
    restart: unless-stopped

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/health"]
      interval: 10s
      retries: 5

  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: ai_framework
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d ai_framework"]
      interval: 10s
      retries: 5

  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin}
    volumes:
      - grafana_data:/var/lib/grafana

volumes:
  qdrant_data:
  postgres_data:
  prometheus_data:
  grafana_data:
```

## Production Considerations

### 1. Environment Configuration

```bash
# .env.production
LLM__PROVIDER=openai
LLM__API_KEY=<your-api-key>
LLM__MODEL=gpt-4o
LLM__MAX_TOKENS=4096
LLM__TIMEOUT_SECONDS=120
LLM__MAX_RETRIES=5

SECURITY__RATE_LIMIT_PER_MINUTE=120
SECURITY__MAX_INPUT_LENGTH=32000
SECURITY__CORS_ORIGINS=["https://app.example.com"]

OBSERVABILITY__ENABLED=true
OBSERVABILITY__ENVIRONMENT=production
OBSERVABILITY__ENABLE_TRACE_EXPORT=true
OBSERVABILITY__TRACE_ENDPOINT=http://otel-collector:4318

DATABASE__URL=postgresql+asyncpg://user:pass@postgres:5432/ai_framework
VECTOR_DB__URL=http://qdrant:6333
VECTOR_DB__COLLECTION=ai_framework_prod
```

### 2. Security Hardening

- **API Keys**: Use secrets manager (AWS Secrets Manager, HashiCorp Vault)
- **HTTPS**: Terminate TLS at load balancer
- **CORS**: Restrict to known origins
- **Rate Limiting**: Configure per-environment limits
- **Input Validation**: Never disable injection detection in production

### 3. Scaling

```yaml
# docker-compose.override.yml for scaling
services:
  app:
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2'
          memory: 4G
        reservations:
          cpus: '1'
          memory: 2G
```

### 4. Monitoring & Alerting

- **Prometheus**: Scrape `/metrics` every 10s
- **Grafana**: Dashboards for request rate, latency, errors, token usage
- **Alerts**:
  - P95 latency > 5s
  - Error rate > 1%
  - Token usage > 80% of quota
  - Provider unavailable

### 5. Backup & Recovery

```bash
# Backup Qdrant
docker exec qdrant tar -czf /backups/qdrant-$(date +%Y%m%d).tar.gz /storage

# Backup PostgreSQL
docker exec postgres pg_dump -U user ai_framework > backup.sql

# Restore
cat backup.sql | docker exec -i postgres psql -U user ai_framework
```

## CI/CD Pipeline

### GitHub Actions Release

The release pipeline (`.github/workflows/release.yml`):
1. Runs full quality gates (ruff, black, mypy)
2. Executes test suite with coverage
3. Runs E2E and performance tests
4. Builds and pushes Docker image to GitHub Container Registry
5. Tags image with `latest` and commit SHA

### GitHub Container Registry

```bash
# Pull from GHCR
docker pull ghcr.io/your-org/ai-framework:latest

# Run with specific version
docker run -p 8000:8000 \
  --env-file .env.prod \
  ghcr.io/your-org/ai-framework:v1.2.3
```

## Health Checks

```bash
# Application health
curl https://api.example.com/health

# Provider health
curl https://api.example.com/health/providers

# Readiness probe (for k8s)
curl -f http://localhost:8000/health
```

## Kubernetes Deployment (Example)

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-framework
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-framework
  template:
    metadata:
      labels:
        app: ai-framework
    spec:
      containers:
      - name: app
        image: ghcr.io/your-org/ai-framework:latest
        ports:
        - containerPort: 8000
        - containerPort: 8001
        envFrom:
        - secretRef:
            name: ai-framework-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          limits:
            cpu: "2"
            memory: "4Gi"
          requests:
            cpu: "1"
            memory: "2Gi"