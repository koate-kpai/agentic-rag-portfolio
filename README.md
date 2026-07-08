# Agentic RAG Portfolio

A portfolio of two agentic AI microservices built with **FastAPI** and **OpenAI GPT-4o-mini**, deployed to **Google Cloud Run** via **GitHub Actions**. Each app demonstrates a different approach to combining LLMs with external data retrieval.

---

## Architecture at a Glance

```
                                     ┌─────────────────────────────────────┐
                                     │         GitHub Actions CI/CD        │
                                     │  (lint → test → Cloud Build → Run) │
                                     └─────────────────────────────────────┘
                                                   │
                    ┌──────────────────────────────┼──────────────────────────────┐
                    ▼                              ▼                              ▼
          ┌──────────────────────┐     ┌──────────────────────┐     ┌──────────────────────┐
          │      WebScout        │     │      DocuMind        │     │      Shared Infra     │
          │  (app1-webscout)     │     │  (app2-documind)     │     │                      │
          │                      │     │                      │     │  .github/workflows/   │
          │  POST /search/stream │     │  POST /rag           │     │  pytest.ini           │
          │                      │     │                      │     │  ruff.toml            │
          │  ┌──────────────┐    │     │  ┌──────────────┐    │     │  conftest.py          │
          │  │ GPT-4o-mini  │    │     │  │ GPT-4o-mini  │    │     └──────────────────────┘
          │  │ (tool decide)│    │     │  │ (classify)   │    │
          │  └──────┬───────┘    │     │  └──────┬───────┘    │
          │         │            │     │         │            │
          │         ▼            │     │         ▼            │
          │  ┌──────────────┐    │     │  ┌──────────────┐    │
          │  │ DuckDuckGo   │    │     │  │ TF-IDF       │    │
          │  │ Web Search   │    │     │  │ Vector Store │    │
          │  └──────────────┘    │     │  └──────────────┘    │
          │                      │     │                      │
          │  SSE stream answer   │     │  JSON response       │
          │  + citations         │     │  + thought_process   │
          └──────────────────────┘     └──────────────────────┘
```

---

## App Comparison

| Aspect | WebScout | DocuMind |
|---|---|---|
| **Purpose** | Agentic web search for up-to-date factual lookups | Domain-specific RAG over curated knowledge |
| **Retrieval source** | DuckDuckGo (live web, max 3 results) | TF-IDF cosine similarity (in-memory, 9 mock docs) |
| **Response format** | Server-Sent Events (token stream) | Single JSON response |
| **Self-correction** | No (one-shot tool call) | Yes (evaluate → reformulate → re-retrieve, up to 3 attempts) |
| **Domain routing** | Implicit (LLM decides tool_choice) | Explicit (function calling: finance/healthcare/legal) |
| **Citations** | Extracted from search results (guaranteed accurate) | Document IDs from vector store |
| **Streaming** | Yes | No |

---

## Common Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| **Python** | 3.11 | Runtime language |
| **FastAPI** | 0.110.0 | Web framework |
| **OpenAI** | 1.12.0 | GPT-4o-mini LLM access |
| **Pydantic** | 2.6.1 | Request/response validation |
| **Uvicorn** | 0.27.0 | ASGI server |
| **Google Cloud Run** | — | Serverless container deployment |
| **GitHub Actions** | — | CI/CD (lint + test + deploy) |

---

## Shared Architectural Principles

- **Environment-driven config** (12-factor app) — `LOG_LEVEL`, `OPENAI_API_KEY`, etc. are read from the environment, never hard-coded.
- **Explicit packages** — `__init__.py` files in all `src/` directories for backward-compatible tooling support.
- **Function calling over prompt parsing** — Structured LLM output via OpenAI tools parameter instead of brittle `.strip().lower()` heuristics.
- **Fail-fast error handling** — Unknown domains, empty retrievals, and tool failures return clear messages rather than compounding into confusing responses.
- **Blocking CI quality gates** — Ruff linting, mypy type checking, and pytest must pass before any deployment proceeds.
- **Heavily commented code** — Every file contains architectural reasoning and design rationale for non-technical reviewers.
- **Docker layer caching** — `--cache-from` in Cloud Build reduces deploy time by ~4 minutes per push.

---

## Repository Structure

```
quantitative-finance-modelvIG/
├── .github/
│   └── workflows/
│       └── deploy.yml           CI/CD: lint → test → Cloud Build → Cloud Run
├── app1-webscout/               Agentic web search
│   ├── src/                     FastAPI app + agent logic
│   ├── tests/                   Pytest suite (11 tests)
│   ├── Dockerfile               Multi-stage build
│   └── cloudbuild.yaml          GCP Cloud Build config
├── app2-documind/               Domain-routed RAG
│   ├── src/                     FastAPI app + RAG pipeline
│   ├── tests/                   Pytest suite (16 tests)
│   ├── Dockerfile               Multi-stage build
│   └── cloudbuild.yaml          GCP Cloud Build config
├── conftest.py                  Root-level pytest configuration
├── pytest.ini                   Pytest discovery settings
├── ruff.toml                    Ruff linter configuration
└── .env                         Local API keys (gitignored)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/koate-kpai/quantitative-finance-modelvIG
cd quantitative-finance-modelvIG

# Set your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run WebScout (terminal 1)
cd app1-webscout
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8080

# Run DocuMind (terminal 2)
cd app2-documind
pip install -r requirements.txt
uvicorn src.main:app --reload --port 8081
```

See the individual READMEs for detailed API docs and usage:
- [WebScout README](app1-webscout/README.md)
- [DocuMind README](app2-documind/README.md)

---

## Testing

All 27 tests are mocked (no API key required) and run in under 5 seconds:

```bash
# Run both suites (separate invocations — both apps share the 'src' package name)
pytest app1-webscout/tests/ --tb=short -q
pytest app2-documind/tests/ --tb=short -q
```

---

## Deployment

Every push to `main` triggers CI/CD:

1. **lint-and-test** — Ruff, mypy, pytest (blocks deployment on failure)
2. **deploy-app1** — Cloud Build → Artifact Registry → Cloud Run (`webscout`)
3. **deploy-app2** — Cloud Build → Artifact Registry → Cloud Run (`documind`)

Both services receive `OPENAI_API_KEY` from GitHub Secrets at deploy time.
Authentication uses keyless Workload Identity Federation (no service account keys).
