# WebScout — Agentic Web Search

An intelligent web search agent that uses GPT-4o-mini to decide when to perform a live DuckDuckGo search, then streams back a natural-language answer with guaranteed citations.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Client POST  │ ──> │  FastAPI Server   │ ──> │  WebScoutAgent   │
│  /search/     │     │  (main.py)        │     │  (agent.py)      │
│  stream       │ <── │                   │ <── │                  │
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                    │
                                          ┌─────────┴─────────┐
                                          ▼                   ▼
                                   ┌──────────────┐  ┌──────────────┐
                                   │  OpenAI GPT  │  │  DuckDuckGo  │
                                   │  4o-mini     │  │  Search API  │
                                   │  (decision)  │  │  (tools.py)  │
                                   └──────────────┘  └──────────────┘
```

### Flow

1. **POST /search/stream** with a query
2. Agent sends the query to GPT-4o-mini with a `search_web` function definition
3. If the model calls `search_web`, the agent runs a DuckDuckGo search (max 3 results)
4. Citations are extracted directly from search results (not from LLM text) for accuracy
5. Final answer is streamed token-by-token via Server-Sent Events (SSE)
6. Final event contains the complete answer and all citations

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone and navigate
cd app1-webscout

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure API key
# Create .env in project root (../.env) with:
# OPENAI_API_KEY=sk-your-key-here

# Run
uvicorn src.main:app --reload --port 8080
```

### Test

```bash
pytest tests/ --tb=short -q
```

## API Reference

### POST /search/stream

Accepts a search query and streams back an SSE response.

**Request:**
```json
{
  "query": "What is the capital of France?"
}
```

**Response (SSE stream):**
```
data: {"token": "Paris"}
data: {"token": " is"}
data: {"token": " the"}
data: {"token": " capital"}
data: {"token": " of"}
data: {"token": " France"}
data: {"token": "."}
data: {"answer": "Paris is the capital of France.", "citations": [{"domain": "en.wikipedia.org", "url": "https://en.wikipedia.org/wiki/Paris"}]}
event: done
data: [DONE]
```

### GET /healthz

**Response:**
```json
{
  "status": "healthy"
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `LOG_LEVEL` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

## Deployment

Deployed to Google Cloud Run via GitHub Actions on push to `main`. See `.github/workflows/deploy.yml` for details.

```bash
# Manual deploy (if needed)
gcloud builds submit . --config cloudbuild.yaml \
  --substitutions=_TAG=us-central1-docker.pkg.dev/$PROJECT_ID/webscout/app1
gcloud run deploy webscout --image $_TAG --platform managed --region us-central1
```
