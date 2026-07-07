# DocuMind — Agentic RAG (Retrieval-Augmented Generation)

An intelligent document query system that classifies user questions into domains (finance, healthcare, legal), retrieves relevant documents via TF-IDF similarity, and generates answers with a self-correction loop.

## Architecture

```
┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Client POST  │ ──> │  FastAPI Server   │ ──> │  DocuMindRAG    │
│  /rag         │     │  (main.py)        │     │  (rag_pipeline) │
│               │ <── │                   │ <── │                  │
└──────────────┘     └──────────────────┘     └──────────────────┘
                                                    │
                    ┌───────────────────────────────┼───────────────┐
                    ▼                               ▼               ▼
          ┌──────────────────┐           ┌──────────────────┐  ┌──────────┐
          │  Domain Classify  │           │  TF-IDF Vector   │  │  OpenAI  │
          │  (function call)  │ ──retrieve─>  Store (scikit) │  │  GPT-4o  │
          │  finance/health/  │           │  (vectorstore)   │  │  -mini   │
          │  legal/unknown    │           └──────────────────┘  └──────────┘
          └──────────────────┘                     ▲
                │                                  │
                └──── self-correction loop ────────┘
                     (evaluate → reformulate → re-retrieve)
```

### Flow

1. **POST /rag** with a query
2. **Domain classification**: GPT-4o-mini classifies the query into `finance`, `healthcare`, `legal`, or `unknown` (uses function calling for constrained output)
3. **Early exit**: if domain is unknown, returns immediately with a clear message
4. **Initial retrieval**: TF-IDF cosine similarity finds the top-2 documents in the classified domain
5. **Self-correction loop** (up to 3 attempts):
   - Evaluator determines if retrieved context is sufficient
   - If not, LLM reformulates the query and re-retrieves
   - Identity guard prevents looping on stale reformulations
6. **Answer generation**: GPT-4o-mini answers using only the retrieved context (max 3 sentences)
7. Response includes thought process and retrieved documents for transparency

## Quick Start

### Prerequisites

- Python 3.11+
- OpenAI API key

### Setup

```bash
# Clone and navigate
cd app2-documind

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

### POST /rag

**Request:**
```json
{
  "query": "What does HIPAA require for patient data?"
}
```

**Response:**
```json
{
  "answer": "HIPAA requires covered entities to ensure the confidentiality and integrity of protected health information (PHI).",
  "domain": "healthcare",
  "thought_process": [
    "Domain classified as: healthcare",
    "Attempt 1: Evaluator says sufficient"
  ],
  "retrieved_docs": [
    {
      "doc_id": "hlth-001",
      "content": "HIPAA requires covered entities to ensure the confidentiality...",
      "relevance_score": 0.4521
    }
  ],
  "citations": ["hlth-001"]
}
```

### GET /healthz

**Response:**
```json
{
  "status": "healthy"
}
```

## Knowledge Domains

Currently supports 3 mock domains with 3 documents each:

| Domain | Documents |
|--------|-----------|
| **Finance** | SEC Rule 10b-5 (securities fraud), Sarbanes-Oxley Act, Insider trading regulations |
| **Healthcare** | HIPAA privacy requirements, FDA drug approval process, Patient consent requirements |
| **Legal** | Contract formation (offer/acceptance/consideration), Stare decisis, Force majeure |

> **Note:** These are mock documents for prototyping. In production, connect to a real vector database (Pinecone, Qdrant, pgvector) with domain-specific embeddings.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | — | OpenAI API key |
| `LOG_LEVEL` | No | `INFO` | Logging level: DEBUG, INFO, WARNING, ERROR |

## Deployment

Deployed to Google Cloud Run via GitHub Actions on push to `main`. See `.github/workflows/deploy.yml` for details.
