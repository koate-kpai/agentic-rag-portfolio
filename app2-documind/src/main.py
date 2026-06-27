import os
from dotenv import load_dotenv
from fastapi import FastAPI
from .models import RagRequest, RagResponse, HealthResponse
from .rag_pipeline import DocuMindRAG
from .logger import setup_logger

load_dotenv()
logger = setup_logger("documind.api")

app = FastAPI(title="DocuMind - Agentic RAG", version="1.0.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable required")

rag = DocuMindRAG(api_key=OPENAI_API_KEY)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


@app.post("/rag", response_model=RagResponse)
async def rag_query(request: RagRequest):
    result = await rag.process_query(request.query)
    return RagResponse(**result)
