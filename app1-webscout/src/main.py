# main.py

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from .models import SearchRequest, HealthResponse
from .agent import WebScoutAgent
from .logger import setup_logger

load_dotenv()
logger = setup_logger("webscout.api")

app = FastAPI(title="WebScout - Agentic Web Search", version="1.0.0")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable required")

agent = WebScoutAgent(api_key=OPENAI_API_KEY)


@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy")


@app.post("/search/stream")
async def search_stream(request: SearchRequest):
    async def event_generator():
        try:
            async for chunk in agent.answer_stream(request.query):
                yield chunk
        except Exception:
            logger.exception("Unhandled streaming error")
            yield f"data: {json.dumps({'error': 'Internal streaming error'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
