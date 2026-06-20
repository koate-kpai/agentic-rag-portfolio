# app2-documind/models.py

from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)


class RetrievedDoc(BaseModel):
    doc_id: str
    content: str
    relevance_score: float


class RagResponse(BaseModel):
    answer: str
    domain: str
    thought_process: List[str]
    retrieved_docs: List[RetrievedDoc]
    citations: List[str]


class HealthResponse(BaseModel):
    status: str = "healthy"
