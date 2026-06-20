# models.py

from pydantic import BaseModel, Field
from typing import List


# Request schema with strict validation
# to prevent excessively long or empty queries
class SearchRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, max_length=500, description="User search query"
    )


# Represents a single source cited by the LLM
class Citation(BaseModel):
    domain: str
    url: str


# Standard JSON response schema for non-streaming calls
class SearchResponse(BaseModel):
    answer: str
    citations: List[Citation]


# Schema for the health check endpoint
class HealthResponse(BaseModel):
    status: str = "healthy"
