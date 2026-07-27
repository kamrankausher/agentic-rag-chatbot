"""
FastAPI application for the Agentic AI RAG Chatbot.
"""

from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    EMBEDDING_MODEL,
    GEMINI_MODEL,
)
from app.graph import AgenticRAGGraph

# ------------------------------------------------------------------
# FastAPI Application
# ------------------------------------------------------------------

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
)

# ------------------------------------------------------------------
# Initialize Graph
# ------------------------------------------------------------------

graph = AgenticRAGGraph()


# ------------------------------------------------------------------
# Pydantic Models
# ------------------------------------------------------------------

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=2,
        max_length=1000,
        description="User question",
        example="What is Agentic AI?",
    )


class RetrievedChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
    score: float


class ChatResponse(BaseModel):
    question: str
    answer: str
    confidence: float
    retrieved_chunks: List[RetrievedChunk]


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def calculate_confidence(chunks: list) -> float:
    """
    Convert Chroma distance scores into a simple confidence score.

    Lower distance = Higher confidence
    """

    if not chunks:
        return 0.0

    avg_distance = sum(chunk["score"] for chunk in chunks) / len(chunks)

    confidence = max(0.0, min(1.0, 1 - avg_distance))

    return round(confidence, 2)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/", tags=["System"])
def root():
    return {
        "project": API_TITLE,
        "version": API_VERSION,
        "status": "running",
        "framework": "FastAPI",
        "workflow": "LangGraph",
        "vector_database": "ChromaDB",
        "embedding_model": EMBEDDING_MODEL,
        "llm": GEMINI_MODEL,
        "documentation": "/docs",
    }


@app.get("/health", tags=["System"])
def health():
    return {
        "status": "healthy",
        "api": "running",
        "vector_database": "connected",
        "llm": GEMINI_MODEL,
    }


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["Chat"],
)
def chat(request: ChatRequest):

    try:

        result = graph.invoke(request.question)

        confidence = calculate_confidence(
            result["retrieved_chunks"]
        )

        return ChatResponse(
            question=request.question,
            answer=result["answer"],
            confidence=confidence,
            retrieved_chunks=result["retrieved_chunks"],
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e),
        )