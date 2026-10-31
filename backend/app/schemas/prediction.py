"""Pydantic schemas for API request and response contracts."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class EmotionPredictionResponse(BaseModel):
    """Schema for individual image prediction result."""

    dominant_emotion: str = Field(..., description="Emotion label with highest confidence")
    confidence: float = Field(..., description="Probability of dominant emotion")
    probabilities: Dict[str, float] = Field(..., description="Probabilities for all classes")
    valence: float = Field(..., description="Continuous valence score [-1, 1]")
    arousal: float = Field(..., description="Continuous arousal score [-1, 1]")
    bounding_box: List[int] = Field(..., description="Face bounding box [x, y, w, h]")
    fused_vector_dim: int = Field(..., description="Dimension of the fused feature vector")
    execution_provider: str = Field(..., description="Inference runtime provider (e.g., CPU/CUDA)")


class SimilarFeatureResult(BaseModel):
    """Schema for a single similar face record from vector search."""

    record_id: int
    image_name: str
    emotion_label: str
    confidence: float
    similarity_score: float
    created_at: datetime


class SimilaritySearchResponse(BaseModel):
    """Response wrapper for pgvector similarity search."""

    top_k: int
    results: List[SimilarFeatureResult]


class HealthCheckResponse(BaseModel):
    """Service health check response schema."""

    status: str
    service: str
    version: str