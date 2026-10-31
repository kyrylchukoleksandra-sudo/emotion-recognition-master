"""SQLAlchemy models for emotion vectors and metadata persistence using pgvector."""

from __future__ import annotations

import datetime
from typing import Any, Dict, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from backend.app.db.session import Base

# 512 visual embedding + 61 geometric/mimic features from Module 3.9
FUSED_VECTOR_DIMENSION = 573


class FacialFeatureRecord(Base):
    """Stores fused emotion feature vectors and associated image metadata."""

    __tablename__ = "facial_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)
    emotion_label: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)

    # 573-dimensional L2-normalized fused feature vector
    fused_vector: Mapped[list[float]] = mapped_column(
        Vector(FUSED_VECTOR_DIMENSION), nullable=False
    )

    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    __table_args__ = (
        Index(
            "idx_facial_features_fused_vector_cosine",
            fused_vector,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"fused_vector": "vector_cosine_ops"},
        ),
    )