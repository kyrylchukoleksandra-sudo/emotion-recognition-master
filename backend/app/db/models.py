"""SQLAlchemy models for users, emotion vectors, and metadata persistence using pgvector."""

from __future__ import annotations

import datetime
import enum
from typing import Any, Dict, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, String, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.app.db.session import Base

# Fused vector dimension (visual embedding + geometric/blendshape features from Module 3.9)
FUSED_VECTOR_DIMENSION = 1021


class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"


class User(Base):
    """User account model supporting Google OAuth and role-based access control."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    picture_url: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    google_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole, name="user_role_enum"), default=UserRole.USER, nullable=False
    )
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship to user's feature records
    records: Mapped[list[FacialFeatureRecord]] = relationship(
        "FacialFeatureRecord", back_populates="user", cascade="all, delete-orphan"
    )


class FacialFeatureRecord(Base):
    """Stores fused emotion feature vectors, cognitive metrics, privacy flags, and image metadata."""

    __tablename__ = "facial_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    image_name: Mapped[str] = mapped_column(String(255), nullable=False)
    emotion_label: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    
    # Cognitive dimension metrics
    valence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    arousal: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # Privacy Consent Flag (Public vs Private record)
    is_public: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)

    # 1021-dimensional L2-normalized fused feature vector
    fused_vector: Mapped[list[float]] = mapped_column(
        Vector(FUSED_VECTOR_DIMENSION), nullable=False
    )

    metadata_json: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSONB, nullable=True
    )

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Relationship back to User
    user: Mapped[Optional[User]] = relationship("User", back_populates="records")

    __table_args__ = (
        Index(
            "idx_facial_features_fused_vector_cosine",
            fused_vector,
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"fused_vector": "vector_cosine_ops"},
        ),
    )