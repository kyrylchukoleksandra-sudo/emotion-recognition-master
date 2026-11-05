"""Repository pattern implementation for feature persistence and similarity search."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import FacialFeatureRecord


class FeatureRepository:
    """Provides methods for saving and similarity-searching emotion vectors."""

    def __init__(self, db_session: Session) -> None:
        self.db = db_session

    def save_feature_record(
        self,
        image_name: str,
        emotion_label: str,
        fused_vector: np.ndarray,
        confidence: float = 1.0,
        valence: float = 0.0,
        arousal: float = 0.0,
        is_public: bool = False,
        user_id: Optional[int] = None,
        metadata_json: Optional[Dict[str, Any]] = None,
    ) -> FacialFeatureRecord:
        """Saves fused feature vector record along with cognitive metrics and privacy consent."""
        if fused_vector.ndim != 1:
            raise ValueError(f"Expected 1D vector; received shape={fused_vector.shape}.")

        record = FacialFeatureRecord(
            user_id=user_id,
            image_name=image_name,
            emotion_label=emotion_label,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            is_public=is_public,
            fused_vector=fused_vector.tolist(),
            metadata_json=metadata_json,
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def search_similar_faces(
        self,
        fused_vector: np.ndarray,
        top_k: int = 5,
        only_public: bool = False,
        filter_emotion: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Queries pgvector using Cosine distance for top-k similar facial feature embeddings."""
        if fused_vector.ndim != 1:
            raise ValueError(f"Expected 1D query vector; received shape={fused_vector.shape}.")

        query_list = fused_vector.tolist()
        cosine_distance = FacialFeatureRecord.fused_vector.cosine_distance(query_list)

        stmt = select(FacialFeatureRecord, cosine_distance.label("distance"))

        if only_public:
            stmt = stmt.where(FacialFeatureRecord.is_public == True)

        if filter_emotion:
            stmt = stmt.where(FacialFeatureRecord.emotion_label == filter_emotion)

        stmt = stmt.order_by("distance").limit(top_k)
        results = self.db.execute(stmt).all()

        return [
            {
                "id": record.id,
                "image_name": record.image_name,
                "emotion_label": record.emotion_label,
                "confidence": float(record.confidence),
                "distance": float(distance),
            }
            for record, distance in results
        ]

    def search_similar_features(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_emotion: Optional[str] = None,
    ) -> List[Tuple[FacialFeatureRecord, float]]:
        """Legacy helper returning raw model tuple and similarity score."""
        if query_vector.ndim != 1:
            raise ValueError(f"Expected 1D query vector; received shape={query_vector.shape}.")

        query_list = query_vector.tolist()
        cosine_distance = FacialFeatureRecord.fused_vector.cosine_distance(query_list)

        stmt = select(FacialFeatureRecord, cosine_distance.label("distance"))

        if filter_emotion:
            stmt = stmt.where(FacialFeatureRecord.emotion_label == filter_emotion)

        stmt = stmt.order_by("distance").limit(top_k)
        results = self.db.execute(stmt).all()

        return [
            (record, float(1.0 - distance)) for record, distance in results
        ]