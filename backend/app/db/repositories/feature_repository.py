"""Repository pattern implementation for feature persistence and similarity search."""

from __future__ import annotations

from typing import List, Optional, Tuple

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
        metadata_json: Optional[dict] = None,
    ) -> FacialFeatureRecord:
        if fused_vector.ndim != 1:
            raise ValueError(f"Expected 1D vector; received shape={fused_vector.shape}.")

        record = FacialFeatureRecord(
            image_name=image_name,
            emotion_label=emotion_label,
            confidence=confidence,
            fused_vector=fused_vector.tolist(),
            metadata_json=metadata_json,
        )

        self.db.add(record)
        self.db.commit()
        self.db.refresh(record)
        return record

    def search_similar_features(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_emotion: Optional[str] = None,
    ) -> List[Tuple[FacialFeatureRecord, float]]:
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