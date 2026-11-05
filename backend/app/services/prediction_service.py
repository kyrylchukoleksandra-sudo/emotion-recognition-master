"""
Service layer orchestrating image preprocessing, landmark extraction,
model inference, feature fusion, and database persistence.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np
from sqlalchemy.orm import Session

from backend.app.core.config import settings
from backend.app.db.repositories.feature_repository import FeatureRepository
from backend.app.ml.features.fusion import fuse_feature_vectors
from backend.app.ml.features.geometric import extract_geometric_features
from backend.app.ml.inference.predictor import EmotionPredictor
from backend.app.ml.landmarks.detector import FaceLandmarkDetector
from backend.app.ml.preprocessing.image import preprocess_camera_face
from backend.app.schemas.prediction import EmotionPredictionResponse


class PredictionOrchestratorService:
    """Orchestrates end-to-end emotion prediction and vector indexing."""

    def __init__(self, db_session: Optional[Session] = None) -> None:
        self.db = db_session
        self.predictor = EmotionPredictor(model_path=settings.MODEL_ONNX_PATH)

    def process_image_bytes(
        self,
        image_bytes: bytes,
        image_name: str = "upload.jpg",
        save_to_db: bool = True,
        is_public: bool = False,
    ) -> Tuple[EmotionPredictionResponse, np.ndarray]:
        """
        Executes end-to-end emotion recognition pipeline on raw image bytes.
        """
        # 1. Декодування байтів у RGB
        nparr = np.frombuffer(image_bytes, np.uint8)
        bgr_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if bgr_img is None:
            raise ValueError("Could not decode image bytes.")
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

        # 2. Детекція точок обличчя через MediaPipe (face_landmarker.task)
        with FaceLandmarkDetector(model_path=settings.FACE_LANDMARKER_PATH) as detector:
            detection_res = detector.process_frame(rgb_img, target_size=(224, 224))

        # 3. Витягування 61-вимірного вектора FACS
        mimic_features = extract_geometric_features(
            landmarks_norm=detection_res.landmarks_normalized,
            blendshapes=detection_res.blendshapes,
        )

        # 4. Препроцесинг кропу обличчя
        preprocessed = preprocess_camera_face(detection_res.aligned_face, color_order="RGB")

        # 5. Інференс нейромережі classifier_v3.onnx
        prediction = self.predictor.predict(
            face_image=preprocessed.image,
            geom_vector=mimic_features.feature_vector,
        )

        # 6. Злиття векторів (модуль 3.9 -> 1021-d)
        fused_res = fuse_feature_vectors(
            visual_embedding=prediction.visual_embedding,
            geometric_vector=mimic_features.feature_vector,
        )

        similar_faces = []

        if self.db is not None:
            repo = FeatureRepository(self.db)

            # 7. Шукаємо схожі вектори в pgvector ДО або ПОСЛЯ збереження
            # (беремо 5 найближчих збігів за косинусною відстанню)
            if hasattr(repo, "search_similar_faces"):
                similar_faces = repo.search_similar_faces(
                    fused_vector=fused_res.fused_vector,
                    top_k=5,
                    only_public=False,  # Встановіть True, якщо шукати лише серед публічних
                )

            # 8. Збереження у базу даних PostgreSQL (pgvector)
            if save_to_db:
                repo.save_feature_record(
                    image_name=image_name,
                    emotion_label=prediction.dominant_emotion,
                    fused_vector=fused_res.fused_vector,
                    confidence=prediction.confidence,
                    valence=prediction.valence,
                    arousal=prediction.arousal,
                    is_public=is_public,
                    metadata_json=mimic_features.raw_metrics,
                )

        response = EmotionPredictionResponse(
            dominant_emotion=prediction.dominant_emotion,
            confidence=prediction.confidence,
            probabilities=prediction.probabilities,
            valence=prediction.valence,
            arousal=prediction.arousal,
            bounding_box=list(detection_res.bounding_box),
            fused_vector_dim=fused_res.total_dim,
            execution_provider=prediction.execution_provider,
            similar_faces=similar_faces,  # 👈 
        )

        return response, fused_res.fused_vector