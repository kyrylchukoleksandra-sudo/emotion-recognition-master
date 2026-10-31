"""
Emotion predictor module using trained ONNX Runtime classifier (classifier_v3.onnx).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import onnxruntime as ort

from backend.app.core.config import settings


@dataclass
class PredictionResult:
    """Dataclass holding output predictions and internal embeddings."""

    dominant_emotion: str
    confidence: float
    probabilities: Dict[str, float]
    valence: float
    arousal: float
    visual_embedding: np.ndarray  # 960-d visual embedding from MobileNetV3
    logits: np.ndarray
    execution_provider: str = "ONNXExecutionProvider"


class EmotionPredictor:
    """Predicts emotions from preprocessed face image & geometric vector using ONNX Runtime."""

    # 7 стандартних класів емоцій (згідно з підготовленим датасетом і v3)
    EMOTIONS: List[str] = [
        "angry",
        "disgust",
        "fear",
        "happy",
        "sad",
        "surprise",
        "neutral",
    ]

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
    ) -> None:
        path = Path(model_path) if model_path else settings.MODEL_ONNX_PATH
        if not path.is_file():
            raise FileNotFoundError(f"ONNX classifier model not found at {path}")

        # Ініціалізація ONNX Runtime сесії
        self.session = ort.InferenceSession(str(path))
        self.execution_provider = self.session.get_providers()[0]

    def predict(
        self, face_image: np.ndarray, geom_vector: np.ndarray
    ) -> PredictionResult:
        """
        Executes emotion prediction using ONNX model.

        Args:
            face_image: RGB uint8 or float32 image crop of shape (224, 224, 3)
            geom_vector: Geometric feature vector of shape (61,) or (1, 61)

        Returns:
            PredictionResult object containing probabilities, embeddings, and valence/arousal scores.
        """
        # 1. Підготовка зображення для MobileNetV3 (HWC -> CHW, Normalization)
        img_np = face_image.astype(np.float32)
        if img_np.max() > 1.0:
            img_np /= 255.0
        img_np = (img_np - self.IMAGENET_MEAN) / self.IMAGENET_STD
        img_tensor = np.expand_dims(np.transpose(img_np, (2, 0, 1)), axis=0).astype(np.float32)

        # 2. Підготовка геометричного вектора (1, 61)
        if geom_vector.ndim == 1:
            geom_tensor = np.expand_dims(geom_vector, axis=0).astype(np.float32)
        else:
            geom_tensor = geom_vector.astype(np.float32)

        # 3. Запуск вашої нейронки classifier_v3.onnx
        logits, visual_embedding = self.session.run(
            ["logits", "visual_embedding"],
            {
                "input_image": img_tensor,
                "input_geom_vector": geom_tensor,
            },
        )

        # 4. Обчислення ймовірностей (Softmax)
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = (exp_logits / np.sum(exp_logits, axis=1, keepdims=True))[0]

        probabilities_dict = {
            emotion: float(prob) for emotion, prob in zip(self.EMOTIONS, probs)
        }

        dominant_idx = int(np.argmax(probs))
        dominant_emotion = self.EMOTIONS[dominant_idx]
        confidence = float(probs[dominant_idx])

        # Апроксимація Valence/Arousal на основі емоційних класів
        # (Happy -> +Valence, Sad/Fear -> -Valence, Surprise/Anger -> +Arousal)
        valence = float(probs[3] - (probs[0] + probs[2] + probs[4]))  # happy - (angry+fear+sad)
        arousal = float((probs[0] + probs[2] + probs[5]) - probs[6])  # (angry+fear+surprise) - neutral

        return PredictionResult(
            dominant_emotion=dominant_emotion,
            confidence=confidence,
            probabilities=probabilities_dict,
            valence=np.clip(valence, -1.0, 1.0),
            arousal=np.clip(arousal, -1.0, 1.0),
            visual_embedding=visual_embedding[0],  # 960-d
            logits=logits[0],
            execution_provider=self.execution_provider,
        )