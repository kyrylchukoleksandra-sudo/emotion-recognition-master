"""
Module 3.8: Facial Mimic and Geometric Feature Extraction.

Extracts normalized geometric distances, aspect ratios, asymmetries,
and maps FACS blendshapes into a structured feature vector.

Feature vector length is ALWAYS fixed at GEOMETRIC_VECTOR_DIM (61),
regardless of whether blendshapes were successfully detected.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from backend.app.ml.features.errors import InvalidLandmarksError, ZeroDistanceError

# Canonical MediaPipe Face Mesh landmark indices
LEFT_EYE_TOP_BOTTOM = (386, 374)
RIGHT_EYE_TOP_BOTTOM = (159, 145)
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
MOUTH_CORNER_LEFT_RIGHT = (61, 291)
MOUTH_TOP_BOTTOM = (13, 14)
LEFT_EYEBROW_EYE = (70, 159)
RIGHT_EYEBROW_EYE = (300, 386)

# Канонічний, ФІКСОВАНИЙ список усіх 52 категорій MediaPipe Face Blendshapes.
# Порядок і кількість завжди однакові, незалежно від того, чи було виявлено обличчя.
CANONICAL_BLENDSHAPE_KEYS: list[str] = [
    "_neutral",
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight",
    "eyeLookDownLeft", "eyeLookDownRight",
    "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight",
    "eyeLookUpLeft", "eyeLookUpRight",
    "eyeSquintLeft", "eyeSquintRight",
    "eyeWideLeft", "eyeWideRight",
    "jawForward", "jawLeft", "jawOpen", "jawRight",
    "mouthClose",
    "mouthDimpleLeft", "mouthDimpleRight",
    "mouthFrownLeft", "mouthFrownRight",
    "mouthFunnel",
    "mouthLeft",
    "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight",
    "mouthPucker",
    "mouthRight",
    "mouthRollLower", "mouthRollUpper",
    "mouthShrugLower", "mouthShrugUpper",
    "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight",
    "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]
assert len(CANONICAL_BLENDSHAPE_KEYS) == 52

# Загальна фіксована довжина результуючого вектора: 9 геометричних метрик + 52 blendshape-ознаки
NUM_GEOMETRIC_METRICS = 9
GEOMETRIC_VECTOR_DIM = NUM_GEOMETRIC_METRICS + len(CANONICAL_BLENDSHAPE_KEYS)  # = 61

GEOMETRIC_METRIC_NAMES: list[str] = [
    "left_eye_opening",
    "right_eye_opening",
    "eye_opening_asymmetry",
    "mouth_width",
    "mouth_height",
    "mouth_aspect_ratio",
    "left_eyebrow_elevation",
    "right_eyebrow_elevation",
    "eyebrow_asymmetry",
]
assert len(GEOMETRIC_METRIC_NAMES) == NUM_GEOMETRIC_METRICS

@dataclass(frozen=True)
class FacialMimicFeatures:
    """Structured result of mimic and geometric feature extraction."""

    feature_vector: np.ndarray  # Shape (GEOMETRIC_VECTOR_DIM,), float32 normalized vector
    feature_names: list[str]    # Interpretable names for each feature
    raw_metrics: dict[str, float]  # Unrolled distance ratios and asymmetries


def extract_geometric_features(
    landmarks_norm: np.ndarray,
    blendshapes: dict[str, float] | None = None,
) -> FacialMimicFeatures:
    """
    Calculates normalized geometric mimic features from 3D facial landmarks.

    Args:
        landmarks_norm: Array of shape (N, 3) or (N, 2) in normalized [0, 1] space.
        blendshapes: Optional dictionary of MediaPipe FACS action unit scores.

    Returns:
        FacialMimicFeatures dataclass containing vector and metrics.

    Raises:
        InvalidLandmarksError: If landmarks shape is invalid or insufficient.
        ZeroDistanceError: If interocular distance is too small (e.g. invalid points).
    """
    if landmarks_norm is None or not isinstance(landmarks_norm, np.ndarray):
        raise InvalidLandmarksError("Landmarks must be a non-null numpy.ndarray.")

    if landmarks_norm.ndim != 2 or landmarks_norm.shape[0] < 468 or landmarks_norm.shape[1] < 2:
        raise InvalidLandmarksError(
            f"Expected shape (>=468, >=2); received shape={landmarks_norm.shape}."
        )

    # 1. Base interocular distance for scale normalization
    left_eye_center = np.mean(landmarks_norm[list(LEFT_EYE_CORNERS), :2], axis=0)
    right_eye_center = np.mean(landmarks_norm[list(RIGHT_EYE_CORNERS), :2], axis=0)
    interocular_dist = float(np.linalg.norm(left_eye_center - right_eye_center))

    if interocular_dist < 1e-6:
        raise ZeroDistanceError("Interocular distance is near zero; landmarks are degraded.")

    metrics: dict[str, float] = {}

    # 2. Eye Opening Ratios and Asymmetry
    left_eye_height = np.linalg.norm(
        landmarks_norm[LEFT_EYE_TOP_BOTTOM[0], :2] - landmarks_norm[LEFT_EYE_TOP_BOTTOM[1], :2]
    )
    right_eye_height = np.linalg.norm(
        landmarks_norm[RIGHT_EYE_TOP_BOTTOM[0], :2] - landmarks_norm[RIGHT_EYE_TOP_BOTTOM[1], :2]
    )

    left_opening = float(left_eye_height / interocular_dist)
    right_opening = float(right_eye_height / interocular_dist)

    metrics["left_eye_opening"] = left_opening
    metrics["right_eye_opening"] = right_opening
    metrics["eye_opening_asymmetry"] = float(abs(left_opening - right_opening))

    # 3. Mouth Width, Height, and Aspect Ratio
    mouth_width = float(
        np.linalg.norm(
            landmarks_norm[MOUTH_CORNER_LEFT_RIGHT[0], :2] - landmarks_norm[MOUTH_CORNER_LEFT_RIGHT[1], :2]
        )
        / interocular_dist
    )
    mouth_height = float(
        np.linalg.norm(
            landmarks_norm[MOUTH_TOP_BOTTOM[0], :2] - landmarks_norm[MOUTH_TOP_BOTTOM[1], :2]
        )
        / interocular_dist
    )

    metrics["mouth_width"] = mouth_width
    metrics["mouth_height"] = mouth_height
    metrics["mouth_aspect_ratio"] = float(mouth_height / max(mouth_width, 1e-6))

    # 4. Eyebrow Elevation and Asymmetry
    left_brow_dist = float(
        np.linalg.norm(
            landmarks_norm[LEFT_EYEBROW_EYE[0], :2] - landmarks_norm[LEFT_EYEBROW_EYE[1], :2]
        )
        / interocular_dist
    )
    right_brow_dist = float(
        np.linalg.norm(
            landmarks_norm[RIGHT_EYEBROW_EYE[0], :2] - landmarks_norm[RIGHT_EYEBROW_EYE[1], :2]
        )
        / interocular_dist
    )

    metrics["left_eyebrow_elevation"] = left_brow_dist
    metrics["right_eyebrow_elevation"] = right_brow_dist
    metrics["eyebrow_asymmetry"] = float(abs(left_brow_dist - right_brow_dist))

    assert len(metrics) == NUM_GEOMETRIC_METRICS, (
        f"Expected {NUM_GEOMETRIC_METRICS} base metrics, got {len(metrics)}"
    )

    # 5. Build Combined Feature Vector (ЗАВЖДИ фіксованої довжини = 61)
    feature_values = list(metrics.values())
    feature_names = list(metrics.keys())

    blendshapes = blendshapes or {}
    for key in CANONICAL_BLENDSHAPE_KEYS:
        # Якщо конкретний blendshape відсутній - підставляємо 0.0,
        # замість того, щоб повністю пропускати ознаку (що ламало б розмірність).
        feature_values.append(float(blendshapes.get(key, 0.0)))
        feature_names.append(f"facs_{key}")

    vector = np.array(feature_values, dtype=np.float32)

    assert vector.shape[0] == GEOMETRIC_VECTOR_DIM, (
        f"Feature vector length mismatch: got {vector.shape[0]}, expected {GEOMETRIC_VECTOR_DIM}"
    )

    return FacialMimicFeatures(
        feature_vector=vector,
        feature_names=feature_names,
        raw_metrics=metrics,
    )