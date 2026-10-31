"""Face landmark detection and analysis using modern MediaPipe Tasks Vision API."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from backend.app.ml.landmarks.alignment import align_face_by_eyes
from backend.app.ml.landmarks.errors import (
    LandmarkExtractionError,
    NoFaceDetectedError,
)

# Canonical 468 landmark indices in MediaPipe Face Mesh
# Left eye (subject's left, viewer's right)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Right eye (subject's right, viewer's left)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "artifacts" / "models" / "face_landmarker.task"


@dataclass(frozen=True)
class FaceDetectionResult:
    """Structured result of face detection, landmarking, and alignment."""

    aligned_face: np.ndarray          # 224x224 RGB aligned face crop
    landmarks_normalized: np.ndarray  # Shape (478, 3) in [0, 1] relative to source
    landmarks_pixel: np.ndarray       # Shape (478, 2) absolute pixel coordinates in source
    bounding_box: tuple[int, int, int, int]  # (x, y, w, h) in source image
    eye_centers: tuple[tuple[float, float], tuple[float, float]]  # (left_eye, right_eye)
    blendshapes: dict[str, float]     # 52 FACS action unit scores (e.g. 'mouthSmileLeft')


class FaceLandmarkDetector:
    """
    State-of-the-art detector using Google MediaPipe Tasks FaceLandmarker.
    Extracts 478 3D landmarks, 52 FACS blendshapes, and aligns face orientation.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise ImportError(
                "mediapipe is required. Install with `pip install mediapipe`."
            ) from exc

        self._mp = mp

        path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        if not path.is_file():
            raise FileNotFoundError(
                f"MediaPipe FaceLandmarker model not found at {path}. "
                "Download it with: "
                "curl -L -o artifacts/models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(path))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )

        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process_frame(
        self,
        frame_rgb: np.ndarray,
        target_size: tuple[int, int] = (224, 224),
    ) -> FaceDetectionResult:
        """
        Detect face, extract 478 landmarks + 52 FACS blendshapes, and align by eyes.

        Args:
            frame_rgb: RGB uint8 numpy array, HxWx3.
            target_size: Target (height, width) for aligned crop.

        Returns:
            FaceDetectionResult with crop, points, and blendshapes.
        """
        if frame_rgb is None or frame_rgb.size == 0:
            raise NoFaceDetectedError("Input frame is empty.")

        h, w = frame_rgb.shape[:2]

        # Wrap in MediaPipe Image container
        mp_image = self._mp.Image(
            image_format=self._mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame_rgb, dtype=np.uint8),
        )

        detection = self._landmarker.detect(mp_image)

        if not detection.face_landmarks:
            raise NoFaceDetectedError("No face detected in the frame.")

        # Take first detected face
        face_lms = detection.face_landmarks[0]
        landmarks_norm = np.array(
            [[lm.x, lm.y, lm.z] for lm in face_lms],
            dtype=np.float32,
        )

        if landmarks_norm.shape[0] < 468:
            raise LandmarkExtractionError(
                f"Expected >=468 landmarks, found {landmarks_norm.shape[0]}."
            )

        # Pixel space coordinates
        landmarks_px = np.zeros((landmarks_norm.shape[0], 2), dtype=np.float32)
        landmarks_px[:, 0] = landmarks_norm[:, 0] * w
        landmarks_px[:, 1] = landmarks_norm[:, 1] * h

        # Bounding box
        x_min = int(np.clip(np.min(landmarks_px[:, 0]), 0, w))
        y_min = int(np.clip(np.min(landmarks_px[:, 1]), 0, h))
        x_max = int(np.clip(np.max(landmarks_px[:, 0]), 0, w))
        y_max = int(np.clip(np.max(landmarks_px[:, 1]), 0, h))
        bbox = (x_min, y_min, x_max - x_min, y_max - y_min)

        # Eye centers
        left_eye_center = (
            float(np.mean(landmarks_px[LEFT_EYE_INDICES, 0])),
            float(np.mean(landmarks_px[LEFT_EYE_INDICES, 1])),
        )
        right_eye_center = (
            float(np.mean(landmarks_px[RIGHT_EYE_INDICES, 0])),
            float(np.mean(landmarks_px[RIGHT_EYE_INDICES, 1])),
        )

        # Align face based on eye centers
        aligned_face, _ = align_face_by_eyes(
            image=frame_rgb,
            left_eye_center=left_eye_center,
            right_eye_center=right_eye_center,
            target_size=target_size,
        )

        # Extract 52 FACS blendshapes if available
        blendshapes_dict: dict[str, float] = {}
        if detection.face_blendshapes:
            for category in detection.face_blendshapes[0]:
                blendshapes_dict[category.category_name] = float(category.score)

        return FaceDetectionResult(
            aligned_face=aligned_face,
            landmarks_normalized=landmarks_norm,
            landmarks_pixel=landmarks_px,
            bounding_box=bbox,
            eye_centers=(left_eye_center, right_eye_center),
            blendshapes=blendshapes_dict,
        )

    def close(self) -> None:
        if hasattr(self, "_landmarker") and self._landmarker:
            self._landmarker.close()

    def __enter__(self) -> FaceLandmarkDetector:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()