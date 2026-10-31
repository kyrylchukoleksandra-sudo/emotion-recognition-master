"""Unit tests for Module 3.8: Geometric & Mimic Feature Extraction."""

import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from backend.app.ml.features.errors import InvalidLandmarksError, ZeroDistanceError
from backend.app.ml.features.geometric import extract_geometric_features
from backend.app.ml.landmarks.detector import DEFAULT_MODEL_PATH, FaceLandmarkDetector


class GeometricFeaturesModuleTests(unittest.TestCase):
    def setUp(self):
        # Mocking 478 normalized 3D landmarks
        rng = np.random.default_rng(seed=42)
        self.dummy_landmarks = rng.uniform(low=0.2, high=0.8, size=(478, 3)).astype(np.float32)

        # Set distinct eye positions to ensure non-zero interocular distance
        # Left eye center ~ (0.65, 0.4)
        self.dummy_landmarks[362] = [0.60, 0.40, 0.0]
        self.dummy_landmarks[263] = [0.70, 0.40, 0.0]

        # Right eye center ~ (0.35, 0.4)
        self.dummy_landmarks[33] = [0.30, 0.40, 0.0]
        self.dummy_landmarks[133] = [0.40, 0.40, 0.0]

        self.mock_blendshapes = {
            "mouthSmileLeft": 0.85,
            "browOuterUpLeft": 0.12,
        }

    def test_extract_geometric_features_success(self):
        result = extract_geometric_features(
            self.dummy_landmarks,
            blendshapes=self.mock_blendshapes,
        )

        self.assertIsNotNone(result.feature_vector)
        self.assertIsInstance(result.feature_vector, np.ndarray)
        self.assertEqual(result.feature_vector.dtype, np.float32)

        # Check raw metrics
        self.assertIn("left_eye_opening", result.raw_metrics)
        self.assertIn("mouth_aspect_ratio", result.raw_metrics)
        self.assertGreater(result.raw_metrics["left_eye_opening"], 0)

        # Check that blendshapes were mapped into the vector
        self.assertIn("facs_mouthSmileLeft", result.feature_names)
        self.assertEqual(len(result.feature_vector), len(result.feature_names))

    def test_invalid_landmarks_shape_raises_error(self):
        invalid_landmarks = np.zeros((100, 3), dtype=np.float32)  # < 468 points

        with self.assertRaises(InvalidLandmarksError):
            extract_geometric_features(invalid_landmarks)

    def test_zero_interocular_distance_raises_error(self):
        # Eyes overlapping at the same point
        degraded_landmarks = np.zeros((478, 3), dtype=np.float32)

        with self.assertRaises(ZeroDistanceError):
            extract_geometric_features(degraded_landmarks)

    def test_features_extraction_on_real_image(self):
        candidate_path = Path("AffectNet/train/3/image0000015.jpg")
        if not DEFAULT_MODEL_PATH.is_file():
            self.skipTest(f"Model file {DEFAULT_MODEL_PATH} not found.")
        if not candidate_path.is_file():
            self.skipTest(f"Sample image {candidate_path} not found.")

        with Image.open(candidate_path) as img:
            rgb_img = np.array(img.convert("RGB"), dtype=np.uint8)

        with FaceLandmarkDetector() as detector:
            detection_res = detector.process_frame(rgb_img, target_size=(224, 224))
            features_res = extract_geometric_features(
                landmarks_norm=detection_res.landmarks_normalized,
                blendshapes=detection_res.blendshapes,
            )

            self.assertGreater(len(features_res.feature_vector), 0)
            self.assertIn("facs_mouthSmileLeft", features_res.feature_names)


if __name__ == "__main__":
    unittest.main()