import unittest
from pathlib import Path
import numpy as np
from PIL import Image

from backend.app.ml.landmarks.alignment import align_face_by_eyes
from backend.app.ml.landmarks.detector import FaceLandmarkDetector, DEFAULT_MODEL_PATH
from backend.app.ml.landmarks.errors import NoFaceDetectedError


class LandmarksModuleTests(unittest.TestCase):
    def setUp(self):
        if not DEFAULT_MODEL_PATH.is_file():
            self.skipTest(f"Model file {DEFAULT_MODEL_PATH} not found. Run curl first.")

    def test_alignment_produces_expected_shape(self):
        image = np.zeros((400, 400, 3), dtype=np.uint8)
        left_eye = (250.0, 150.0)
        right_eye = (150.0, 150.0)

        aligned, matrix = align_face_by_eyes(
            image=image,
            left_eye_center=left_eye,
            right_eye_center=right_eye,
            target_size=(224, 224),
        )

        self.assertEqual(aligned.shape, (224, 224, 3))
        self.assertEqual(matrix.shape, (2, 3))

    def test_empty_or_black_frame_raises_no_face_detected(self):
        with FaceLandmarkDetector() as detector:
            black_frame = np.zeros((300, 300, 3), dtype=np.uint8)
            with self.assertRaises(NoFaceDetectedError):
                detector.process_frame(black_frame)

    def test_detector_on_real_image(self):
        candidate_path = Path("AffectNet/train/3/image0000015.jpg")
        if not candidate_path.is_file():
            self.skipTest(f"Sample image {candidate_path} not found.")

        with Image.open(candidate_path) as img:
            rgb_img = np.array(img.convert("RGB"), dtype=np.uint8)

        with FaceLandmarkDetector() as detector:
            result = detector.process_frame(rgb_img, target_size=(224, 224))
            self.assertEqual(result.aligned_face.shape, (224, 224, 3))
            self.assertGreaterEqual(result.landmarks_normalized.shape[0], 468)
            self.assertGreaterEqual(result.landmarks_pixel.shape[0], 468)
            self.assertEqual(len(result.bounding_box), 4)
            self.assertIn("mouthSmileLeft", result.blendshapes)
            self.assertIn("mouthSmileRight", result.blendshapes)


if __name__ == "__main__":
    unittest.main()