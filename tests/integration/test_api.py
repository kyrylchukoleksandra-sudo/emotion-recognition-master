"""
Integration tests for Module 3.11: FastAPI Endpoints & Pipeline Integration.
"""

from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest

# Додаємо корінь проєкту (3 рівні вгору від tests/integration/) у sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image
from fastapi.testclient import TestClient

from backend.app.main import app


class ApiIntegrationTests(unittest.TestCase):
    """Integration test suite verifying full API router behavior."""

    def setUp(self) -> None:
        self.client = TestClient(app)

        # Generate a dummy RGB image (224x224x3)
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        self.valid_image_bytes = img_byte_arr.getvalue()

    def test_health_check_endpoint(self) -> None:
        """Tests GET /api/v1/health endpoint."""
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_upload_endpoint_valid_image(self) -> None:
        """Tests POST /api/v1/upload with valid image."""
        files = {"file": ("test.jpg", self.valid_image_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/upload", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "validated")

    def test_upload_endpoint_invalid_content_type(self) -> None:
        """Tests POST /api/v1/upload rejecting non-image payloads."""
        files = {"file": ("test.txt", b"plain text data", "text/plain")}
        response = self.client.post("/api/v1/upload", files=files)
        self.assertEqual(response.status_code, 400)

    def test_preprocess_endpoint(self) -> None:
        """Tests POST /api/v1/preprocess endpoint execution."""
        files = {"file": ("sample.jpg", self.valid_image_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/preprocess", files=files)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "preprocessed_successfully")

    def test_landmarks_endpoint_no_face_detected(self) -> None:
        """Tests POST /api/v1/landmarks returning 400 when no face is detected."""
        files = {"file": ("blank.jpg", self.valid_image_bytes, "image/jpeg")}
        response = self.client.post("/api/v1/landmarks", files=files)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()