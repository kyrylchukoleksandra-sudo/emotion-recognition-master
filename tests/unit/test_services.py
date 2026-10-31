"""
Unit tests for Module 3.11 Service Layer (PredictionOrchestratorService).
"""

from __future__ import annotations

import io
import unittest
from PIL import Image

from backend.app.services.prediction_service import PredictionOrchestratorService


class PredictionServiceModuleTests(unittest.TestCase):
    """Test suite for isolated orchestrator logic."""

    def setUp(self) -> None:
        self.service = PredictionOrchestratorService(db_session=None)

        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        self.test_bytes = buf.getvalue()

    def test_invalid_image_bytes_raises_value_error(self) -> None:
        """Ensures corrupted bytes trigger a ValueError."""
        corrupted_bytes = b"not_an_image_file_bytes"
        with self.assertRaises(ValueError):
            self.service.process_image_bytes(corrupted_bytes)


if __name__ == "__main__":
    unittest.main()