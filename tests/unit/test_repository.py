"""Unit tests for Module 3.10: Feature Repository & Model Constraints."""

import unittest
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.app.db.models import Base, FUSED_VECTOR_DIMENSION, FacialFeatureRecord
from backend.app.db.repositories.feature_repository import FeatureRepository


class FeatureRepositoryModuleTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        self.db = self.SessionLocal()
        self.repo = FeatureRepository(self.db)

        self.valid_vector = np.random.randn(FUSED_VECTOR_DIMENSION).astype(np.float32)
        self.valid_vector /= np.linalg.norm(self.valid_vector)

    def tearDown(self):
        self.db.close()

    def test_save_feature_record_success(self):
        record = self.repo.save_feature_record(
            image_name="test_image_001.jpg",
            emotion_label="happy",
            fused_vector=self.valid_vector,
            confidence=0.95,
            metadata_json={"mouth_width": 0.45},
        )

        self.assertIsNotNone(record.id)
        self.assertEqual(record.image_name, "test_image_001.jpg")
        self.assertEqual(record.emotion_label, "happy")
        self.assertEqual(len(record.fused_vector), FUSED_VECTOR_DIMENSION)

        fetched = self.db.query(FacialFeatureRecord).filter_by(id=record.id).first()
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.emotion_label, "happy")

    def test_invalid_vector_dimensions_raises_error(self):
        matrix_2d = np.ones((10, 10), dtype=np.float32)

        with self.assertRaises(ValueError):
            self.repo.save_feature_record(
                image_name="bad_image.jpg",
                emotion_label="neutral",
                fused_vector=matrix_2d,
            )


if __name__ == "__main__":
    unittest.main()