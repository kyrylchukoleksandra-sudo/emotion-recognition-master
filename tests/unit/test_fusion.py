"""Unit tests for Module 3.9: Feature Fusion."""

import unittest
import numpy as np

from backend.app.ml.features.errors import FeatureExtractionError
from backend.app.ml.features.fusion import fuse_feature_vectors


class FeatureFusionModuleTests(unittest.TestCase):
    def test_fuse_feature_vectors_success(self):
        visual_emb = np.random.randn(512).astype(np.float32)
        geometric_vec = np.random.randn(61).astype(np.float32)

        result = fuse_feature_vectors(visual_emb, geometric_vec)

        self.assertEqual(result.fused_vector.shape, (573,))
        self.assertEqual(result.dim_visual, 512)
        self.assertEqual(result.dim_geometric, 61)
        self.assertEqual(result.total_dim, 573)

        # Verify unit L2 norm constraint (for pgvector Cosine Distance indexing)
        norm = np.linalg.norm(result.fused_vector)
        self.assertAlmostEqual(float(norm), 1.0, places=4)

    def test_fuse_empty_vector_raises_error(self):
        empty_vec = np.array([], dtype=np.float32)
        valid_vec = np.ones(512, dtype=np.float32)

        with self.assertRaises(FeatureExtractionError):
            fuse_feature_vectors(empty_vec, valid_vec)

    def test_fuse_invalid_dimensions_raises_error(self):
        matrix_2d = np.ones((10, 10), dtype=np.float32)
        valid_vec = np.ones(512, dtype=np.float32)

        with self.assertRaises(FeatureExtractionError):
            fuse_feature_vectors(matrix_2d, valid_vec)


if __name__ == "__main__":
    unittest.main()