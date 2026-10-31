"""
Module 3.9: Final Feature Vector Formation and Feature Fusion.

Combines visual embeddings (from Module 3.8) with geometric/mimic features
(from Module 3.8 geometric.py) into a unified, L2-normalized feature vector
ready for classification and pgvector storage.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from backend.app.ml.features.errors import FeatureExtractionError


@dataclass(frozen=True)
class FusedFeatureVector:
    """Container for the fused final feature representation."""

    fused_vector: np.ndarray  # Shape (D,), float32 L2-normalized vector
    dim_visual: int
    dim_geometric: int
    total_dim: int


def fuse_feature_vectors(
    visual_embedding: np.ndarray,
    geometric_vector: np.ndarray,
    visual_weight: float = 0.7,
    geometric_weight: float = 0.3,
) -> FusedFeatureVector:
    """
    Fuses deep visual embeddings with geometric mimic features using weighted concatenation.

    Args:
        visual_embedding: 1D numpy array from CNN/ViT (e.g. 512-d).
        geometric_vector: 1D numpy array from extract_geometric_features (e.g. 61-d).
        visual_weight: Weight scale applied to the normalized visual vector.
        geometric_weight: Weight scale applied to the normalized geometric vector.

    Returns:
        FusedFeatureVector with L2-normalized concatenated vector.

    Raises:
        FeatureExtractionError: If input arrays are invalid or empty.
    """
    if visual_embedding is None or geometric_vector is None:
        raise FeatureExtractionError("Visual and geometric feature vectors must not be None.")

    if not isinstance(visual_embedding, np.ndarray) or not isinstance(geometric_vector, np.ndarray):
        raise FeatureExtractionError("Inputs must be numpy.ndarray instances.")

    if visual_embedding.ndim != 1 or geometric_vector.ndim != 1:
        raise FeatureExtractionError(
            f"Expected 1D vectors; got visual shape={visual_embedding.shape}, "
            f"geometric shape={geometric_vector.shape}."
        )

    if visual_embedding.size == 0 or geometric_vector.size == 0:
        raise FeatureExtractionError("Input feature vectors cannot be empty.")

    # 1. Individual L2 normalization
    norm_vis = np.linalg.norm(visual_embedding)
    vis_normalized = (
        visual_embedding / norm_vis if norm_vis > 1e-6 else visual_embedding
    )

    norm_geo = np.linalg.norm(geometric_vector)
    geo_normalized = (
        geometric_vector / norm_geo if norm_geo > 1e-6 else geometric_vector
    )

    # 2. Weighted Fusion (Concatenation)
    fused_raw = np.concatenate(
        [vis_normalized * visual_weight, geo_normalized * geometric_weight]
    ).astype(np.float32)

    # 3. Final L2 normalization for pgvector cosine similarity search
    final_norm = np.linalg.norm(fused_raw)
    fused_vector = (
        fused_raw / final_norm if final_norm > 1e-6 else fused_raw
    )

    return FusedFeatureVector(
        fused_vector=fused_vector,
        dim_visual=visual_embedding.size,
        dim_geometric=geometric_vector.size,
        total_dim=fused_vector.size,
    )