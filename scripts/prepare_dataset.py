"""
Offline dataset preparation pipeline matching online API camera preprocessing.
Uses Modules 3.6, 3.7, and 3.8 to generate aligned 224x224 crops and 61-d geometric vectors.
Reads directly from manifest files (e.g. data/manifests/all_clean.csv).
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.features.geometric import extract_geometric_features, GEOMETRIC_VECTOR_DIM
from backend.app.ml.landmarks.detector import FaceLandmarkDetector
from backend.app.ml.landmarks.errors import NoFaceDetectedError
from backend.app.ml.preprocessing.image import preprocess_camera_face

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PrepareDataset")


def process_dataset_manifest(
    manifest_path: Path,
    base_dir: Path,
    output_dir: Path,
) -> None:
    """
    Processes all dataset entries from a manifest CSV (all_clean.csv).
    Applies EXACT camera preprocessing pipeline:
    1. MediaPipe alignment & crop (Module 3.7) [or fallbacks for pre-aligned RAF-DB]
    2. Image standardization & CLAHE (Module 3.6)
    3. Facial landmark & Blendshape geometric extraction (Module 3.8)

    Args:
        manifest_path: Path to all_clean.csv or split CSV.
        base_dir: Project root or data directory containing source images.
        output_dir: Path to write processed crops and feature json files.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    logger.info("Loaded manifest %s with %d rows.", manifest_path.name, len(df))
    logger.info("Expected geometric vector dimension: %d", GEOMETRIC_VECTOR_DIM)

    crops_dir = output_dir / "crops"
    features_dir = output_dir / "features"
    crops_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    success_count = 0
    skipped_count = 0
    fallback_landmarks_count = 0

    with FaceLandmarkDetector() as detector:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Preparing dataset"):
            rel_image_path = Path(row["path"])
            full_image_path = (base_dir / rel_image_path).resolve()

            if not full_image_path.is_file():
                logger.warning("File missing: %s", full_image_path)
                skipped_count += 1
                continue

            # Load image using OpenCV
            bgr_img = cv2.imread(str(full_image_path))
            if bgr_img is None:
                skipped_count += 1
                continue
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            is_raf_db = row.get("dataset") == "RAF-DB"
            landmarks_detected = True

            try:
                if not is_raf_db:
                    # Module 3.7: Full MediaPipe face detection, alignment, and crop
                    detection_res = detector.process_frame(rgb_img, target_size=(224, 224))
                    face_crop_rgb = detection_res.aligned_face
                    landmarks_norm = detection_res.landmarks_normalized
                    blendshapes = detection_res.blendshapes
                    bbox = list(detection_res.bounding_box)
                else:
                    # RAF-DB is already pre-cropped and aligned.
                    # Resize directly to 224x224 to match standard camera input size
                    face_crop_rgb = cv2.resize(rgb_img, (224, 224), interpolation=cv2.INTER_AREA)

                    # Extract landmarks/blendshapes on the cropped face
                    try:
                        detection_res = detector.process_frame(face_crop_rgb, target_size=(224, 224))
                        landmarks_norm = detection_res.landmarks_normalized
                        blendshapes = detection_res.blendshapes
                        bbox = list(detection_res.bounding_box)
                    except NoFaceDetectedError:
                        # Якщо лендмарки не знайдено навіть на вже вирізаному обличчі RAF-DB,
                        # цей семпл буде відсіяний нижче через ZeroDistanceError
                        # (інтероколярна дистанція = 0 для нульових лендмарків).
                        landmarks_detected = False
                        landmarks_norm = np.zeros((478, 3), dtype=np.float32)
                        blendshapes = {}
                        bbox = [0, 0, 224, 224]

                # Module 3.6: Standard image preprocessing (CLAHE, aspect ratio, normalization)
                preprocessed = preprocess_camera_face(face_crop_rgb, color_order="RGB")

                # Module 3.8: Geometric & Mimic feature extraction
                # Якщо landmarks_norm - суцільні нулі, тут буде піднято ZeroDistanceError,
                # і семпл потрапить у except Exception нижче (буде пропущений).
                geometric_res = extract_geometric_features(
                    landmarks_norm=landmarks_norm,
                    blendshapes=blendshapes,
                )

                if not landmarks_detected:
                    fallback_landmarks_count += 1

                # Save preprocessed crop matching camera format (RGB -> BGR for OpenCV write)
                out_crop_path = crops_dir / rel_image_path
                out_crop_path.parent.mkdir(parents=True, exist_ok=True)
                crop_bgr = cv2.cvtColor(preprocessed.image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out_crop_path), crop_bgr)

                # Save metadata and geometric vectors
                out_feat_path = (features_dir / rel_image_path).with_suffix(".json")
                out_feat_path.parent.mkdir(parents=True, exist_ok=True)

                metadata: Dict[str, Any] = {
                    "original_path": str(rel_image_path),
                    "dataset": row.get("dataset", "unknown"),
                    "split": row.get("split", "train"),
                    "emotion": row.get("emotion", "neutral"),
                    "label": int(row.get("label", 0)),
                    "bounding_box": bbox,
                    "geometric_vector": geometric_res.feature_vector.tolist(),
                    "raw_metrics": geometric_res.raw_metrics,
                    "landmarks_detected": landmarks_detected,
                }

                with open(out_feat_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                success_count += 1

            except NoFaceDetectedError:
                skipped_count += 1
            except Exception as exc:
                logger.warning("Error processing %s: %s", rel_image_path, exc)
                skipped_count += 1

    logger.info(
        "Dataset preparation complete! Processed: %d, Skipped: %d, Fallback (no landmarks but kept): %d",
        success_count,
        skipped_count,
        fallback_landmarks_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset Preparation Pipeline")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/all_clean.csv"),
        help="Path to manifest file",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing dataset images",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed"),
        help="Output directory for preprocessed crops and feature JSONs",
    )
    args = parser.parse_args()

    process_dataset_manifest(
        manifest_path=args.manifest,
        base_dir=args.base_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()