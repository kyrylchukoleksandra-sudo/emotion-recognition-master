"""
Unit and Integration Test suite for ONNX Model Inference & Evaluation Metrics (v3).
Validates artifacts/models/classifier_v3.onnx against data/manifests/val.csv.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from PIL import Image

# 3 рівні вгору від tests/integration/ до кореня проєкту
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "classifier_v3.onnx"
VAL_MANIFEST = PROJECT_ROOT / "data" / "manifests" / "val.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_OUTPUT = PROJECT_ROOT / "artifacts" / "models" / "test_evaluation_report_v3.json"

EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
NUM_CLASSES = len(EMOTION_NAMES)
GEOM_DIM = 61

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)


def preprocess_image(crop_path: Path) -> np.ndarray:
    """Loads and normalizes face crop image for ONNX model input."""
    img = Image.open(crop_path).convert("RGB").resize((224, 224))
    img_np = np.array(img, dtype=np.float32) / 255.0
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD
    img_tensor = np.transpose(img_np, (2, 0, 1))  # HWC to CHW
    return np.expand_dims(img_tensor, axis=0).astype(np.float32)


def load_geom_vector(feat_path: Path) -> np.ndarray:
    """Loads geometric feature vector from JSON."""
    with open(feat_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    geom_vec = np.array(meta.get("geometric_vector", [0.0] * GEOM_DIM), dtype=np.float32)
    return np.expand_dims(geom_vec, axis=0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Computes Accuracy, Confusion Matrix, and Macro F1-Score."""
    acc = float(np.mean(y_true == y_pred))
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    f1_per_class = {}
    for c in range(NUM_CLASSES):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class[EMOTION_NAMES[c]] = float(f1)

    macro_f1 = float(np.mean(list(f1_per_class.values())))
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "f1_per_class": f1_per_class,
        "confusion_matrix": cm.tolist(),
    }


def test_onnx_model_file_exists():
    """Unit Test: Check if classifier_v3.onnx artifact exists."""
    assert MODEL_PATH.is_file(), f"ONNX model file not found at: {MODEL_PATH}"


def test_onnx_session_and_shapes():
    """Unit Test: Verify ONNX input/output shapes and forward execution."""
    session = ort.InferenceSession(str(MODEL_PATH))
    inputs = {i.name: i for i in session.get_inputs()}
    outputs = [o.name for o in session.get_outputs()]

    assert "input_image" in inputs
    assert "input_geom_vector" in inputs
    assert "logits" in outputs
    assert "visual_embedding" in outputs

    dummy_img = np.random.randn(1, 3, 224, 224).astype(np.float32)
    dummy_geom = np.random.randn(1, GEOM_DIM).astype(np.float32)

    res_logits, res_embed = session.run(
        outputs,
        {"input_image": dummy_img, "input_geom_vector": dummy_geom},
    )

    assert res_logits.shape == (1, NUM_CLASSES), f"Unexpected logits shape: {res_logits.shape}"
    assert res_embed.shape == (1, 960), f"Unexpected visual embedding shape: {res_embed.shape}"


def test_evaluate_full_validation_set():
    """Integration Test: Runs full val.csv through ONNX model v3, asserts accuracy, and saves metrics."""
    assert VAL_MANIFEST.is_file(), f"Validation manifest not found: {VAL_MANIFEST}"

    session = ort.InferenceSession(str(MODEL_PATH))
    df = pd.read_csv(VAL_MANIFEST)

    y_true, y_pred = [], []
    valid_samples = 0

    crops_dir = PROCESSED_DIR / "crops"
    features_dir = PROCESSED_DIR / "features"

    for _, row in df.iterrows():
        rel_path = Path(row["path"])
        crop_path = crops_dir / rel_path
        feat_path = (features_dir / rel_path).with_suffix(".json")

        if not (crop_path.is_file() and feat_path.is_file()):
            continue

        img_tensor = preprocess_image(crop_path)
        geom_tensor = load_geom_vector(feat_path)

        logits, _ = session.run(
            ["logits", "visual_embedding"],
            {"input_image": img_tensor, "input_geom_vector": geom_tensor},
        )

        pred_class = int(np.argmax(logits[0]))
        y_true.append(int(row["label"]))
        y_pred.append(pred_class)
        valid_samples += 1

    assert valid_samples > 0, "No valid validation samples were processed!"

    y_true_np = np.array(y_true, dtype=np.int64)
    y_pred_np = np.array(y_pred, dtype=np.int64)

    metrics = compute_metrics(y_true_np, y_pred_np)
    metrics["total_samples_evaluated"] = valid_samples

    REPORT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("📊 ONNX V3 MODEL EVALUATION REPORT")
    print("=" * 50)
    print(f"Total Samples Evaluated : {valid_samples}")
    print(f"Validation Accuracy     : {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1-Score          : {metrics['macro_f1']:.4f}")
    print("\nPer-class F1 Scores:")
    for cls_name, score in metrics["f1_per_class"].items():
        print(f"  • {cls_name:8s}: {score * 100:6.2f}%")
    print("=" * 50)

    # Підвищений поріг перевірки для нової моделі V3
    assert metrics["accuracy"] >= 0.74, (
        f"Model accuracy ({metrics['accuracy']*100:.2f}%) dropped below expected 74% threshold!"
    )