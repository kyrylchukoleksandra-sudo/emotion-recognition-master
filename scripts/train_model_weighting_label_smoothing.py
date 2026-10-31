"""
Model training and ONNX export pipeline with Multimodal Feature Fusion (Module 3.9).
Uses fully static shape initialization for both training and ONNX export.
Includes class-weighted loss and label smoothing to address class imbalance.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Add project root directory to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.features.geometric import GEOMETRIC_VECTOR_DIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelTrainer")

GEOM_DIM = GEOMETRIC_VECTOR_DIM  # 61, імпортовано напряму з джерела правди
NUM_CLASSES = 7


class MultimodalDataset(Dataset):
    """PyTorch Dataset loading preprocessed crops and geometric feature JSONs."""

    def __init__(self, manifest_path: Path, processed_dir: Path) -> None:
        self.df = pd.read_csv(manifest_path)
        self.processed_dir = processed_dir
        self.crops_dir = processed_dir / "crops"
        self.features_dir = processed_dir / "features"

        self.samples = []
        for _, row in self.df.iterrows():
            rel_path = Path(row["path"])
            crop_path = self.crops_dir / rel_path
            feat_path = (self.features_dir / rel_path).with_suffix(".json")

            if crop_path.is_file() and feat_path.is_file():
                self.samples.append((crop_path, feat_path, int(row["label"])))

        logger.info(
            "Dataset %s: found %d valid samples out of %d.",
            manifest_path.name,
            len(self.samples),
            len(self.df),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> np.ndarray:
        """Returns labels only for actually valid (filtered) samples."""
        return np.array([label for _, _, label in self.samples], dtype=np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        crop_path, feat_path, label = self.samples[idx]

        img = Image.open(crop_path).convert("RGB")
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)

        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        geom_vec = np.array(meta.get("geometric_vector", [0.0] * GEOM_DIM), dtype=np.float32)

        assert geom_vec.shape[0] == GEOM_DIM, (
            f"Unexpected geom vector length in {feat_path}: "
            f"got {geom_vec.shape[0]}, expected {GEOM_DIM}. "
            f"Перегенеруйте датасет через prepare_dataset.py з оновленим geometric.py."
        )

        geom_tensor = torch.from_numpy(geom_vec)

        return img_tensor, geom_tensor, label


class MultimodalEmotionClassifier(nn.Module):
    """Neural Network performing visual feature extraction & geometric feature fusion."""

    def __init__(self, num_classes: int = NUM_CLASSES, geom_dim: int = GEOM_DIM) -> None:
        super().__init__()
        self.visual_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        visual_out_dim = 128
        proj_dim = 512
        self.visual_proj = nn.Linear(visual_out_dim, proj_dim)

        self.geom_dim = geom_dim
        fused_dim = proj_dim + geom_dim
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Linear(fused_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes),
        )

    def forward(self, image: torch.Tensor, geom_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            assert geom_vector.shape[1] == self.geom_dim, (
                f"Expected geom_vector dim {self.geom_dim}, got {geom_vector.shape[1]}"
            )

        vis_features = self.visual_backbone(image)
        vis_embeddings = self.visual_proj(vis_features)

        # Module 3.9 Feature Fusion (Concatenation)
        fused_vector = torch.cat([vis_embeddings, geom_vector], dim=1)
        logits = self.classifier(fused_vector)

        return logits, vis_embeddings


def export_to_onnx(model: nn.Module, output_path: Path) -> None:
    """Експортує навчену модель у ONNX формат після завершення навчання."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    model_cpu = model.to(torch.device("cpu"))

    export_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    export_geom = torch.randn(1, GEOM_DIM, dtype=torch.float32)

    torch.onnx.export(
        model_cpu,
        (export_img, export_geom),
        str(output_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input_image", "input_geom_vector"],
        output_names=["logits", "visual_embedding"],
        dynamic_axes={
            "input_image": {0: "batch_size"},
            "input_geom_vector": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "visual_embedding": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info("Successfully exported ONNX model artifact to: %s", output_path)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """
    Computes inverse-frequency class weights to address dataset imbalance.

    weight[c] = total_samples / (num_classes * count[c])

    Classes with more samples get a LOWER weight, rare classes get a HIGHER weight,
    so the loss function penalizes mistakes on rare classes more heavily.
    """
    class_counts = np.bincount(labels, minlength=num_classes)
    total_samples = class_counts.sum()

    # np.clip запобігає діленню на нуль, якщо якийсь клас відсутній у train split
    safe_counts = np.clip(class_counts, a_min=1, a_max=None)
    weights = total_samples / (num_classes * safe_counts)

    logger.info("Class distribution (train): %s", class_counts.tolist())
    logger.info("Computed class weights:      %s", np.round(weights, 3).tolist())

    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Model Training and ONNX Export Script")
    parser.add_argument("--epochs", type=int, default=15, help="Maximum training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience")
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing factor")
    parser.add_argument(
        "--no-class-weights",
        action="store_true",
        help="Disable class weighting (use plain CrossEntropyLoss)",
    )
    parser.add_argument(
        "--output-onnx",
        type=Path,
        default=Path("artifacts/models/classifier.onnx"),
        help="Output ONNX path",
    )
    args = parser.parse_args()

    train_manifest = Path("data/manifests/train.csv")
    val_manifest = Path("data/manifests/val.csv")
    processed_dir = Path("data/processed")

    train_dataset = MultimodalDataset(train_manifest, processed_dir)
    val_dataset = MultimodalDataset(val_manifest, processed_dir)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Device Selection (Prefers Apple Silicon MPS)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Training on device: %s", device)
    logger.info("Geometric vector dimension: %d", GEOM_DIM)

    model = MultimodalEmotionClassifier(num_classes=NUM_CLASSES, geom_dim=GEOM_DIM)
    model.to(device)

    # --- Class Weighting (боротьба з незбалансованістю датасету) ---
    if args.no_class_weights:
        class_weights_tensor = None
        logger.info("Class weighting disabled (--no-class-weights).")
    else:
        train_labels = train_dataset.get_labels()
        class_weights_tensor = compute_class_weights(train_labels, NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing,
    )
    logger.info("Using label_smoothing=%.2f", args.label_smoothing)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    for epoch in range(args.epochs):
        # --- Training Phase ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for imgs, geoms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
            imgs, geoms, labels = imgs.to(device), geoms.to(device), labels.to(device)

            optimizer.zero_grad()
            logits, _ = model(imgs, geoms)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # --- Validation Phase ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0

        # Per-class метрики - корисно для аналізу, чи допомогло зважування
        per_class_correct = np.zeros(NUM_CLASSES, dtype=np.int64)
        per_class_total = np.zeros(NUM_CLASSES, dtype=np.int64)

        with torch.no_grad():
            for imgs, geoms, labels in val_loader:
                imgs, geoms, labels = imgs.to(device), geoms.to(device), labels.to(device)
                logits, _ = model(imgs, geoms)
                loss = criterion(logits, labels)

                val_loss += loss.item() * len(labels)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

                for c in range(NUM_CLASSES):
                    mask = labels == c
                    per_class_total[c] += mask.sum().item()
                    per_class_correct[c] += (preds[mask] == c).sum().item()

        avg_val_loss = val_loss / val_total if val_total > 0 else float("inf")
        val_acc = val_correct / val_total if val_total > 0 else 0.0

        logger.info(
            "Epoch %d/%d - Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%%",
            epoch + 1,
            args.epochs,
            train_acc * 100,
            avg_val_loss,
            val_acc * 100,
        )

        # Логуємо per-class recall - найважливіша метрика при роботі з дисбалансом
        per_class_recall = np.divide(
            per_class_correct,
            per_class_total,
            out=np.zeros_like(per_class_correct, dtype=np.float64),
            where=per_class_total > 0,
        )
        logger.info("  Per-class recall: %s", np.round(per_class_recall * 100, 1).tolist())

        # --- Best Model Checkpoint (у пам'яті, без зайвого ONNX експорту щоепохи) ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("--> New best model found!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping triggered at epoch %d.", epoch + 1)
                break

    # Завантажуємо найкращі ваги та експортуємо в ONNX один раз після завершення навчання
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    export_to_onnx(model, args.output_onnx)


if __name__ == "__main__":
    main()