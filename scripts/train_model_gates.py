"""
Model training and ONNX export pipeline with Adaptive Gated Multimodal Fusion.

Scientific contribution: instead of naive feature concatenation, an adaptive
gating mechanism learns how much to trust visual (CNN) vs geometric (FACS)
features on a per-sample basis, providing both improved accuracy and
interpretability of the fusion decision.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
import sys
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.ml.features.geometric import (
    GEOMETRIC_VECTOR_DIM,
    GEOMETRIC_METRIC_NAMES,
    CANONICAL_BLENDSHAPE_KEYS,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ModelTrainer")

GEOM_DIM = GEOMETRIC_VECTOR_DIM
NUM_CLASSES = 7
EMOTION_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]  # ПЕРЕВІРТЕ порядок під ваш label encoding!

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------------------
# Геометрична horizontal-flip перестановка 
# --------------------------------------------------------------------------------------
def _mirror_feature_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[len("left_"):]
    if name.startswith("right_"):
        return "left_" + name[len("right_"):]
    if "Left" in name:
        return name.replace("Left", "Right")
    if "Right" in name:
        return name.replace("Right", "Left")
    return name


def _build_flip_permutation() -> np.ndarray:
    all_names = list(GEOMETRIC_METRIC_NAMES) + [f"facs_{k}" for k in CANONICAL_BLENDSHAPE_KEYS]
    assert len(all_names) == GEOM_DIM
    name_to_idx = {n: i for i, n in enumerate(all_names)}
    perm = np.arange(GEOM_DIM)
    for i, name in enumerate(all_names):
        mirrored = _mirror_feature_name(name)
        if mirrored in name_to_idx:
            perm[i] = name_to_idx[mirrored]
    return perm


GEOM_FLIP_PERMUTATION = _build_flip_permutation()


# --------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------
class MultimodalDataset(Dataset):
    def __init__(self, manifest_path: Path, processed_dir: Path, augment: bool = False) -> None:
        self.df = pd.read_csv(manifest_path)
        self.crops_dir = processed_dir / "crops"
        self.features_dir = processed_dir / "features"
        self.augment = augment

        self.samples = []
        for _, row in self.df.iterrows():
            rel_path = Path(row["path"])
            crop_path = self.crops_dir / rel_path
            feat_path = (self.features_dir / rel_path).with_suffix(".json")
            if crop_path.is_file() and feat_path.is_file():
                self.samples.append((crop_path, feat_path, int(row["label"])))

        logger.info("Dataset %s: %d/%d valid samples (augment=%s)",
                    manifest_path.name, len(self.samples), len(self.df), augment)

        self.color_jitter = T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15)
        self.rotation = T.RandomRotation(degrees=8)
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def __len__(self) -> int:
        return len(self.samples)

    def get_labels(self) -> np.ndarray:
        return np.array([label for _, _, label in self.samples], dtype=np.int64)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        crop_path, feat_path, label = self.samples[idx]
        img = Image.open(crop_path).convert("RGB")

        with open(feat_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        geom_vec = np.array(meta.get("geometric_vector", [0.0] * GEOM_DIM), dtype=np.float32)
        assert geom_vec.shape[0] == GEOM_DIM, f"Bad geom vector in {feat_path}: {geom_vec.shape[0]}"

        if self.augment:
            img = self.color_jitter(img)
            img = self.rotation(img)
            if random.random() < 0.5:
                img = T.functional.hflip(img)
                geom_vec = geom_vec[GEOM_FLIP_PERMUTATION]

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
        img_tensor = self.normalize(img_tensor)
        geom_tensor = torch.from_numpy(np.ascontiguousarray(geom_vec))

        return img_tensor, geom_tensor, label


# --------------------------------------------------------------------------------------
# Adaptive Gated Multimodal Fusion (НАУКОВИЙ ВНЕСОК цієї роботи)
# --------------------------------------------------------------------------------------
class GatedMultimodalFusion(nn.Module):
    """
    Adaptively weights visual vs geometric modality contributions per-sample,
    instead of naive concatenation. The learned gate z in [0,1] represents
    how much the model trusts the visual stream over the geometric stream
    for a given input, which can be logged/visualized for interpretability.
    """

    def __init__(self, visual_dim: int, geom_dim: int, fusion_dim: int = 256) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, fusion_dim)
        self.geom_proj = nn.Sequential(
            nn.Linear(geom_dim, fusion_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.Sigmoid(),
        )
        self.last_gate_mean: float = 0.5  # діагностика, не бере участі в autograd

    def forward(self, vis_embed: torch.Tensor, geom_vec: torch.Tensor) -> torch.Tensor:
        v = self.visual_proj(vis_embed)
        g = self.geom_proj(geom_vec)

        z = self.gate(torch.cat([v, g], dim=1))
        fused = z * v + (1.0 - z) * g

        # Зберігаємо для діагностики/логування (детач - без впливу на граф обчислень)
        self.last_gate_mean = float(z.detach().mean().item())

        # Поєднуємо адаптивно-злиті ознаки з "сирими" проєкціями обох модальностей,
        # щоб не втрачати деталізовану інформацію, яку gate міг "притупити"
        return torch.cat([fused, v, g], dim=1)  # fusion_dim * 3


class MultimodalEmotionClassifier(nn.Module):
    """Multimodal Emotion Classifier: MobileNetV3-Large + Adaptive Gated Fusion."""

    def __init__(self, num_classes: int = NUM_CLASSES, geom_dim: int = GEOM_DIM, fusion_dim: int = 256) -> None:
        super().__init__()

        weights = tv_models.MobileNet_V3_Large_Weights.DEFAULT
        backbone = tv_models.mobilenet_v3_large(weights=weights)
        self.visual_backbone = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        backbone_out_dim = 960
        self.geom_dim = geom_dim
        self.num_classes = num_classes

        self.fusion = GatedMultimodalFusion(backbone_out_dim, geom_dim, fusion_dim=fusion_dim)
        classifier_in_dim = fusion_dim * 3

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(classifier_in_dim),
            nn.Linear(classifier_in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.visual_backbone.parameters():
            param.requires_grad = trainable

    def forward(self, image: torch.Tensor, geom_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not torch.jit.is_tracing():
            assert geom_vector.shape[1] == self.geom_dim

        x = self.visual_backbone(image)
        x = self.avgpool(x)
        vis_features = torch.flatten(x, 1)

        fused_vector = self.fusion(vis_features, geom_vector)
        logits = self.classifier(fused_vector)

        return logits, vis_features


def export_to_onnx(model: nn.Module, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    model_cpu = model.to(torch.device("cpu"))

    export_img = torch.randn(1, 3, 224, 224, dtype=torch.float32)
    export_geom = torch.randn(1, GEOM_DIM, dtype=torch.float32)

    torch.onnx.export(
        model_cpu, (export_img, export_geom), str(output_path),
        export_params=True, opset_version=14, do_constant_folding=True,
        input_names=["input_image", "input_geom_vector"],
        output_names=["logits", "visual_embedding"],
        dynamic_axes={
            "input_image": {0: "batch_size"}, "input_geom_vector": {0: "batch_size"},
            "logits": {0: "batch_size"}, "visual_embedding": {0: "batch_size"},
        },
        dynamo=False,
    )
    logger.info("Successfully exported ONNX model artifact to: %s", output_path)


def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    class_counts = np.bincount(labels, minlength=num_classes)
    total = class_counts.sum()
    safe_counts = np.clip(class_counts, 1, None)
    weights = total / (num_classes * safe_counts)
    logger.info("Class distribution (train): %s", class_counts.tolist())
    logger.info("Computed class weights:      %s", np.round(weights, 3).tolist())
    return torch.tensor(weights, dtype=torch.float32)


def mixup_batch(imgs: torch.Tensor, geoms: torch.Tensor, labels: torch.Tensor, alpha: float):
    """Mixup augmentation applied jointly to both modalities with the same lambda."""
    if alpha <= 0:
        return imgs, geoms, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    perm = torch.randperm(imgs.size(0), device=imgs.device)
    mixed_imgs = lam * imgs + (1 - lam) * imgs[perm]
    mixed_geoms = lam * geoms + (1 - lam) * geoms[perm]
    return mixed_imgs, mixed_geoms, labels, labels[perm], lam


def compute_confusion_matrix(preds: np.ndarray, labels: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for p, l in zip(preds, labels):
        cm[l, p] += 1
    return cm


def macro_f1_from_confusion(cm: np.ndarray) -> Tuple[float, np.ndarray]:
    num_classes = cm.shape[0]
    f1_per_class = np.zeros(num_classes)
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_per_class[c] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(f1_per_class.mean()), f1_per_class


def run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    optimizer: optim.Optimizer | None, num_classes: int, mixup_alpha: float = 0.0,
) -> Tuple[float, float, np.ndarray, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    per_class_correct = np.zeros(num_classes, dtype=np.int64)
    per_class_total = np.zeros(num_classes, dtype=np.int64)
    gate_values = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for imgs, geoms, labels in loader:
            imgs, geoms, labels = imgs.to(device), geoms.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()
                imgs, geoms, labels_a, labels_b, lam = mixup_batch(imgs, geoms, labels, mixup_alpha)
                logits, _ = model(imgs, geoms)
                loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss.backward()
                optimizer.step()
                eval_labels = labels_a  # наближена train accuracy при mixup
            else:
                logits, _ = model(imgs, geoms)
                loss = criterion(logits, labels)
                eval_labels = labels

            gate_values.append(model.fusion.last_gate_mean)

            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == eval_labels).sum().item()
            total += len(labels)

            for c in range(num_classes):
                mask = eval_labels == c
                per_class_total[c] += mask.sum().item()
                per_class_correct[c] += (preds[mask] == c).sum().item()

    avg_loss = total_loss / total if total > 0 else float("inf")
    acc = correct / total if total > 0 else 0.0
    recall = np.divide(per_class_correct, per_class_total,
                        out=np.zeros_like(per_class_correct, dtype=np.float64),
                        where=per_class_total > 0)
    avg_gate = float(np.mean(gate_values)) if gate_values else 0.5
    return avg_loss, acc, recall, avg_gate


def final_evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> dict:
    """Детальна фінальна оцінка: confusion matrix, macro-F1, gate-статистика по класах."""
    model.eval()
    all_preds, all_labels, all_gates = [], [], []

    with torch.no_grad():
        for imgs, geoms, labels in loader:
            imgs, geoms = imgs.to(device), geoms.to(device)
            logits, _ = model(imgs, geoms)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())
            all_gates.extend([model.fusion.last_gate_mean] * len(labels))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = compute_confusion_matrix(all_preds, all_labels, num_classes)
    macro_f1, f1_per_class = macro_f1_from_confusion(cm)

    # Середнє значення gate (довіра до visual-модальності) по кожному ІСТИННОМУ класу -
    # ключовий матеріал для розділу "Interpretability" у дипломі
    gate_by_class = {}
    for c in range(num_classes):
        mask = all_labels == c
        gate_by_class[EMOTION_NAMES[c]] = float(np.mean(np.array(all_gates)[mask])) if mask.sum() > 0 else None

    return {
        "confusion_matrix": cm.tolist(),
        "macro_f1": macro_f1,
        "f1_per_class": {EMOTION_NAMES[c]: float(f1_per_class[c]) for c in range(num_classes)},
        "avg_visual_trust_gate_by_true_class": gate_by_class,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gated Multimodal Fusion Training Script")
    parser.add_argument("--freeze-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr-stage1", type=float, default=1e-3)
    parser.add_argument("--lr-stage2", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-onnx", type=Path, default=Path("artifacts/models/classifier.onnx"))
    parser.add_argument("--metrics-output", type=Path, default=Path("artifacts/models/final_metrics.json"))
    args = parser.parse_args()

    set_global_seed(args.seed)
    logger.info("Random seed set to %d", args.seed)

    train_manifest = Path("data/manifests/train.csv")
    val_manifest = Path("data/manifests/val.csv")
    processed_dir = Path("data/processed")

    train_dataset = MultimodalDataset(train_manifest, processed_dir, augment=True)
    val_dataset = MultimodalDataset(val_manifest, processed_dir, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    val_loader_final = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info("Training on device: %s | Geometric dim: %d", device, GEOM_DIM)

    model = MultimodalEmotionClassifier(num_classes=NUM_CLASSES, geom_dim=GEOM_DIM)
    model.to(device)

    if args.no_class_weights:
        class_weights_tensor = None
    else:
        class_weights_tensor = compute_class_weights(train_dataset.get_labels(), NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=args.label_smoothing)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None
    total_epochs = args.freeze_epochs + args.finetune_epochs
    epoch_counter = 0

    # ===================== STAGE 1: Frozen Backbone =====================
    logger.info("=== STAGE 1: frozen backbone, %d epochs, lr=%.1e ===", args.freeze_epochs, args.lr_stage1)
    model.set_backbone_trainable(False)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=args.lr_stage1, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.freeze_epochs, 1))

    for epoch in range(args.freeze_epochs):
        epoch_counter += 1
        train_loss, train_acc, _, train_gate = run_epoch(
            model, train_loader, criterion, device, optimizer, NUM_CLASSES, args.mixup_alpha)
        scheduler.step()
        val_loss, val_acc, val_recall, val_gate = run_epoch(
            model, val_loader, criterion, device, None, NUM_CLASSES, 0.0)

        logger.info("[Stage1] Epoch %d/%d - Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%% | "
                    "Avg Gate(visual trust): %.3f",
                    epoch_counter, total_epochs, train_acc * 100, val_loss, val_acc * 100, val_gate)
        logger.info("  Per-class recall: %s", np.round(val_recall * 100, 1).tolist())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info("--> New best model found!")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info("Early stopping in Stage 1 at epoch %d.", epoch_counter)
                break

    # ===================== STAGE 2: Full Fine-Tuning =====================
    if patience_counter < args.patience:
        logger.info("=== STAGE 2: full fine-tuning, up to %d epochs, lr=%.1e ===",
                    args.finetune_epochs, args.lr_stage2)
        model.set_backbone_trainable(True)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=1e-2)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.finetune_epochs, 1))

        for epoch in range(args.finetune_epochs):
            epoch_counter += 1
            train_loss, train_acc, _, train_gate = run_epoch(
                model, train_loader, criterion, device, optimizer, NUM_CLASSES, args.mixup_alpha)
            scheduler.step()
            val_loss, val_acc, val_recall, val_gate = run_epoch(
                model, val_loader, criterion, device, None, NUM_CLASSES, 0.0)

            logger.info("[Stage2] Epoch %d/%d - Train Acc: %.2f%% | Val Loss: %.4f | Val Acc: %.2f%% | "
                        "Avg Gate(visual trust): %.3f",
                        epoch_counter, total_epochs, train_acc * 100, val_loss, val_acc * 100, val_gate)
            logger.info("  Per-class recall: %s", np.round(val_recall * 100, 1).tolist())

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                logger.info("--> New best model found!")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    logger.info("Early stopping in Stage 2 at epoch %d.", epoch_counter)
                    break
    else:
        logger.info("Stage 2 skipped - early stopping already triggered.")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # ===================== Фінальна детальна оцінка =====================
    logger.info("Running final detailed evaluation (confusion matrix, macro-F1, gate interpretability)...")
    metrics = final_evaluate(model, val_loader_final, device, NUM_CLASSES)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_output, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    logger.info("Macro F1: %.4f", metrics["macro_f1"])
    logger.info("Per-class F1: %s", metrics["f1_per_class"])
    logger.info("Avg visual-trust gate by true class: %s", metrics["avg_visual_trust_gate_by_true_class"])
    logger.info("Full metrics saved to: %s", args.metrics_output)

    export_to_onnx(model, args.output_onnx)


if __name__ == "__main__":
    main()