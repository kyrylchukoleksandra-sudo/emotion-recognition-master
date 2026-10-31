from dataclasses import asdict, replace
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from backend.app.core.config import IMAGE_PREPROCESSING
from backend.app.ml.preprocessing import preprocess_dataset_face


BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "data_reports" / "preprocessing_preview"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "AffectNet_happy": Path(
        "AffectNet/train/3/image0000015.jpg"
    ),
    "RAFDB_happy": Path(
        "archive/RAF-DB/train/4/train_00003_aligned.jpg"
    ),
}

config_clahe_off = replace(
    IMAGE_PREPROCESSING,
    apply_clahe=False,
)

config_clahe_on = replace(
    IMAGE_PREPROCESSING,
    apply_clahe=True,
)

panel_size = (320, 300)
canvas = Image.new("RGB", (panel_size[0] * 3, panel_size[1] * len(SAMPLES)), "white")
draw = ImageDraw.Draw(canvas)

metadata_report = {}


def make_panel(title: str, image: Image.Image) -> Image.Image:
    panel = Image.new("RGB", panel_size, "white")
    panel_draw = ImageDraw.Draw(panel)

    thumbnail = ImageOps.contain(image.convert("RGB"), (296, 250))
    x = (panel_size[0] - thumbnail.width) // 2
    y = 30 + (250 - thumbnail.height) // 2

    panel.paste(thumbnail, (x, y))
    panel_draw.text((10, 8), title, fill="black")
    return panel


for row, (name, relative_path) in enumerate(SAMPLES.items()):
    source_path = (BASE / relative_path).resolve()

    if not source_path.is_file():
        raise FileNotFoundError(f"Image not found: {source_path}")

    with Image.open(source_path) as opened:
        original = ImageOps.exif_transpose(opened).convert("RGB")

    without_clahe = preprocess_dataset_face(
        source_path,
        config=config_clahe_off,
    )
    with_clahe = preprocess_dataset_face(
        source_path,
        config=config_clahe_on,
    )

    original_panel = make_panel("Original", original)

    without_clahe_image = Image.fromarray(
        np.clip(without_clahe.image, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    without_clahe_panel = make_panel(
        "Preprocessed — CLAHE off",
        without_clahe_image,
    )

    with_clahe_image = Image.fromarray(
        np.clip(with_clahe.image, 0, 255).astype(np.uint8),
        mode="RGB",
    )
    with_clahe_panel = make_panel(
        "Preprocessed — CLAHE on",
        with_clahe_image,
    )

    y_offset = row * panel_size[1]
    canvas.paste(original_panel, (0, y_offset))
    canvas.paste(without_clahe_panel, (panel_size[0], y_offset))
    canvas.paste(with_clahe_panel, (panel_size[0] * 2, y_offset))

    without_path = OUTPUT_DIR / f"{name}_clahe_off.png"
    with_path = OUTPUT_DIR / f"{name}_clahe_on.png"

    without_clahe_image.save(without_path)
    with_clahe_image.save(with_path)

    metadata_report[name] = {
        "source": relative_path.as_posix(),
        "clahe_off": asdict(without_clahe.metadata),
        "clahe_on": asdict(with_clahe.metadata),
    }

    print(f"{name}:")
    print("  source:", source_path)
    print("  output shape:", without_clahe.image.shape)
    print("  output dtype:", without_clahe.image.dtype)
    print("  CLAHE off:", without_path)
    print("  CLAHE on:", with_path)

comparison_path = OUTPUT_DIR / "comparison.png"
canvas.save(comparison_path)

metadata_path = OUTPUT_DIR / "metadata.json"
metadata_path.write_text(
    json.dumps(metadata_report, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("\nComparison image:", comparison_path)
print("Metadata:", metadata_path)