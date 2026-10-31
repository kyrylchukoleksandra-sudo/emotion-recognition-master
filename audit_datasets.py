from pathlib import Path
from collections import defaultdict
import csv
import hashlib

from PIL import Image, ImageOps

BASE = Path(__file__).resolve().parent

ROOTS = {
    "AffectNet": BASE / "AffectNet",
    "RAF-DB": BASE / "archive" / "RAF-DB",
}

OUT = BASE / "data_reports" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

rows = []
errors = []


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path, records, fields):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


for dataset, root in ROOTS.items():
    if not root.is_dir():
        raise FileNotFoundError(f"Не знайдено: {root}")

    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue

        print(f"Перевіряю {dataset}/{split}...", flush=True)

        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for path in sorted(class_dir.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix.lower() not in EXTENSIONS:
                    continue

                try:
                    # Перевірка структури файлу.
                    with Image.open(path) as image:
                        image.verify()

                    # Повне декодування та EXIF-орієнтація.
                    with Image.open(path) as original:
                        original_mode = original.mode
                        image = ImageOps.exif_transpose(original)
                        rgb = image.convert("RGB")
                        rgb.load()
                        width, height = rgb.size

                        digest = hashlib.sha256()
                        digest.update(f"{width}x{height}:RGB".encode())
                        digest.update(rgb.tobytes())

                    rows.append({
                        "dataset": dataset,
                        "split": split,
                        "source_class": class_dir.name,
                        "path": path.relative_to(BASE).as_posix(),
                        "width": width,
                        "height": height,
                        "original_mode": original_mode,
                        "file_sha256": file_hash(path),
                        "rgb_sha256": digest.hexdigest(),
                    })

                except Exception as exc:
                    errors.append({
                        "dataset": dataset,
                        "split": split,
                        "source_class": class_dir.name,
                        "path": path.relative_to(BASE).as_posix(),
                        "error": str(exc),
                    })

inventory_fields = [
    "dataset", "split", "source_class", "path",
    "width", "height", "original_mode",
    "file_sha256", "rgb_sha256"
]

write_csv(OUT / "inventory.csv", rows, inventory_fields)

write_csv(
    OUT / "read_errors.csv",
    errors,
    ["dataset", "split", "source_class", "path", "error"]
)

duplicates = []

for hash_type in ("file_sha256", "rgb_sha256"):
    groups = defaultdict(list)

    for row in rows:
        groups[row[hash_type]].append(row)

    for digest, members in groups.items():
        if len(members) < 2:
            continue

        split_names = {row["split"] for row in members}
        datasets = {row["dataset"] for row in members}

        for row in members:
            duplicates.append({
                "hash_type": hash_type,
                "hash": digest,
                "cross_split": len(split_names) > 1,
                "cross_dataset": len(datasets) > 1,
                "dataset": row["dataset"],
                "split": row["split"],
                "source_class": row["source_class"],
                "path": row["path"],
            })

write_csv(
    OUT / "exact_duplicates.csv",
    duplicates,
    [
        "hash_type", "hash", "cross_split", "cross_dataset",
        "dataset", "split", "source_class", "path"
    ]
)

print("\nПеревірку завершено.")
print("Звіти:", OUT)
print("Вихідні фото не змінено.")
print("Перекомпресовані, змінені або близькі фото можуть не збігтися:")
print("цей скрипт не є повною перевіркою близьких дублікатів.")