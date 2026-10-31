from pathlib import Path
import json
import zipfile
import pandas as pd

BASE = Path(__file__).resolve().parent
MANIFESTS = BASE / "data" / "manifests"
ARCHIVE = BASE / "emotion_dataset_clean.zip"

MANIFEST_FILES = [
    "all_clean.csv",
    "train.csv",
    "val.csv",
    "test_affectnet.csv",
    "test_raf-db.csv",
]

REPORT_FILES = [
    "summary.json",
    "label_mapping.json",
    "clean_class_counts.csv",
]

EXPECTED_LABELS = set(range(7))


def load_manifest(filename):
    path = MANIFESTS / filename
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)
    required = {"path", "label", "split", "dataset", "rgb_sha256"}

    if not required.issubset(frame.columns):
        raise ValueError(f"{filename}: відсутні обов'язкові колонки")

    if frame[list(required)].isna().any().any():
        raise ValueError(f"{filename}: пропущені обов'язкові значення")

    if frame["path"].duplicated().any():
        raise ValueError(f"{filename}: повторюються шляхи")

    if set(frame["label"]) != EXPECTED_LABELS:
        raise ValueError(
            f"{filename}: потрібні всі класи 0–6; "
            f"знайдено {sorted(frame['label'].unique())}"
        )

    return frame


frames = {
    name: load_manifest(name)
    for name in MANIFEST_FILES
}
all_clean = frames["all_clean.csv"]

# Перевіряємо відповідність окремих manifests загальному.
specifications = {
    "train.csv": ("train", None),
    "val.csv": ("val", None),
    "test_affectnet.csv": ("test", "AffectNet"),
    "test_raf-db.csv": ("test", "RAF-DB"),
}

for filename, (split, dataset) in specifications.items():
    expected = all_clean[all_clean["split"] == split]
    if dataset is not None:
        expected = expected[expected["dataset"] == dataset]

    actual = frames[filename]

    if set(expected["path"]) != set(actual["path"]):
        raise ValueError(f"{filename}: склад не відповідає all_clean.csv")

    expected_labels = expected.set_index("path")["label"].to_dict()
    actual_labels = actual.set_index("path")["label"].to_dict()

    if expected_labels != actual_labels:
        raise ValueError(f"{filename}: мітки не відповідають all_clean.csv")

# Контроль точних RGB-перетинів.
hashes = {
    split: set(
        all_clean.loc[all_clean["split"] == split, "rgb_sha256"]
    )
    for split in ("train", "val", "test")
}

for first, second in (
    ("train", "val"),
    ("train", "test"),
    ("val", "test"),
):
    if hashes[first] & hashes[second]:
        raise ValueError(f"RGB-перетин: {first}/{second}")

# Перевіряємо шляхи до створення архіву.
image_files = []

for relative_path in all_clean["path"]:
    relative = Path(relative_path)

    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Небезпечний шлях: {relative_path}")

    full_path = (BASE / relative).resolve()
    full_path.relative_to(BASE)

    if not full_path.is_file():
        raise FileNotFoundError(full_path)

    image_files.append((full_path, relative.as_posix()))

for filename in REPORT_FILES:
    if not (MANIFESTS / filename).is_file():
        raise FileNotFoundError(MANIFESTS / filename)

temporary = ARCHIVE.with_suffix(".zip.partial")

try:
    # Фото вже стиснуті; додаткове ZIP-стиснення не застосовуємо.
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_STORED
    ) as archive:
        for filename in MANIFEST_FILES + REPORT_FILES:
            archive.write(
                MANIFESTS / filename,
                arcname=f"data/manifests/{filename}"
            )

        for index, (full_path, relative_path) in enumerate(
            image_files, start=1
        ):
            archive.write(full_path, arcname=relative_path)

            if index % 1000 == 0:
                print(f"Запаковано {index}/{len(image_files)}", flush=True)

        archive.writestr(
            "package_info.json",
            json.dumps(
                {
                    "image_records": len(image_files),
                    "preprocessing_applied": False,
                    "alignment_applied_by_packager": False,
                    "near_duplicates_verified": False,
                    "subject_independent_split": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    temporary.replace(ARCHIVE)

except Exception:
    temporary.unlink(missing_ok=True)
    raise

print("\nГотово:", ARCHIVE)
print("Оригінальні файли не змінено.")
print("Наявність усіх класів і manifests перевірено.")