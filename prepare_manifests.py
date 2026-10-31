from pathlib import Path
import json

import pandas as pd
from sklearn.model_selection import train_test_split

BASE = Path(__file__).resolve().parent
AUDIT = BASE / "data_reports" / "audit"
OUTPUT = BASE / "data" / "manifests"
OUTPUT.mkdir(parents=True, exist_ok=True)

# Запропоновані параметри розділення, а не характеристики датасету.
SEED = 42
RAF_VAL_FRACTION = 0.15

EMOTIONS = [
    "angry", "disgust", "fear",
    "happy", "sad", "surprise", "neutral"
]
EMOTION_TO_ID = {
    emotion: index for index, emotion in enumerate(EMOTIONS)
}

LABEL_MAPS = {
    "AffectNet": {
        "0": "angry",
        "1": "disgust",
        "2": "fear",
        "3": "happy",
        "4": "sad",
        "5": "surprise",
        "6": "neutral",
        "7": "contempt",
    },
    "RAF-DB": {
        "1": "surprise",
        "2": "fear",
        "3": "disgust",
        "4": "happy",
        "5": "sad",
        "6": "angry",
        "7": "neutral",
    },
}


def save_csv(frame, filename):
    frame.to_csv(OUTPUT / filename, index=False)


inventory_path = AUDIT / "inventory.csv"
errors_path = AUDIT / "read_errors.csv"

if not inventory_path.is_file():
    raise FileNotFoundError(
        f"Не знайдено {inventory_path}. "
        "Вкажи фактичне розташування звітів у змінній AUDIT."
    )

df = pd.read_csv(
    inventory_path,
    dtype={
        "dataset": "string",
        "split": "string",
        "source_class": "string",
        "path": "string",
        "rgb_sha256": "string",
    }
)

required = {
    "dataset", "split", "source_class", "path",
    "width", "height", "rgb_sha256"
}

missing = required - set(df.columns)
if missing:
    raise ValueError(f"В inventory відсутні колонки: {missing}")

if df[list(required)].isna().any().any():
    raise ValueError("Обов'язкові поля містять пропущені значення.")

if df["path"].duplicated().any():
    raise ValueError(
        "В inventory повторюються шляхи. "
        "Потрібно перевірити цей звіт перед підготовкою."
    )

unknown_datasets = set(df["dataset"]) - set(LABEL_MAPS)
if unknown_datasets:
    raise ValueError(f"Невідомі джерела: {unknown_datasets}")

if not set(df["split"]).issubset({"train", "val", "test"}):
    raise ValueError("Знайдено невідоме позначення split.")

for dataset, mapping in LABEL_MAPS.items():
    source_classes = set(
        df.loc[df["dataset"] == dataset, "source_class"]
    )
    unknown_classes = source_classes - set(mapping)
    if unknown_classes:
        raise ValueError(
            f"{dataset}: невідомі класи {unknown_classes}"
        )

df["emotion"] = [
    LABEL_MAPS[dataset][source_class]
    for dataset, source_class in zip(
        df["dataset"], df["source_class"]
    )
]

df["original_split"] = df["split"]
original_count = len(df)

# Конфлікти шукаємо ДО виключення contempt:
# однакові пікселі з різними мітками не повинні потрапляти
# до навчання як нібито однозначно розмічені.
label_counts = df.groupby("rgb_sha256")["emotion"].nunique()
conflict_hashes = set(label_counts[label_counts > 1].index)

conflict_mask = df["rgb_sha256"].isin(conflict_hashes)
conflicts = df.loc[conflict_mask].copy()
conflicts["exclusion_reason"] = "identical_rgb_conflicting_labels"
save_csv(conflicts, "quarantine_label_conflicts.csv")

df = df.loc[~conflict_mask].copy()

contempt = df.loc[df["emotion"] == "contempt"].copy()
contempt["exclusion_reason"] = "outside_selected_classes"
save_csv(contempt, "excluded_contempt.csv")

df = df.loc[df["emotion"] != "contempt"].copy()

# Глобально залишаємо один запис для однакових RGB-пікселів.
# Пріоритет захищає test/val від потрапляння копій у train.
priority = {"test": 0, "val": 1, "train": 2}
df["_priority"] = df["split"].map(priority)

df = df.sort_values(
    ["_priority", "dataset", "path"],
    kind="stable"
)

duplicate_mask = df.duplicated("rgb_sha256", keep="first")

duplicate_exclusions = df.loc[duplicate_mask].copy()
duplicate_exclusions["exclusion_reason"] = "identical_rgb_duplicate"
save_csv(
    duplicate_exclusions.drop(columns="_priority"),
    "excluded_exact_duplicates.csv"
)

df = df.loc[~duplicate_mask].drop(columns="_priority").copy()
df["label"] = df["emotion"].map(EMOTION_TO_ID).astype(int)

# RAF-DB validation: лише з очищеної навчальної частини.
raf_train = df[
    (df["dataset"] == "RAF-DB")
    & (df["split"] == "train")
].copy()

if raf_train.empty:
    raise ValueError("Після очищення RAF-DB train порожній.")

try:
    _, val_indices = train_test_split(
        raf_train.index.to_numpy(),
        test_size=RAF_VAL_FRACTION,
        random_state=SEED,
        stratify=raf_train["label"].to_numpy()
    )
except ValueError as exc:
    raise ValueError(
        "Не вдалося створити стратифікований RAF-DB validation. "
        "Перевір кількість прикладів після очищення."
    ) from exc

df.loc[val_indices, "split"] = "val"

# Це не subject-independent split:
# у наявному inventory немає ідентифікаторів людей.
df["subject_id"] = ""

df = df.sort_values(
    ["dataset", "split", "label", "path"]
).reset_index(drop=True)

save_csv(df, "all_clean.csv")

for split in ("train", "val"):
    save_csv(df[df["split"] == split], f"{split}.csv")

for dataset in LABEL_MAPS:
    test = df[
        (df["dataset"] == dataset)
        & (df["split"] == "test")
    ]
    save_csv(test, f"test_{dataset.lower()}.csv")

# Перевірка відсутності точних RGB-перетинів після підготовки.
hash_sets = {
    split: set(df.loc[df["split"] == split, "rgb_sha256"])
    for split in ("train", "val", "test")
}

for first, second in (
    ("train", "val"),
    ("train", "test"),
    ("val", "test")
):
    if hash_sets[first] & hash_sets[second]:
        raise RuntimeError(
            f"Залишився точний RGB-перетин {first}/{second}."
        )

counts = (
    df.groupby(["dataset", "split", "emotion"])
    .size()
    .reset_index(name="images")
)
save_csv(counts, "clean_class_counts.csv")

read_error_count = None
if errors_path.is_file():
    read_error_count = len(pd.read_csv(errors_path))

summary = {
    "inventory_readable_images": original_count,
    "read_error_records": read_error_count,
    "quarantined_conflict_images": len(conflicts),
    "excluded_contempt_after_conflict_filter": len(contempt),
    "excluded_same_label_exact_duplicates": len(duplicate_exclusions),
    "retained_images": len(df),
    "split_counts": {
        split: int((df["split"] == split).sum())
        for split in ("train", "val", "test")
    },
    "seed": SEED,
    "raf_validation_fraction": RAF_VAL_FRACTION,
    "subject_independent_split": False,
    "near_duplicate_check_completed": False,
    "test_subsets_may_have_changed": True,
}

with (OUTPUT / "summary.json").open("w", encoding="utf-8") as file:
    json.dump(summary, file, ensure_ascii=False, indent=2)

with (OUTPUT / "label_mapping.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(
        {
            "emotions": EMOTIONS,
            "source_mappings": LABEL_MAPS,
            "affectnet_mapping_source": "user_confirmed",
        },
        file,
        ensure_ascii=False,
        indent=2
    )

print(json.dumps(summary, ensure_ascii=False, indent=2))
print("\nГотово:", OUTPUT)
print("Оригінальні зображення не змінено.")