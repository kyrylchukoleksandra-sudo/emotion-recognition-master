from pathlib import Path

# Заміни на фактичні шляхи до папок на своєму Mac.
ROOTS = {
    "AffectNet": Path("AffectNet"),
    "RAF-DB": Path("archive/RAF-DB"),
}

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
}
ANNOTATION_EXTENSIONS = {
    ".csv", ".yaml", ".yml", ".json"
}


def inspect_dataset(name, root):
    print(f"\n{'=' * 60}")
    print(f"DATASET: {name}")
    print(f"ROOT: {root}")

    if not root.is_dir():
        print("ERROR: папку не знайдено. Перевір шлях.")
        return

    print("\nВміст кореневої папки:")
    for item in sorted(root.iterdir()):
        kind = "DIR " if item.is_dir() else "FILE"
        print(f"  {kind}: {item.name}")

    for split_name in ("train", "val", "valid", "validation", "test"):
        split = root / split_name
        if not split.is_dir():
            continue

        print(f"\nSPLIT: {split_name}")

        for item in sorted(split.iterdir()):
            if item.is_dir():
                example = next(
                    (
                        p for p in sorted(item.rglob("*"))
                        if p.is_file()
                        and p.suffix.lower() in IMAGE_EXTENSIONS
                    ),
                    None,
                )
                relative = example.relative_to(root) if example else "немає"
                print(f"  Папка: {item.name}; приклад фото: {relative}")
            else:
                print(f"  Файл: {item.name}")

    print("\nМожливі файли анотацій та опису:")
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if (
            path.suffix.lower() in ANNOTATION_EXTENSIONS
            or "readme" in path.name.lower()
            or "label" in path.name.lower()
        ):
            print(f"  {path.relative_to(root)}")


for name, root in ROOTS.items():
    inspect_dataset(name, root)