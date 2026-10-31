from pathlib import Path
import csv
from PIL import Image, ImageOps, ImageDraw

ROOTS = {
    "AffectNet": Path("AffectNet"),
    "RAF-DB": Path("archive/RAF-DB"),
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

OUTPUT = Path("data_reports")
OUTPUT.mkdir(exist_ok=True)

# Параметри лише для оглядових аркушів.
EXAMPLES_PER_CLASS = 6
CELL_WIDTH = 180
CELL_HEIGHT = 210

count_rows = []
preview_errors = []

for dataset_name, root in ROOTS.items():
    if not root.is_dir():
        raise FileNotFoundError(f"Папку не знайдено: {root.resolve()}")

    for split_name in ("train", "val", "test"):
        split_dir = root / split_name
        if not split_dir.is_dir():
            continue

        classes = sorted(
            [p for p in split_dir.iterdir() if p.is_dir()],
            key=lambda p: (
                0, int(p.name)
            ) if p.name.isdigit() else (1, p.name)
        )

        if not classes:
            continue

        sheet = Image.new(
            "RGB",
            (CELL_WIDTH * EXAMPLES_PER_CLASS,
             CELL_HEIGHT * len(classes)),
            "white"
        )
        draw = ImageDraw.Draw(sheet)

        for row_index, class_dir in enumerate(classes):
            files = sorted(
                p for p in class_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in EXTENSIONS
            )

            count_rows.append({
                "dataset": dataset_name,
                "split": split_name,
                "source_class": class_dir.name,
                "image_files": len(files),
            })

            print(
                f"{dataset_name}/{split_name}/{class_dir.name}: "
                f"{len(files)} image files"
            )

            # Розподілені за списком приклади, без оцінки їхніх емоцій.
            if len(files) <= EXAMPLES_PER_CLASS:
                selected = files
            else:
                selected = [
                    files[i * (len(files) - 1)
                          // (EXAMPLES_PER_CLASS - 1)]
                    for i in range(EXAMPLES_PER_CLASS)
                ]

            for column_index, path in enumerate(selected):
                x = column_index * CELL_WIDTH
                y = row_index * CELL_HEIGHT

                try:
                    with Image.open(path) as original:
                        image = ImageOps.exif_transpose(original)
                        original_size = image.size
                        thumbnail = ImageOps.contain(
                            image.convert("RGB"), (170, 165)
                        )

                    sheet.paste(thumbnail, (x + 5, y + 5))

                    draw.text(
                        (x + 5, y + 172),
                        f"class={class_dir.name}",
                        fill="black"
                    )
                    draw.text(
                        (x + 5, y + 185),
                        path.name[:25],
                        fill="black"
                    )
                    draw.text(
                        (x + 5, y + 198),
                        str(original_size),
                        fill="black"
                    )

                except Exception as exc:
                    preview_errors.append({
                        "path": str(path),
                        "error": str(exc),
                    })
                    draw.text(
                        (x + 5, y + 10),
                        "READ ERROR",
                        fill="red"
                    )

        output_path = OUTPUT / f"{dataset_name}_{split_name}_preview.jpg"
        sheet.save(output_path, quality=90)
        print(f"Preview saved: {output_path}")

with (OUTPUT / "class_counts.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(
        file,
        fieldnames=["dataset", "split", "source_class", "image_files"]
    )
    writer.writeheader()
    writer.writerows(count_rows)

with (OUTPUT / "preview_errors.csv").open(
    "w", newline="", encoding="utf-8"
) as file:
    writer = csv.DictWriter(file, fieldnames=["path", "error"])
    writer.writeheader()
    writer.writerows(preview_errors)

print("\nDone. Results:", OUTPUT.resolve())
print("Це огляд вибраних фото, а не повна перевірка всіх файлів.")