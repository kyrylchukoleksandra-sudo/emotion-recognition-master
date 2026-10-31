from pathlib import Path
import argparse

DIRECTORIES = [
    # Backend / FastAPI
    "backend/app/api/routes",
    "backend/app/core",
    "backend/app/schemas",
    "backend/app/services",

    # ML pipeline
    "backend/app/ml/preprocessing",
    "backend/app/ml/landmarks",
    "backend/app/ml/models",
    "backend/app/ml/features",
    "backend/app/ml/inference",

    # Persistence
    "backend/app/db/repositories",
    "backend/app/db/migrations",

    # Frontend / React + TypeScript
    "frontend/src/components",
    "frontend/src/pages",
    "frontend/src/hooks",
    "frontend/src/services",
    "frontend/src/types",
    "frontend/public",

    # Tests
    "tests/unit",
    "tests/integration",
    "tests/e2e",

    # Operations and documentation
    "infra",
    "docs",
    "artifacts/models",
    "artifacts/reports",
    "data/raw",
    "data/manifests",
    "data/processed",
]

# Порожні файли навмисно містять TODO:
# їх наявність не означає, що функціонал уже реалізований.
PLACEHOLDERS = {
    "backend/app/main.py": '''"""
FastAPI application entry point.

TODO:
- create FastAPI app
- register API routers
- configure lifecycle and health endpoint
""",
''',

    "backend/app/api/routes/health.py": '''"""
TODO: implement service health endpoint.
""",
''',

    "backend/app/api/routes/predictions.py": '''"""
TODO: implement image/frame prediction endpoints.
""",
''',

    "backend/app/api/routes/features.py": '''"""
TODO: implement feature persistence and similarity-search endpoints.
""",
''',

    "backend/app/core/config.py": '''"""
TODO: define validated application settings.
""",
''',

    "backend/app/schemas/prediction.py": '''"""
TODO: define request and response schemas for predictions.
""",
''',

    "backend/app/services/prediction_service.py": '''"""
TODO: orchestrate preprocessing, landmarks, model inference,
feature fusion, and response construction.
""",
''',

    "backend/app/ml/preprocessing/image.py": '''"""
TODO: implement shared image validation, color conversion,
resize, and model-specific normalization.
""",
''',

    "backend/app/ml/landmarks/detector.py": '''"""
TODO: implement face detection and landmark extraction.
""",
''',

    "backend/app/ml/landmarks/alignment.py": '''"""
TODO: implement landmark-based face alignment and failure handling.
""",
''',

    "backend/app/ml/features/geometric.py": '''"""
TODO: implement normalized geometric features from landmarks.
""",
''',

    "backend/app/ml/features/fusion.py": '''"""
TODO: combine visual embedding and geometric features.
""",
''',

    "backend/app/ml/models/classifier.py": '''"""
TODO: define/load the trained expression-classification model.
""",
''',

    "backend/app/ml/inference/predictor.py": '''"""
TODO: return expression probabilities and visual embedding.
""",
''',

    "backend/app/db/session.py": '''"""
TODO: configure PostgreSQL connection and session lifecycle.
""",
''',

    "backend/app/db/models.py": '''"""
TODO: define relational records and pgvector embedding model.
""",
''',

    "backend/app/db/repositories/feature_repository.py": '''"""
TODO: implement embedding persistence and similarity search.
""",
''',

    "backend/README.md": '''# Backend

FastAPI service and ML inference pipeline.

TODO: document installation, configuration, model loading,
API routes, and local run command.
''',

    "frontend/src/App.tsx": '''/**
 * TODO: compose camera/upload, prediction, and result components.
 */
export default function App() {
  return <main>{/* TODO: application UI */}</main>;
}
''',

    "frontend/src/main.tsx": '''/**
 * TODO: mount the React application.
 */
''',

    "frontend/src/services/api.ts": '''/**
 * TODO: implement typed client for the FastAPI backend.
 */
''',

    "frontend/src/hooks/useCamera.ts": '''/**
 * TODO: request browser camera permission, manage stream,
 * and stop tracks when the component unmounts.
 */
''',

    "frontend/src/types/prediction.ts": '''/**
 * TODO: define prediction response and emotion probability types.
 */
''',

    "frontend/README.md": '''# Frontend

React + TypeScript interface.

TODO: document install, development server, API URL,
camera permissions, and production build.
''',

    "tests/README.md": '''# Tests

TODO: add unit, API integration, and end-to-end tests.

Directory structure alone does not mean tests exist or pass.
''',

    "docs/architecture.md": '''# Architecture

TODO: document implemented modules, data flow, model contract,
database schema, API contract, and deployment.
''',

    "infra/docker-compose.yml": '''# TODO:
# Add backend and PostgreSQL with pgvector after local
# configuration and database schema are implemented.
''',

    ".gitignore": '''# Python
__pycache__/
*.py[cod]
.venv/
myenv/
.env

# Local datasets and generated data
data/raw/**
data/processed/**
!data/raw/.gitkeep
!data/processed/.gitkeep

# Model artifacts and generated reports
artifacts/models/**
artifacts/reports/**
!artifacts/models/.gitkeep
!artifacts/reports/.gitkeep

# Frontend
frontend/node_modules/
frontend/dist/

# OS/editor
.DS_Store
.idea/
.vscode/
''',
}

REQUIRED_DIRS_BY_THESIS_SECTION = {
    "3.6 Preprocessing": [
        "backend/app/ml/preprocessing",
    ],
    "3.7 Landmarks / detection / alignment": [
        "backend/app/ml/landmarks",
    ],
    "3.8 Expression features / classifier": [
        "backend/app/ml/models",
        "backend/app/ml/inference",
    ],
    "3.9 Final feature vector": [
        "backend/app/ml/features",
    ],
    "3.10 PostgreSQL / pgvector": [
        "backend/app/db",
        "backend/app/db/repositories",
    ],
    "3.11 FastAPI": [
        "backend/app/api",
        "backend/app/services",
    ],
    "3.12 React + TypeScript": [
        "frontend/src",
        "frontend/src/components",
        "frontend/src/services",
        "frontend/src/types",
    ],
}


def create_scaffold(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)

    for relative_dir in DIRECTORIES:
        (root / relative_dir).mkdir(parents=True, exist_ok=True)

    # Python package markers.
    python_packages = [
        "backend/app",
        "backend/app/api",
        "backend/app/api/routes",
        "backend/app/core",
        "backend/app/schemas",
        "backend/app/services",
        "backend/app/ml",
        "backend/app/ml/preprocessing",
        "backend/app/ml/landmarks",
        "backend/app/ml/models",
        "backend/app/ml/features",
        "backend/app/ml/inference",
        "backend/app/db",
        "backend/app/db/repositories",
    ]

    for package in python_packages:
        init_file = root / package / "__init__.py"
        init_file.touch(exist_ok=True)

    # Keep empty directories visible to Git.
    for relative_dir in DIRECTORIES:
        directory = root / relative_dir
        if not any(directory.iterdir()):
            (directory / ".gitkeep").touch(exist_ok=True)

    # Create placeholders only when absent; never overwrite user code.
    for relative_file, content in PLACEHOLDERS.items():
        path = root / relative_file
        path.parent.mkdir(parents=True, exist_ok=True)

        if not path.exists():
            path.write_text(content, encoding="utf-8")

    write_architecture_check(root)


def write_architecture_check(root: Path) -> None:
    lines = [
        "# Architecture scaffold check",
        "",
        "Перевірено наявність папок, пов'язаних із розділами 3.6–3.12.",
        "",
        "> Це перевірка структури. Вона не підтверджує, що модулі",
        "> реалізовані, запускаються або проходять тести.",
        "",
    ]

    all_present = True

    for section, paths in REQUIRED_DIRS_BY_THESIS_SECTION.items():
        missing = [
            path for path in paths
            if not (root / path).is_dir()
        ]

        if missing:
            all_present = False
            lines.append(f"- [ ] **{section}** — відсутні: "
                         + ", ".join(f"`{p}`" for p in missing))
        else:
            lines.append(f"- [x] **{section}** — структурна папка наявна")

    lines += [
        "",
        "## Що ще треба реалізувати й перевірити",
        "",
        "- [ ] Реальна передобробка, однакова для train та inference.",
        "- [ ] Детекція обличчя, landmarks і вирівнювання.",
        "- [ ] Навчена модель, класифікація й embedding.",
        "- [ ] Геометричні ознаки та, якщо це заявлено як метод, fusion.",
        "- [ ] PostgreSQL/pgvector: схема, збереження і пошук.",
        "- [ ] API-контракти FastAPI.",
        "- [ ] React UI та оброблення дозволу на камеру.",
        "- [ ] Автоматичні тести й інтеграційний запуск.",
        "",
        "## Архітектурні рішення, які треба зафіксувати",
        "",
        "- Обрати один ML-фреймворк для реалізації; не дублювати",
        "  TensorFlow і PyTorch без конкретної потреби.",
        "- Зафіксувати детектор і спосіб отримання landmarks.",
        "- Узгодити категорії виходу моделі, preprocessing і формат embedding.",
        "- Не називати модуль реалізованим, доки немає коду й перевірки.",
        "",
        (
            "**Результат перевірки структури:** усі потрібні папки наявні."
            if all_present
            else "**Результат перевірки структури:** є відсутні папки."
        ),
        "",
    ]

    (root / "docs" / "architecture_check.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def check_scaffold(root: Path) -> bool:
    missing_sections = {}

    for section, paths in REQUIRED_DIRS_BY_THESIS_SECTION.items():
        missing = [
            path for path in paths
            if not (root / path).is_dir()
        ]
        if missing:
            missing_sections[section] = missing

    print(f"Project root: {root}")
    print("\nПеревірка відповідності структури розділам 3.6–3.12:")

    for section, paths in REQUIRED_DIRS_BY_THESIS_SECTION.items():
        missing = missing_sections.get(section, [])
        if missing:
            print(f"[MISSING] {section}: {', '.join(missing)}")
        else:
            print(f"[OK]      {section}")

    print("\nВажливо: перевірено лише наявність папок, не реалізацію модулів.")

    return not missing_sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Корінь проєкту; за замовчуванням поточна папка.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Лише перевірити структуру, нічого не створюючи.",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()

    if args.check:
        ok = check_scaffold(root)
        raise SystemExit(0 if ok else 1)

    create_scaffold(root)
    ok = check_scaffold(root)

    print("\nСтворено/перевірено структуру проєкту.")
    print(f"Звіт: {root / 'docs' / 'architecture_check.md'}")
    print("Існуючі файли скрипт не перезаписує.")
    print("Структура не означає, що функціонал уже реалізований.")

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()