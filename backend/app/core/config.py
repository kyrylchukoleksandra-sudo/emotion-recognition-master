import os
from pathlib import Path
from dataclasses import dataclass

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

@dataclass(frozen=True)
class ImagePreprocessingConfig:
    """Immutable configuration for image preprocessing."""

    target_height: int = 224
    target_width: int = 224

    # EfficientNet input contract: RGB, float32, pixel range [0, 255].
    output_color_space: str = "RGB"
    output_dtype: str = "float32"
    output_min_value: float = 0.0
    output_max_value: float = 255.0

    # Preserve aspect ratio and pad instead of stretching the face crop.
    resize_mode: str = "letterbox"
    padding_rgb: tuple[int, int, int] = (127, 127, 127)

    # Apply CLAHE to luminance rather than independently to RGB channels.
    apply_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    min_input_side: int = 32
    max_input_side: int = 8192


IMAGE_PREPROCESSING = ImagePreprocessingConfig()


# --- Глобальні налаштування додатка та бази даних ---
class Settings:
    PROJECT_NAME: str = "Emotion Recognition API"

    MODEL_ONNX_PATH: Path = BASE_DIR / "artifacts" / "models" / "classifier_v3.onnx"
    FACE_LANDMARKER_PATH: Path = BASE_DIR / "artifacts" / "models" / "face_landmarker.task"
    # URL для підключення до PostgreSQL / pgvector
    SQLALCHEMY_DATABASE_URI: str = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg://alexandrakirilchuk@localhost:5432/emotion_db",
)


settings = Settings()