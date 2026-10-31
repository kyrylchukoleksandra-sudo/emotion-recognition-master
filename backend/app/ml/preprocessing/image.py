from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from backend.app.core.config import (
    IMAGE_PREPROCESSING,
    ImagePreprocessingConfig,
)
from backend.app.ml.preprocessing.errors import (
    ImageProcessingBackendError,
    ImageTooLargeError,
    ImageTooSmallError,
    InvalidImageError,
    InvalidImageShapeError,
    InvalidPixelRangeError,
)

ColorOrder = Literal["RGB", "BGR"]
FloatRange = Literal["0_1", "0_255"]
ImageInput = str | Path | Image.Image | np.ndarray


@dataclass(frozen=True)
class PreprocessingMetadata:
    """Metadata describing the preprocessing operations applied."""

    source_width: int
    source_height: int
    output_width: int
    output_height: int
    input_mode: str
    output_color_space: str
    output_dtype: str
    output_min: float
    output_max: float
    clahe_applied: bool
    resize_mode: str


@dataclass(frozen=True)
class PreprocessedFace:
    """A preprocessed face image and its processing metadata."""

    image: np.ndarray
    metadata: PreprocessingMetadata


def _normalize_float_array(
    image: np.ndarray,
    *,
    float_range: FloatRange | None,
) -> np.ndarray:
    """Convert a numeric array to uint8 without guessing its value range."""

    if not np.issubdtype(image.dtype, np.number):
        raise InvalidPixelRangeError(
            f"Expected a numeric array; received dtype={image.dtype}."
        )

    if not np.isfinite(image).all():
        raise InvalidPixelRangeError(
            "The image array contains NaN or infinite values."
        )

    if image.dtype == np.uint8:
        if float_range is not None:
            raise InvalidPixelRangeError(
                "float_range must only be specified for floating-point arrays."
            )
        return image

    if not np.issubdtype(image.dtype, np.floating):
        raise InvalidPixelRangeError(
            "Only uint8 and floating-point arrays are supported. "
            "Convert other data types explicitly."
        )

    if float_range == "0_1":
        if image.min() < 0.0 or image.max() > 1.0:
            raise InvalidPixelRangeError(
                "Values must be within [0, 1] when float_range='0_1'."
            )
        scaled = image.astype(np.float32) * 255.0

    elif float_range == "0_255":
        if image.min() < 0.0 or image.max() > 255.0:
            raise InvalidPixelRangeError(
                "Values must be within [0, 255] when float_range='0_255'."
            )
        scaled = image.astype(np.float32)

    else:
        raise InvalidPixelRangeError(
            "For floating-point arrays, explicitly set "
            "float_range='0_1' or float_range='0_255'."
        )

    return np.rint(scaled).astype(np.uint8)


def _composite_alpha(
    image: Image.Image,
    *,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """Composite a transparent image onto the configured neutral background."""

    rgba = image.convert("RGBA")
    background = Image.new(
        "RGBA",
        rgba.size,
        (*background_rgb, 255),
    )
    return Image.alpha_composite(background, rgba).convert("RGB")


def _array_to_pil_rgb(
    image: np.ndarray,
    *,
    color_order: ColorOrder,
    float_range: FloatRange | None,
    config: ImagePreprocessingConfig,
) -> tuple[Image.Image, str]:
    image = _normalize_float_array(image, float_range=float_range)

    if image.ndim == 2:
        grayscale = Image.fromarray(image, mode="L")
        return grayscale.convert("RGB"), "L"

    if image.ndim != 3:
        raise InvalidImageShapeError(
            "Expected an H×W, H×W×1, H×W×3, or H×W×4 array; "
            f"received shape={image.shape}."
        )

    channels = image.shape[-1]

    if channels == 1:
        grayscale = Image.fromarray(image[:, :, 0], mode="L")
        return grayscale.convert("RGB"), "L"

    if channels == 3:
        if color_order == "BGR":
            image = image[:, :, ::-1]
        elif color_order != "RGB":
            raise InvalidImageError(
                f"Unsupported color_order={color_order!r}."
            )

        image = np.ascontiguousarray(image)
        return Image.fromarray(image, mode="RGB"), color_order

    if channels == 4:
        if color_order == "RGB":
            rgb = image[:, :, :3]
        elif color_order == "BGR":
            rgb = image[:, :, [2, 1, 0]]
        else:
            raise InvalidImageError(
                f"Unsupported color_order={color_order!r}."
            )

        rgba = np.dstack((rgb, image[:, :, 3]))
        pil_image = Image.fromarray(
            np.ascontiguousarray(rgba),
            mode="RGBA",
        )
        return (
            _composite_alpha(
                pil_image,
                background_rgb=config.padding_rgb,
            ),
            f"{color_order}A",
        )

    raise InvalidImageShapeError(
        f"Expected 1, 3, or 4 channels; received {channels}."
    )


def _load_pil_rgb(
    image: ImageInput,
    *,
    color_order: ColorOrder,
    float_range: FloatRange | None,
    config: ImagePreprocessingConfig,
) -> tuple[Image.Image, str]:
    if isinstance(image, (str, Path)):
        try:
            with Image.open(image) as opened:
                input_mode = opened.mode
                oriented = ImageOps.exif_transpose(opened)

                if "A" in oriented.getbands():
                    return (
                        _composite_alpha(
                            oriented,
                            background_rgb=config.padding_rgb,
                        ),
                        input_mode,
                    )

                return oriented.convert("RGB"), input_mode

        except (OSError, UnidentifiedImageError) as exc:
            raise InvalidImageError(
                f"Could not open image: {image}"
            ) from exc

    if isinstance(image, Image.Image):
        oriented = ImageOps.exif_transpose(image)
        input_mode = oriented.mode

        if "A" in oriented.getbands():
            return (
                _composite_alpha(
                    oriented,
                    background_rgb=config.padding_rgb,
                ),
                input_mode,
            )

        return oriented.convert("RGB"), input_mode

    if isinstance(image, np.ndarray):
        return _array_to_pil_rgb(
            image,
            color_order=color_order,
            float_range=float_range,
            config=config,
        )

    raise InvalidImageError(
        "Input must be a file path, PIL.Image, or numpy.ndarray."
    )


def _apply_clahe_rgb(
    image: Image.Image,
    config: ImagePreprocessingConfig,
) -> Image.Image:
    """Apply CLAHE to the luminance channel of an RGB image."""

    try:
        import cv2
    except ImportError as exc:
        raise ImageProcessingBackendError(
            "CLAHE is enabled, but OpenCV is not installed. "
            "Install opencv-python or disable apply_clahe in the config."
        ) from exc

    rgb = np.asarray(image, dtype=np.uint8)
    ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)

    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit,
        tileGridSize=config.clahe_tile_grid,
    )
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])

    result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    return Image.fromarray(result, mode="RGB")


def preprocess_face_crop(
    image: ImageInput,
    *,
    color_order: ColorOrder = "RGB",
    float_range: FloatRange | None = None,
    config: ImagePreprocessingConfig = IMAGE_PREPROCESSING,
) -> PreprocessedFace:
    """
    Convert an already-cropped face to the model input format.

    Returns:
        PreprocessedFace containing an RGB float32 array with shape
        (target_height, target_width, 3) and pixel values in [0, 255].

    This function does not perform face detection, landmark extraction,
    alignment, data augmentation, or classification.

    For OpenCV arrays, pass color_order="BGR".
    For floating-point arrays, explicitly specify float_range.
    """

    pil_image, input_mode = _load_pil_rgb(
        image,
        color_order=color_order,
        float_range=float_range,
        config=config,
    )

    source_width, source_height = pil_image.size

    if source_width <= 0 or source_height <= 0:
        raise InvalidImageShapeError(
            f"Invalid image dimensions: {pil_image.size}."
        )

    if max(source_width, source_height) > config.max_input_side:
        raise ImageTooLargeError(
            f"Input size {source_width}×{source_height} exceeds "
            f"max_input_side={config.max_input_side}."
        )

    if min(source_width, source_height) < config.min_input_side:
        raise ImageTooSmallError(
            f"Input size is {source_width}×{source_height}; "
            f"the minimum side is {config.min_input_side}px."
        )

    if config.apply_clahe:
        pil_image = _apply_clahe_rgb(pil_image, config)

    if config.resize_mode != "letterbox":
        raise InvalidImageError(
            f"Unsupported resize_mode={config.resize_mode!r}."
        )

    # Preserve aspect ratio and pad instead of stretching the face.
    pil_image = ImageOps.pad(
        pil_image,
        size=(config.target_width, config.target_height),
        method=Image.Resampling.LANCZOS,
        color=config.padding_rgb,
        centering=(0.5, 0.5),
    )

    output = np.asarray(pil_image, dtype=np.float32)

    expected_shape = (
        config.target_height,
        config.target_width,
        3,
    )
    if output.shape != expected_shape:
        raise InvalidImageShapeError(
            f"Expected output shape {expected_shape}; "
            f"received {output.shape}."
        )

    if not np.isfinite(output).all():
        raise InvalidPixelRangeError(
            "The processed image contains NaN or infinite values."
        )

    actual_min = float(output.min())
    actual_max = float(output.max())

    if actual_min < config.output_min_value:
        raise InvalidPixelRangeError(
            f"Output minimum {actual_min} is below the allowed range."
        )

    if actual_max > config.output_max_value:
        raise InvalidPixelRangeError(
            f"Output maximum {actual_max} is above the allowed range."
        )

    metadata = PreprocessingMetadata(
        source_width=source_width,
        source_height=source_height,
        output_width=config.target_width,
        output_height=config.target_height,
        input_mode=input_mode,
        output_color_space=config.output_color_space,
        output_dtype=str(output.dtype),
        output_min=actual_min,
        output_max=actual_max,
        clahe_applied=config.apply_clahe,
        resize_mode=config.resize_mode,
    )

    return PreprocessedFace(
        image=output,
        metadata=metadata,
    )


def preprocess_dataset_face(
    image: ImageInput,
    *,
    config: ImagePreprocessingConfig = IMAGE_PREPROCESSING,
) -> PreprocessedFace:
    """Preprocess an already-cropped face image from a dataset."""

    return preprocess_face_crop(
        image,
        color_order="RGB",
        config=config,
    )


def preprocess_camera_face(
    aligned_face: np.ndarray,
    *,
    color_order: ColorOrder = "RGB",
    config: ImagePreprocessingConfig = IMAGE_PREPROCESSING,
) -> PreprocessedFace:
    """
    Preprocess a face crop already detected and aligned by module 3.7.

    If the array comes directly from OpenCV, pass color_order="BGR".
    """

    return preprocess_face_crop(
        aligned_face,
        color_order=color_order,
        config=config,
    )