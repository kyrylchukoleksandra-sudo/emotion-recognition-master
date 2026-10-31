import unittest
from dataclasses import replace

import numpy as np
from PIL import Image

from backend.app.core.config import IMAGE_PREPROCESSING
from backend.app.ml.preprocessing import (
    ImageTooSmallError,
    InvalidPixelRangeError,
    preprocess_camera_face,
    preprocess_dataset_face,
    preprocess_face_crop,
)


CONFIG_WITHOUT_CLAHE = replace(
    IMAGE_PREPROCESSING,
    apply_clahe=False,
)


class PreprocessingTests(unittest.TestCase):
    def test_rgb_input_shape_dtype_and_range(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        result = preprocess_dataset_face(
            image,
            config=CONFIG_WITHOUT_CLAHE,
        )

        self.assertEqual(result.image.shape, (224, 224, 3))
        self.assertEqual(result.image.dtype, np.float32)
        self.assertGreaterEqual(float(result.image.min()), 0.0)
        self.assertLessEqual(float(result.image.max()), 255.0)
        self.assertEqual(result.metadata.output_color_space, "RGB")
        self.assertFalse(result.metadata.clahe_applied)

    def test_grayscale_input_is_converted_to_three_channels(self):
        image = np.full((80, 80), 128, dtype=np.uint8)

        result = preprocess_face_crop(
            image,
            config=CONFIG_WITHOUT_CLAHE,
        )

        np.testing.assert_allclose(
            result.image[:, :, 0],
            result.image[:, :, 1],
        )
        np.testing.assert_allclose(
            result.image[:, :, 1],
            result.image[:, :, 2],
        )

    def test_bgr_input_is_converted_to_rgb(self):
        # In BGR order, the blue channel is at index 0.
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :, 0] = 255

        result = preprocess_camera_face(
            image,
            color_order="BGR",
            config=CONFIG_WITHOUT_CLAHE,
        )

        center_pixel = result.image[112, 112]
        self.assertGreater(float(center_pixel[2]), 240.0)
        self.assertLess(float(center_pixel[0]), 15.0)

    def test_non_square_input_preserves_aspect_ratio(self):
        image = Image.new("RGB", (120, 60), color=(200, 30, 20))

        result = preprocess_face_crop(
            image,
            config=CONFIG_WITHOUT_CLAHE,
        )

        self.assertEqual(result.image.shape, (224, 224, 3))
        self.assertGreater(
            float(result.image[112, 112, 0]),
            150.0,
        )

    def test_float_array_requires_explicit_range(self):
        image = np.zeros((64, 64, 3), dtype=np.float32)

        with self.assertRaises(InvalidPixelRangeError):
            preprocess_face_crop(
                image,
                config=CONFIG_WITHOUT_CLAHE,
            )

    def test_float_array_in_zero_to_one_range_is_supported(self):
        image = np.full((64, 64, 3), 0.5, dtype=np.float32)

        result = preprocess_face_crop(
            image,
            float_range="0_1",
            config=CONFIG_WITHOUT_CLAHE,
        )

        self.assertEqual(result.image.dtype, np.float32)
        pixel_value = float(result.image[112, 112, 0])
        self.assertTrue(120.0 <= pixel_value <= 135.0)

    def test_too_small_input_is_rejected(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)

        with self.assertRaises(ImageTooSmallError):
            preprocess_face_crop(
                image,
                config=CONFIG_WITHOUT_CLAHE,
            )


if __name__ == "__main__":
    unittest.main()