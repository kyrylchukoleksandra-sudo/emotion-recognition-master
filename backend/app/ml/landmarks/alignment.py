"""Face alignment based on facial landmarks."""

from __future__ import annotations

import cv2
import numpy as np


def align_face_by_eyes(
    image: np.ndarray,
    left_eye_center: tuple[float, float],
    right_eye_center: tuple[float, float],
    target_size: tuple[int, int] = (224, 224),
    desired_left_eye_x: float = 0.35,
    desired_eye_y: float = 0.38,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align face by rotating and scaling based on left and right eye centers.

    Args:
        image: Source image (BGR or RGB), HxWxC.
        left_eye_center: (x, y) coordinates of the person's left eye (viewer's right).
        right_eye_center: (x, y) coordinates of the person's right eye (viewer's left).
        target_size: (height, width) of the resulting aligned face crop.
        desired_left_eye_x: Target horizontal position of person's left eye (0.0 to 1.0).
        desired_eye_y: Target vertical position of both eyes (0.0 to 1.0).

    Returns:
        aligned_face: Aligned image of shape (target_height, target_width, C).
        affine_matrix: 2x3 transformation matrix used for warpAffine.
    """
    # Vector from right eye to left eye (in image coordinate space)
    dx = left_eye_center[0] - right_eye_center[0]
    dy = left_eye_center[1] - right_eye_center[1]
    dist = np.sqrt(dx * dx + dy * dy)

    # Angle of rotation in degrees
    angle = np.degrees(np.arctan2(dy, dx))

    # Determine desired scale so distance between eyes matches target proportion
    target_h, target_w = target_size
    desired_dist = (1.0 - 2.0 * desired_left_eye_x) * target_w
    scale = desired_dist / max(dist, 1e-6)

    # Midpoint between eyes
    eye_midpoint = (
        (left_eye_center[0] + right_eye_center[0]) * 0.5,
        (left_eye_center[1] + right_eye_center[1]) * 0.5,
    )

    # Rotation matrix around eye midpoint
    rot_mat = cv2.getRotationMatrix2D(eye_midpoint, angle, scale)

    # Adjust translation so eyes end up at desired target position
    t_x = target_w * 0.5
    t_y = target_h * desired_eye_y
    rot_mat[0, 2] += t_x - eye_midpoint[0]
    rot_mat[1, 2] += t_y - eye_midpoint[1]

    aligned_face = cv2.warpAffine(
        image,
        rot_mat,
        (target_w, target_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT101,
    )

    return aligned_face, rot_mat