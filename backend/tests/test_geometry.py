from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.slam.geometry import (
    camera_center,
    compose_world_to_camera,
    estimate_relative_pose,
)
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


def _project(
    points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, matrix: np.ndarray
) -> np.ndarray:
    camera = (rotation @ points.T + translation.reshape(3, 1)).T
    pixels = (matrix @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:]


def test_essential_matrix_recovers_translation_direction_with_strong_support() -> None:
    rng = np.random.default_rng(41)
    points = np.column_stack(
        (rng.uniform(-2, 2, 120), rng.uniform(-1.2, 1.2, 120), rng.uniform(5, 11, 120))
    )
    intrinsics = CameraIntrinsics(700, 700, 320, 180, 640, 360)
    rotation, _ = cv2.Rodrigues(np.array([0.01, 0.025, -0.005]))
    translation = np.array([0.5, 0.02, 0.04])
    first = _project(points, np.eye(3), np.zeros(3), intrinsics.matrix)
    second = _project(points, rotation, translation, intrinsics.matrix)

    relative = estimate_relative_pose(first, second, intrinsics, min_inliers=60)

    expected_direction = translation / np.linalg.norm(translation)
    assert np.dot(relative.translation, expected_direction) > 0.9
    assert relative.inlier_count >= 100
    assert np.linalg.det(relative.rotation) == pytest.approx(1.0, abs=1e-6)


def test_essential_matrix_rejects_weak_support() -> None:
    intrinsics = CameraIntrinsics(700, 700, 320, 180, 640, 360)
    points = np.array([[10.0 + index, 20.0] for index in range(5)])

    with pytest.raises(SlamFailure) as error:
        estimate_relative_pose(points, points + 1, intrinsics, min_inliers=8)
    assert error.value.category is SlamErrorCategory.INITIALIZATION_FAILED


def test_pose_composition_and_camera_center_use_world_to_camera_convention() -> None:
    previous = np.eye(4)
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))
    translation = np.array([1.0, 0.0, 0.0])

    current = compose_world_to_camera(previous, rotation, translation)

    assert np.allclose(current[:3, :3], rotation)
    assert np.allclose(current[:3, 3], translation)
    assert np.allclose(camera_center(current), -rotation.T @ translation)
