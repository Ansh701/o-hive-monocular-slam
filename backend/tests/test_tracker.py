from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.slam.features import TrackedFeatures
from backend.slam.tracker import associate_landmarks, estimate_pose_pnp
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


def _project(points: np.ndarray, pose: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    camera = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    pixels = (matrix @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:]


def test_pnp_recovers_world_to_camera_pose_and_rejects_outliers() -> None:
    rng = np.random.default_rng(12)
    landmarks = np.column_stack(
        (rng.uniform(-2, 2, 80), rng.uniform(-1, 1, 80), rng.uniform(5, 10, 80))
    )
    intrinsics = CameraIntrinsics(720, 715, 320, 180, 640, 360)
    pose = np.eye(4)
    pose[:3, :3], _ = cv2.Rodrigues(np.array([0.02, -0.05, 0.01]))
    pose[:3, 3] = np.array([-0.35, 0.04, 0.08])
    image_points = _project(landmarks, pose, intrinsics.matrix)
    image_points[:8] += rng.uniform(35, 80, (8, 2))

    result = estimate_pose_pnp(landmarks, image_points, intrinsics, min_inliers=50)

    assert result.inlier_count >= 70
    assert np.allclose(result.world_to_camera[:3, :3], pose[:3, :3], atol=2e-3)
    assert np.allclose(result.world_to_camera[:3, 3], pose[:3, 3], atol=2e-2)
    assert result.median_reprojection_error < 0.1


def test_pnp_rejects_insufficient_correspondence_support() -> None:
    intrinsics = CameraIntrinsics(700, 700, 320, 180, 640, 360)
    with pytest.raises(SlamFailure) as error:
        estimate_pose_pnp(np.ones((5, 3)), np.ones((5, 2)), intrinsics, min_inliers=8)
    assert error.value.category is SlamErrorCategory.TRACKING_LOST


def test_landmark_association_uses_lk_source_indices_and_caps_output() -> None:
    tracked = TrackedFeatures(
        previous_points=np.array([[1, 1], [2, 2], [3, 3], [4, 4]], dtype=float),
        current_points=np.array([[2, 1], [3, 2], [4, 3], [5, 4]], dtype=float),
        source_indices=np.array([3, 0, 4, 1]),
        forward_backward_error=np.array([0.1, 0.2, 0.1, 0.3]),
    )
    prior_ids = np.array([10, -1, 12, 13, 14], dtype=np.int64)

    associations = associate_landmarks(tracked, prior_ids, max_landmarks=2)

    assert associations.landmark_ids.tolist() == [13, 14]
    assert associations.image_points.tolist() == [[2.0, 1.0], [4.0, 3.0]]

    with pytest.raises(ValueError, match="bounds"):
        associate_landmarks(tracked, np.array([1, 2]), max_landmarks=10)

