from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.slam.map import (
    TriangulationConfig,
    initialize_two_view,
    triangulate_filtered,
)
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]])
POSE_A = np.eye(4)
POSE_B = np.eye(4)
POSE_B[0, 3] = -1.0


def _project(points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    camera = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    pixels = (K @ camera.T).T
    return pixels[:, :2] / pixels[:, 2:]


def _good_points() -> np.ndarray:
    return np.array(
        [
            [-1.5 + 0.3 * (index % 10), -0.7 + 0.35 * (index // 10), 5.0 + index * 0.08]
            for index in range(30)
        ],
        dtype=np.float64,
    )


def test_triangulation_recovers_finite_positive_depth_points() -> None:
    world = _good_points()
    result = triangulate_filtered(
        K,
        POSE_A,
        POSE_B,
        _project(world, POSE_A),
        _project(world, POSE_B),
        TriangulationConfig(min_points=12),
    )

    assert len(result.points) == 30
    assert np.all(np.isfinite(result.points))
    assert np.allclose(result.points, world, atol=1e-5)
    assert np.max(result.reprojection_errors) < 1e-5
    assert np.min(result.parallax_degrees) > 2.0


def test_filters_nonfinite_negative_depth_and_bad_reprojection() -> None:
    good = _good_points()[:12]
    world = np.vstack((good, np.array([[0.0, 0.0, -3.0]])))
    first = _project(world, POSE_A)
    second = _project(world, POSE_B)
    first = np.vstack((first, np.array([[np.nan, 10.0]])))
    second = np.vstack((second, np.array([[10.0, 10.0]])))
    second[3] += np.array([80.0, -40.0])

    result = triangulate_filtered(
        K,
        POSE_A,
        POSE_B,
        first,
        second,
        TriangulationConfig(min_points=8, max_reprojection_error=1.0),
    )

    assert 3 not in result.source_indices
    assert 12 not in result.source_indices
    assert 13 not in result.source_indices
    assert len(result.points) == 11


def test_filters_low_parallax_and_excess_depth() -> None:
    near = _good_points()[:12]
    far = np.array([[0.4, 0.2, 1000.0]])
    world = np.vstack((near, far))

    low_parallax = triangulate_filtered(
        K,
        POSE_A,
        POSE_B,
        _project(world, POSE_A),
        _project(world, POSE_B),
        TriangulationConfig(min_points=8, min_parallax_degrees=0.5, max_depth=2000),
    )
    assert 12 not in low_parallax.source_indices

    excess_depth = triangulate_filtered(
        K,
        POSE_A,
        POSE_B,
        _project(world, POSE_A),
        _project(world, POSE_B),
        TriangulationConfig(min_points=8, min_parallax_degrees=0.01, max_depth=50),
    )
    assert 12 not in excess_depth.source_indices


def test_initialization_rejects_low_texture_and_pure_rotation() -> None:
    intrinsics = CameraIntrinsics(700, 700, 320, 180, 640, 360)
    sparse = np.array([[20.0 + index, 30.0] for index in range(6)])
    with pytest.raises(SlamFailure) as sparse_error:
        initialize_two_view(intrinsics, sparse, sparse + 2)
    assert sparse_error.value.category is SlamErrorCategory.INITIALIZATION_FAILED

    world = _good_points()
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.12, 0.0]))
    rotation_pose = np.eye(4)
    rotation_pose[:3, :3] = rotation
    with pytest.raises(SlamFailure) as rotation_error:
        initialize_two_view(
            intrinsics,
            _project(world, POSE_A),
            _project(world, rotation_pose),
            min_landmarks=12,
        )
    assert rotation_error.value.category is SlamErrorCategory.INITIALIZATION_FAILED
