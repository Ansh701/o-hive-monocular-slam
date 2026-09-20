from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from backend.slam.geometry import camera_center, compose_world_to_camera, estimate_relative_pose
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


@dataclass(frozen=True, slots=True)
class TriangulationConfig:
    min_parallax_degrees: float = 0.5
    max_reprojection_error: float = 2.0
    min_depth: float = 0.05
    max_depth: float = 100.0
    min_points: int = 12


@dataclass(frozen=True, slots=True)
class TriangulationResult:
    points: npt.NDArray[np.float64]
    source_indices: npt.NDArray[np.int64]
    reprojection_errors: npt.NDArray[np.float64]
    parallax_degrees: npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class InitializationResult:
    pose_a: npt.NDArray[np.float64]
    pose_b: npt.NDArray[np.float64]
    triangulation: TriangulationResult
    inlier_mask: npt.NDArray[np.bool_]


def _project(
    matrix: npt.NDArray[np.float64],
    pose: npt.NDArray[np.float64],
    points: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    camera_points = (pose[:3, :3] @ points.T + pose[:3, 3:4]).T
    homogeneous = (matrix @ camera_points.T).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:]
    return pixels, camera_points[:, 2]


def triangulate_filtered(
    matrix: npt.ArrayLike,
    pose_a: npt.ArrayLike,
    pose_b: npt.ArrayLike,
    points_a: npt.ArrayLike,
    points_b: npt.ArrayLike,
    config: TriangulationConfig | None = None,
) -> TriangulationResult:
    config = config or TriangulationConfig()
    intrinsic = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    first_pose = np.asarray(pose_a, dtype=np.float64).reshape(4, 4)
    second_pose = np.asarray(pose_b, dtype=np.float64).reshape(4, 4)
    first_pixels = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    second_pixels = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if len(first_pixels) != len(second_pixels):
        raise ValueError("point correspondence arrays must have equal length")

    finite_correspondence = np.isfinite(first_pixels).all(axis=1) & np.isfinite(
        second_pixels
    ).all(axis=1)
    source_indices = np.flatnonzero(finite_correspondence).astype(np.int64)
    first_finite = first_pixels[finite_correspondence]
    second_finite = second_pixels[finite_correspondence]
    if len(first_finite) < config.min_points:
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "Too few finite correspondences remain for triangulation.",
        )

    projection_a = intrinsic @ first_pose[:3]
    projection_b = intrinsic @ second_pose[:3]
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        first_finite.T,
        second_finite.T,
    )
    w = homogeneous[3]
    valid_w = np.isfinite(homogeneous).all(axis=0) & (np.abs(w) > 1e-12)
    points = np.full((len(first_finite), 3), np.nan, dtype=np.float64)
    points[valid_w] = (homogeneous[:3, valid_w] / w[valid_w]).T

    reproject_a, depth_a = _project(intrinsic, first_pose, points)
    reproject_b, depth_b = _project(intrinsic, second_pose, points)
    error_a = np.linalg.norm(reproject_a - first_finite, axis=1)
    error_b = np.linalg.norm(reproject_b - second_finite, axis=1)
    reprojection_error = np.maximum(error_a, error_b)

    center_a = camera_center(first_pose)
    center_b = camera_center(second_pose)
    ray_a = points - center_a
    ray_b = points - center_b
    denominator = np.linalg.norm(ray_a, axis=1) * np.linalg.norm(ray_b, axis=1)
    cosine = np.divide(
        np.sum(ray_a * ray_b, axis=1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > 1e-12,
    )
    parallax = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

    valid = (
        valid_w
        & np.isfinite(points).all(axis=1)
        & np.isfinite(reprojection_error)
        & np.isfinite(parallax)
        & (depth_a > config.min_depth)
        & (depth_b > config.min_depth)
        & (depth_a < config.max_depth)
        & (depth_b < config.max_depth)
        & (reprojection_error <= config.max_reprojection_error)
        & (parallax >= config.min_parallax_degrees)
    )
    if int(np.count_nonzero(valid)) < config.min_points:
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "The video has insufficient parallax for a reliable 3D map.",
        )
    return TriangulationResult(
        points=points[valid],
        source_indices=source_indices[valid],
        reprojection_errors=reprojection_error[valid],
        parallax_degrees=parallax[valid],
    )


def initialize_two_view(
    intrinsics: CameraIntrinsics,
    points_a: npt.ArrayLike,
    points_b: npt.ArrayLike,
    *,
    min_landmarks: int = 12,
    config: TriangulationConfig | None = None,
) -> InitializationResult:
    first = np.asarray(points_a, dtype=np.float64).reshape(-1, 2)
    second = np.asarray(points_b, dtype=np.float64).reshape(-1, 2)
    if len(first) < max(8, min_landmarks):
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "This video has too little texture to initialize a 3D map.",
        )
    relative = estimate_relative_pose(
        first,
        second,
        intrinsics,
        min_inliers=max(8, min_landmarks),
    )
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = compose_world_to_camera(pose_a, relative.rotation, relative.translation)
    inlier_indices = np.flatnonzero(relative.inlier_mask)
    triangulation = triangulate_filtered(
        intrinsics.matrix,
        pose_a,
        pose_b,
        first[relative.inlier_mask],
        second[relative.inlier_mask],
        config or TriangulationConfig(min_points=min_landmarks),
    )
    source_indices = inlier_indices[triangulation.source_indices]
    triangulation = TriangulationResult(
        points=triangulation.points,
        source_indices=source_indices.astype(np.int64),
        reprojection_errors=triangulation.reprojection_errors,
        parallax_degrees=triangulation.parallax_degrees,
    )
    return InitializationResult(pose_a, pose_b, triangulation, relative.inlier_mask)
