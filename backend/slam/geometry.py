from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


@dataclass(frozen=True, slots=True)
class RelativePose:
    rotation: npt.NDArray[np.float64]
    translation: npt.NDArray[np.float64]
    inlier_mask: npt.NDArray[np.bool_]

    @property
    def inlier_count(self) -> int:
        return int(np.count_nonzero(self.inlier_mask))


def estimate_relative_pose(
    previous_points: npt.ArrayLike,
    current_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
    *,
    min_inliers: int = 24,
    ransac_threshold_pixels: float = 1.5,
) -> RelativePose:
    previous = np.asarray(previous_points, dtype=np.float64).reshape(-1, 2)
    current = np.asarray(current_points, dtype=np.float64).reshape(-1, 2)
    if len(previous) != len(current) or len(previous) < max(8, min_inliers):
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "Not enough reliable feature matches to initialize the map.",
        )
    essential, ransac_mask = cv2.findEssentialMat(
        previous,
        current,
        intrinsics.matrix,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=ransac_threshold_pixels,
    )
    if essential is None or ransac_mask is None:
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "Camera motion could not be estimated from this video.",
        )
    essential = np.asarray(essential, dtype=np.float64)[:3, :3]
    inlier_count, rotation, translation, pose_mask = cv2.recoverPose(
        essential,
        previous,
        current,
        intrinsics.matrix,
        mask=ransac_mask,
    )
    mask = pose_mask.reshape(-1).astype(bool)
    if inlier_count < min_inliers or int(np.count_nonzero(mask)) < min_inliers:
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "Camera motion has too little geometric support to initialize safely.",
        )
    vector = np.asarray(translation, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "Camera translation is too small to initialize a 3D map.",
        )
    return RelativePose(
        rotation=np.asarray(rotation, dtype=np.float64),
        translation=vector / norm,
        inlier_mask=mask,
    )


def compose_world_to_camera(
    previous_world_to_camera: npt.ArrayLike,
    relative_rotation: npt.ArrayLike,
    relative_translation: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    previous = np.asarray(previous_world_to_camera, dtype=np.float64)
    if previous.shape != (4, 4):
        raise ValueError("previous_world_to_camera must be 4x4")
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = np.asarray(relative_rotation, dtype=np.float64).reshape(3, 3)
    relative[:3, 3] = np.asarray(relative_translation, dtype=np.float64).reshape(3)
    return relative @ previous


def camera_center(world_to_camera: npt.ArrayLike) -> npt.NDArray[np.float64]:
    pose = np.asarray(world_to_camera, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError("world_to_camera must be 4x4")
    rotation = pose[:3, :3]
    translation = pose[:3, 3]
    return -rotation.T @ translation
