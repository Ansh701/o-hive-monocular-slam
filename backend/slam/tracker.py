from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from backend.slam.features import TrackedFeatures
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure


@dataclass(frozen=True, slots=True)
class PnPResult:
    world_to_camera: npt.NDArray[np.float64]
    inlier_indices: npt.NDArray[np.int64]
    median_reprojection_error: float

    @property
    def inlier_count(self) -> int:
        return len(self.inlier_indices)


@dataclass(frozen=True, slots=True)
class LandmarkAssociations:
    landmark_ids: npt.NDArray[np.int64]
    image_points: npt.NDArray[np.float64]
    tracking_errors: npt.NDArray[np.float64]


def estimate_pose_pnp(
    landmarks: npt.ArrayLike,
    image_points: npt.ArrayLike,
    intrinsics: CameraIntrinsics,
    *,
    min_inliers: int = 12,
    reprojection_error: float = 3.0,
    iterations: int = 100,
) -> PnPResult:
    object_points = np.asarray(landmarks, dtype=np.float64).reshape(-1, 3)
    pixels = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    finite = np.isfinite(object_points).all(axis=1) & np.isfinite(pixels).all(axis=1)
    source_indices = np.flatnonzero(finite).astype(np.int64)
    object_points = object_points[finite]
    pixels = pixels[finite]
    if len(object_points) < max(6, min_inliers):
        raise SlamFailure(
            SlamErrorCategory.TRACKING_LOST,
            "Too few mapped landmarks remain to estimate the camera pose.",
        )

    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points,
        pixels,
        intrinsics.matrix,
        None,
        iterationsCount=iterations,
        reprojectionError=reprojection_error,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not success or inliers is None or len(inliers) < min_inliers:
        raise SlamFailure(
            SlamErrorCategory.TRACKING_LOST,
            "The camera pose has too little landmark support.",
        )
    local_inliers = inliers.reshape(-1).astype(np.int64)
    rotation_vector, translation = cv2.solvePnPRefineLM(
        object_points[local_inliers],
        pixels[local_inliers],
        intrinsics.matrix,
        None,
        rotation_vector,
        translation,
    )
    rotation, _ = cv2.Rodrigues(rotation_vector)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)

    projected, _ = cv2.projectPoints(
        object_points[local_inliers],
        rotation_vector,
        translation,
        intrinsics.matrix,
        None,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - pixels[local_inliers], axis=1)
    return PnPResult(
        world_to_camera=pose,
        inlier_indices=source_indices[local_inliers],
        median_reprojection_error=float(np.median(errors)),
    )


def associate_landmarks(
    tracked: TrackedFeatures,
    prior_landmark_ids: npt.ArrayLike,
    *,
    max_landmarks: int,
) -> LandmarkAssociations:
    landmark_ids = np.asarray(prior_landmark_ids, dtype=np.int64).reshape(-1)
    if np.any(tracked.source_indices < 0) or np.any(tracked.source_indices >= len(landmark_ids)):
        raise ValueError("tracked feature source index is outside landmark-id bounds")
    propagated = landmark_ids[tracked.source_indices]
    mapped = propagated >= 0
    candidate_ids = propagated[mapped]
    candidate_points = tracked.current_points[mapped]
    candidate_errors = tracked.forward_backward_error[mapped]
    order = np.argsort(candidate_errors, kind="stable")[:max_landmarks]
    return LandmarkAssociations(
        landmark_ids=candidate_ids[order].astype(np.int64),
        image_points=candidate_points[order].astype(np.float64),
        tracking_errors=candidate_errors[order].astype(np.float64),
    )
