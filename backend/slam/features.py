from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class TrackedFeatures:
    previous_points: npt.NDArray[np.float64]
    current_points: npt.NDArray[np.float64]
    source_indices: npt.NDArray[np.int64]
    forward_backward_error: npt.NDArray[np.float64]


def _gray(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if image.ndim == 2:
        return cast(npt.NDArray[np.uint8], image.astype(np.uint8, copy=False))
    return cast(npt.NDArray[np.uint8], cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))


def detect_corners(
    image: npt.NDArray[np.uint8],
    *,
    max_features: int = 900,
    quality_level: float = 0.01,
    min_distance: float = 7.0,
) -> npt.NDArray[np.float64]:
    corners = cv2.goodFeaturesToTrack(
        _gray(image),
        maxCorners=max_features,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=7,
        useHarrisDetector=False,
    )
    if corners is None:
        return np.empty((0, 2), dtype=np.float64)
    return corners.reshape(-1, 2).astype(np.float64)


def track_lk_forward_backward(
    previous_image: npt.NDArray[np.uint8],
    current_image: npt.NDArray[np.uint8],
    previous_points: npt.ArrayLike,
    *,
    max_error: float = 1.0,
) -> TrackedFeatures:
    points = np.asarray(previous_points, dtype=np.float32).reshape(-1, 1, 2)
    empty_points = np.empty((0, 2), dtype=np.float64)
    empty_indices = np.empty((0,), dtype=np.int64)
    empty_result = TrackedFeatures(
        empty_points,
        empty_points.copy(),
        empty_indices,
        np.empty((0,), dtype=np.float64),
    )
    if len(points) == 0:
        return empty_result

    current_points, forward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        _gray(previous_image),
        _gray(current_image),
        points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if current_points is None or forward_status is None:
        return empty_result
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(  # type: ignore[call-overload]
        _gray(current_image),
        _gray(previous_image),
        current_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward_points is None or backward_status is None:
        return empty_result

    source = points.reshape(-1, 2).astype(np.float64)
    target = current_points.reshape(-1, 2).astype(np.float64)
    backward = backward_points.reshape(-1, 2).astype(np.float64)
    errors = np.linalg.norm(source - backward, axis=1)
    height, width = current_image.shape[:2]
    valid = (
        forward_status.reshape(-1).astype(bool)
        & backward_status.reshape(-1).astype(bool)
        & np.isfinite(target).all(axis=1)
        & np.isfinite(errors)
        & (errors <= max_error)
        & (target[:, 0] >= 0)
        & (target[:, 0] < width)
        & (target[:, 1] >= 0)
        & (target[:, 1] < height)
    )
    indices = np.flatnonzero(valid).astype(np.int64)
    return TrackedFeatures(source[valid], target[valid], indices, errors[valid])
