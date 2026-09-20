from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class KeyframePolicy:
    min_translation: float = 0.18
    min_rotation_degrees: float = 5.0
    min_parallax_pixels: float = 12.0
    low_tracking_landmarks: int = 50
    max_frame_gap: int = 10


@dataclass(slots=True)
class Keyframe:
    frame_index: int
    timestamp_seconds: float
    world_to_camera: npt.NDArray[np.float64]
    image_points: npt.NDArray[np.float64]
    descriptors: npt.NDArray[np.uint8] | None
    landmark_ids: npt.NDArray[np.int64]


def should_insert_keyframe(
    *,
    translation: float,
    rotation_degrees: float,
    median_parallax_pixels: float,
    tracked_landmarks: int,
    frames_since_keyframe: int,
    policy: KeyframePolicy,
) -> bool:
    return (
        translation >= policy.min_translation
        or rotation_degrees >= policy.min_rotation_degrees
        or median_parallax_pixels >= policy.min_parallax_pixels
        or tracked_landmarks <= policy.low_tracking_landmarks
        or frames_since_keyframe >= policy.max_frame_gap
    )


def build_keyframe(
    frame_index: int,
    timestamp_seconds: float,
    world_to_camera: npt.ArrayLike,
    image: npt.NDArray[np.uint8],
    *,
    max_features: int = 900,
) -> Keyframe:
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB.create(nfeatures=max_features, fastThreshold=12)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if not keypoints or descriptors is None:
        points = np.empty((0, 2), dtype=np.float64)
        descriptor_array = None
    else:
        points = np.array([keypoint.pt for keypoint in keypoints], dtype=np.float64)
        descriptor_array = cast(npt.NDArray[np.uint8], descriptors)
    return Keyframe(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        world_to_camera=np.asarray(world_to_camera, dtype=np.float64).reshape(4, 4).copy(),
        image_points=points,
        descriptors=descriptor_array,
        landmark_ids=np.full(len(points), -1, dtype=np.int64),
    )


def bounded_local_map(
    keyframes: list[Keyframe],
    *,
    max_keyframes: int = 5,
    max_landmarks: int = 600,
) -> npt.NDArray[np.int64]:
    selected: list[int] = []
    seen: set[int] = set()
    for keyframe in reversed(keyframes[-max_keyframes:]):
        for landmark_id in keyframe.landmark_ids:
            value = int(landmark_id)
            if value < 0 or value in seen:
                continue
            selected.append(value)
            seen.add(value)
            if len(selected) >= max_landmarks:
                return np.asarray(selected, dtype=np.int64)
    return np.asarray(selected, dtype=np.int64)
