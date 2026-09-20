from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.slam.keyframes import (
    KeyframePolicy,
    bounded_local_map,
    build_keyframe,
    should_insert_keyframe,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"translation": 0.21},
        {"rotation_degrees": 6.0},
        {"median_parallax_pixels": 13.0},
        {"tracked_landmarks": 35},
        {"frames_since_keyframe": 12},
    ],
)
def test_keyframe_policy_has_motion_parallax_tracking_and_temporal_triggers(
    overrides: dict[str, float | int],
) -> None:
    metrics: dict[str, float | int] = {
        "translation": 0.02,
        "rotation_degrees": 0.5,
        "median_parallax_pixels": 2.0,
        "tracked_landmarks": 140,
        "frames_since_keyframe": 2,
    }
    metrics.update(overrides)
    assert should_insert_keyframe(policy=KeyframePolicy(), **metrics)


def test_keyframe_policy_does_not_insert_without_a_trigger() -> None:
    assert not should_insert_keyframe(
        translation=0.02,
        rotation_degrees=0.5,
        median_parallax_pixels=2.0,
        tracked_landmarks=140,
        frames_since_keyframe=2,
        policy=KeyframePolicy(),
    )


def test_keyframe_builds_bounded_orb_descriptors_and_local_map() -> None:
    image = np.zeros((180, 320), dtype=np.uint8)
    for y in range(12, 170, 16):
        for x in range(12, 310, 16):
            cv2.rectangle(image, (x - 3, y - 3), (x + 3, y + 3), 255, -1)
    first = build_keyframe(0, 0.0, np.eye(4), image, max_features=80)
    second = build_keyframe(8, 0.8, np.eye(4), image, max_features=80)
    first.landmark_ids = np.arange(len(first.image_points), dtype=np.int64)
    second.landmark_ids = np.arange(40, 40 + len(second.image_points), dtype=np.int64)

    assert first.descriptors is not None
    assert 10 <= len(first.image_points) <= 80
    assert len(first.descriptors) == len(first.image_points)

    selected = bounded_local_map([first, second], max_keyframes=2, max_landmarks=50)
    assert len(selected) == 50
    assert selected[-1] >= 40
