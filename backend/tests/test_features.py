from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.slam.features import detect_corners, track_lk_forward_backward


def _dot_image() -> np.ndarray:
    image = np.zeros((120, 160), dtype=np.uint8)
    for y in range(15, 110, 18):
        for x in range(15, 155, 18):
            cv2.rectangle(image, (x - 2, y - 2), (x + 2, y + 2), 255, -1)
    return image


def test_corner_detection_is_bounded_and_in_image() -> None:
    points = detect_corners(_dot_image(), max_features=25, quality_level=0.01, min_distance=5)

    assert 12 <= len(points) <= 25
    assert points.shape[1] == 2
    assert np.all((points[:, 0] >= 0) & (points[:, 0] < 160))
    assert np.all((points[:, 1] >= 0) & (points[:, 1] < 120))


def test_forward_backward_tracking_rejects_out_of_bounds_and_recovers_translation() -> None:
    previous = _dot_image()
    transform = np.float32([[1, 0, 5], [0, 1, 3]])
    current = cv2.warpAffine(previous, transform, (160, 120))
    points = np.vstack((detect_corners(previous, max_features=60), np.array([[159.0, 119.0]])))

    tracked = track_lk_forward_backward(previous, current, points, max_error=0.8)

    displacement = tracked.current_points - tracked.previous_points
    assert len(tracked.current_points) >= 20
    assert np.median(displacement, axis=0).tolist() == pytest.approx([5.0, 3.0], abs=0.25)
    assert len(points) - 1 not in tracked.source_indices
    assert np.all(tracked.forward_backward_error <= 0.8)
