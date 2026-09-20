from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import backend.slam.bundle_adjustment as ba_module
from backend.slam.bundle_adjustment import (
    BundleAdjustmentConfig,
    BundleWindow,
    Observation,
    bundle_adjust,
)

K = np.array([[700.0, 0.0, 320.0], [0.0, 700.0, 180.0], [0.0, 0.0, 1.0]])


def _project(point: np.ndarray, pose: np.ndarray) -> np.ndarray:
    camera = pose[:3, :3] @ point + pose[:3, 3]
    pixel = K @ camera
    return pixel[:2] / pixel[2]


def _problem() -> tuple[BundleWindow, list[Observation]]:
    rng = np.random.default_rng(88)
    truth_points = np.column_stack(
        (rng.uniform(-2, 2, 30), rng.uniform(-1, 1, 30), rng.uniform(5, 9, 30))
    )
    true_poses = []
    for translation_x in (0.0, -0.25, -0.5):
        pose = np.eye(4)
        pose[0, 3] = translation_x
        true_poses.append(pose)
    observations = [
        Observation(pose_index, landmark_index, _project(point, true_poses[pose_index]))
        for pose_index in range(3)
        for landmark_index, point in enumerate(truth_points)
    ]

    noisy_poses = [pose.copy() for pose in true_poses]
    noisy_poses[1][:3, :3], _ = cv2.Rodrigues(np.array([0.006, -0.012, 0.004]))
    noisy_poses[1][:3, 3] += np.array([0.04, -0.02, 0.01])
    noisy_poses[2][:3, :3], _ = cv2.Rodrigues(np.array([-0.004, 0.015, -0.006]))
    noisy_poses[2][:3, 3] += np.array([-0.035, 0.025, -0.012])
    noisy_points = truth_points + rng.normal(0, 0.025, truth_points.shape)
    return BundleWindow(tuple(noisy_poses), noisy_points), observations


def test_bundle_adjustment_lowers_error_and_fixes_oldest_pose() -> None:
    window, observations = _problem()
    oldest_before = window.poses[0].copy()

    result = bundle_adjust(
        window,
        observations,
        K,
        BundleAdjustmentConfig(max_nfev=80, max_landmarks=40, max_observations=200),
    )

    assert result.accepted is True
    assert result.after_median_error < result.before_median_error * 0.5
    assert np.array_equal(result.window.poses[0], oldest_before)
    assert result.observations_used == 90
    assert result.landmarks_used == 30


def test_bundle_adjustment_respects_problem_caps() -> None:
    window, observations = _problem()
    result = bundle_adjust(
        window,
        observations,
        K,
        BundleAdjustmentConfig(
            max_nfev=5,
            max_poses=2,
            max_landmarks=8,
            max_observations=12,
        ),
    )

    assert result.observations_used <= 12
    assert result.landmarks_used <= 8
    assert result.poses_used <= 2


def test_nonfinite_optimizer_output_is_rejected_transactionally(
    monkeypatch,
) -> None:
    window, observations = _problem()

    @dataclass
    class FakeResult:
        x: np.ndarray
        success: bool = True

    def fake_least_squares(function, initial, **kwargs):
        del function, kwargs
        return FakeResult(np.full_like(initial, np.nan))

    monkeypatch.setattr(ba_module, "least_squares", fake_least_squares)
    result = bundle_adjust(window, observations, K, BundleAdjustmentConfig())

    assert result.accepted is False
    assert result.after_median_error == result.before_median_error
    assert all(
        np.array_equal(after, before)
        for after, before in zip(result.window.poses, window.poses, strict=True)
    )
    assert np.array_equal(result.window.landmarks, window.landmarks)


def test_worse_finite_optimizer_output_is_rejected(monkeypatch) -> None:
    window, observations = _problem()

    @dataclass
    class FakeResult:
        x: np.ndarray
        success: bool = True

    def fake_least_squares(function, initial, **kwargs):
        del function, kwargs
        return FakeResult(initial + 50.0)

    monkeypatch.setattr(ba_module, "least_squares", fake_least_squares)
    result = bundle_adjust(window, observations, K, BundleAdjustmentConfig())

    assert result.accepted is False
    assert result.after_median_error == result.before_median_error
