from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix


@dataclass(frozen=True, slots=True)
class Observation:
    pose_index: int
    landmark_index: int
    pixel: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        pixel = np.asarray(self.pixel, dtype=np.float64).reshape(2).copy()
        pixel.setflags(write=False)
        object.__setattr__(self, "pixel", pixel)


@dataclass(frozen=True, slots=True)
class BundleWindow:
    poses: tuple[npt.NDArray[np.float64], ...]
    landmarks: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        poses = tuple(
            np.asarray(pose, dtype=np.float64).reshape(4, 4).copy() for pose in self.poses
        )
        landmarks = np.asarray(self.landmarks, dtype=np.float64).reshape(-1, 3).copy()
        object.__setattr__(self, "poses", poses)
        object.__setattr__(self, "landmarks", landmarks)


@dataclass(frozen=True, slots=True)
class BundleAdjustmentConfig:
    max_poses: int = 5
    max_landmarks: int = 120
    max_observations: int = 600
    max_nfev: int = 35
    robust_scale: float = 1.0
    min_improvement: float = 1e-3
    max_rotation_step_degrees: float = 12.0
    max_translation_step: float = 0.8
    max_landmark_step: float = 1.5


@dataclass(frozen=True, slots=True)
class BundleAdjustmentResult:
    window: BundleWindow
    before_median_error: float
    after_median_error: float
    accepted: bool
    observations_used: int
    landmarks_used: int
    poses_used: int


@dataclass(frozen=True, slots=True)
class _Problem:
    pose_indices: tuple[int, ...]
    landmark_indices: npt.NDArray[np.int64]
    observations: tuple[Observation, ...]


def _select_problem(
    window: BundleWindow,
    observations: list[Observation],
    config: BundleAdjustmentConfig,
) -> _Problem:
    first_pose = max(0, len(window.poses) - config.max_poses)
    pose_indices = tuple(range(first_pose, len(window.poses)))
    pose_set = set(pose_indices)
    valid = [
        observation
        for observation in observations
        if observation.pose_index in pose_set
        and 0 <= observation.landmark_index < len(window.landmarks)
        and np.isfinite(observation.pixel).all()
    ]
    counts: dict[int, int] = {}
    for observation in valid:
        counts[observation.landmark_index] = counts.get(observation.landmark_index, 0) + 1
    chosen_landmarks = sorted(counts, key=lambda index: (-counts[index], index))[
        : config.max_landmarks
    ]
    landmark_set = set(chosen_landmarks)
    chosen_observations = [
        observation for observation in valid if observation.landmark_index in landmark_set
    ][: config.max_observations]
    actually_used = sorted({observation.landmark_index for observation in chosen_observations})
    return _Problem(
        pose_indices=pose_indices,
        landmark_indices=np.asarray(actually_used, dtype=np.int64),
        observations=tuple(chosen_observations),
    )


def _encode(window: BundleWindow, problem: _Problem) -> npt.NDArray[np.float64]:
    values: list[float] = []
    for pose_index in problem.pose_indices[1:]:
        pose = window.poses[pose_index]
        rotation_vector, _ = cv2.Rodrigues(pose[:3, :3])
        values.extend(rotation_vector.reshape(3).tolist())
        values.extend(pose[:3, 3].tolist())
    values.extend(window.landmarks[problem.landmark_indices].reshape(-1).tolist())
    return np.asarray(values, dtype=np.float64)


def _decode(
    values: npt.NDArray[np.float64],
    original: BundleWindow,
    problem: _Problem,
) -> BundleWindow:
    poses = [pose.copy() for pose in original.poses]
    cursor = 0
    for pose_index in problem.pose_indices[1:]:
        rotation, _ = cv2.Rodrigues(values[cursor : cursor + 3])
        poses[pose_index][:3, :3] = rotation
        poses[pose_index][:3, 3] = values[cursor + 3 : cursor + 6]
        cursor += 6
    landmarks = original.landmarks.copy()
    landmark_values = values[cursor:].reshape(-1, 3)
    landmarks[problem.landmark_indices] = landmark_values
    return BundleWindow(tuple(poses), landmarks)


def _residuals(
    values: npt.NDArray[np.float64],
    original: BundleWindow,
    problem: _Problem,
    matrix: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    decoded = _decode(values, original, problem)
    residuals = np.empty((len(problem.observations), 2), dtype=np.float64)
    for index, observation in enumerate(problem.observations):
        pose = decoded.poses[observation.pose_index]
        point = decoded.landmarks[observation.landmark_index]
        camera = pose[:3, :3] @ point + pose[:3, 3]
        if not np.isfinite(camera).all() or camera[2] <= 1e-6:
            residuals[index] = np.array([1e4, 1e4])
            continue
        projected = matrix @ camera
        residuals[index] = projected[:2] / projected[2] - observation.pixel
    return residuals.reshape(-1)


def _median_error(residuals: npt.NDArray[np.float64]) -> float:
    if len(residuals) == 0:
        return float("inf")
    return float(np.median(np.linalg.norm(residuals.reshape(-1, 2), axis=1)))


def _parameter_bounds(
    initial: npt.NDArray[np.float64],
    variable_pose_count: int,
    config: BundleAdjustmentConfig,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    delta = np.full_like(initial, config.max_landmark_step)
    rotation_step = np.radians(config.max_rotation_step_degrees)
    for pose_index in range(variable_pose_count):
        offset = pose_index * 6
        delta[offset : offset + 3] = rotation_step
        delta[offset + 3 : offset + 6] = config.max_translation_step
    return initial - delta, initial + delta


def _jacobian_sparsity(problem: _Problem) -> lil_matrix:
    variable_poses = problem.pose_indices[1:]
    pose_offsets = {pose_index: index * 6 for index, pose_index in enumerate(variable_poses)}
    landmark_base = len(variable_poses) * 6
    landmark_offsets = {
        int(landmark_id): landmark_base + index * 3
        for index, landmark_id in enumerate(problem.landmark_indices)
    }
    sparsity = lil_matrix(
        (len(problem.observations) * 2, landmark_base + len(landmark_offsets) * 3)
    )
    for observation_index, observation in enumerate(problem.observations):
        rows = slice(observation_index * 2, observation_index * 2 + 2)
        pose_offset = pose_offsets.get(observation.pose_index)
        if pose_offset is not None:
            sparsity[rows, pose_offset : pose_offset + 6] = 1
        landmark_offset = landmark_offsets[observation.landmark_index]
        sparsity[rows, landmark_offset : landmark_offset + 3] = 1
    return sparsity


def bundle_adjust(
    window: BundleWindow,
    observations: list[Observation],
    matrix: npt.ArrayLike,
    config: BundleAdjustmentConfig | None = None,
) -> BundleAdjustmentResult:
    config = config or BundleAdjustmentConfig()
    intrinsic = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    problem = _select_problem(window, observations, config)
    initial = _encode(window, problem)
    before = _median_error(_residuals(initial, window, problem, intrinsic))
    counts = {
        "observations_used": len(problem.observations),
        "landmarks_used": len(problem.landmark_indices),
        "poses_used": len(problem.pose_indices),
    }
    if len(problem.observations) < 8 or len(problem.landmark_indices) < 4 or len(initial) == 0:
        return BundleAdjustmentResult(window, before, before, False, **counts)

    lower, upper = _parameter_bounds(initial, len(problem.pose_indices) - 1, config)
    optimized = least_squares(
        _residuals,
        initial,
        args=(window, problem, intrinsic),
        method="trf",
        loss="soft_l1",
        f_scale=config.robust_scale,
        max_nfev=config.max_nfev,
        bounds=(lower, upper),
        jac_sparsity=_jacobian_sparsity(problem),
    )
    candidate_values = np.asarray(optimized.x, dtype=np.float64)
    if not optimized.success or not np.isfinite(candidate_values).all():
        return BundleAdjustmentResult(window, before, before, False, **counts)
    candidate_residuals = _residuals(candidate_values, window, problem, intrinsic)
    after = _median_error(candidate_residuals)
    if not np.isfinite(after) or after >= before - config.min_improvement:
        return BundleAdjustmentResult(window, before, before, False, **counts)
    candidate = _decode(candidate_values, window, problem)
    return BundleAdjustmentResult(candidate, before, after, True, **counts)
