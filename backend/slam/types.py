from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _float_array(value: npt.ArrayLike, shape: tuple[int, ...], label: str) -> FloatArray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{label} must have shape {shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    copied = array.copy()
    copied.setflags(write=False)
    return copied


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    approximate: bool = False

    def __post_init__(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(np.isfinite(value) for value in values):
            raise ValueError("intrinsics must be finite")
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")
        if not (0 <= self.cx <= self.width and 0 <= self.cy <= self.height):
            raise ValueError("principal point must be inside the image")

    @property
    def matrix(self) -> FloatArray:
        matrix = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        matrix.setflags(write=False)
        return matrix


@dataclass(frozen=True, slots=True)
class Pose:
    world_to_camera: FloatArray
    frame_index: int = 0
    timestamp_seconds: float = 0.0
    keyframe: bool = False

    def __post_init__(self) -> None:
        matrix = _float_array(self.world_to_camera, (4, 4), "world_to_camera 4x4 matrix")
        if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-9):
            raise ValueError("world_to_camera must have a homogeneous bottom row")
        object.__setattr__(self, "world_to_camera", matrix)

    @classmethod
    def identity(cls, frame_index: int, timestamp_seconds: float) -> Pose:
        return cls(np.eye(4, dtype=np.float64), frame_index, timestamp_seconds, True)


@dataclass(frozen=True, slots=True)
class Landmark:
    id: int
    position: FloatArray
    observations: int
    reprojection_error: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "position", _float_array(self.position, (3,), "position"))
        if self.observations < 1:
            raise ValueError("observations must be positive")


@dataclass(frozen=True, slots=True)
class FrameDiagnostics:
    frame_index: int
    tracked_features: int
    inliers: int
    reprojection_error: float | None = None
    stage: str = "tracking"


class SlamErrorCategory(StrEnum):
    VIDEO_INVALID = "VIDEO_INVALID"
    INITIALIZATION_FAILED = "INITIALIZATION_FAILED"
    TRACKING_LOST = "TRACKING_LOST"
    PROCESSING_TIMEOUT = "PROCESSING_TIMEOUT"
    RATE_LIMITED = "RATE_LIMITED"
    INTERNAL_ERROR = "INTERNAL_ERROR"


class SlamFailure(Exception):
    """Typed operational failure with separate public and diagnostic detail."""

    def __init__(
        self,
        category: SlamErrorCategory,
        public_message: str,
        *,
        internal_detail: str | None = None,
    ) -> None:
        self.category = category
        self.public_message = public_message
        self.internal_detail = internal_detail
        super().__init__(public_message)

    def __str__(self) -> str:
        return f"{self.category.value}: {self.public_message}"

    def to_public_dict(self) -> dict[str, str]:
        return {"code": self.category.value, "message": self.public_message}


@dataclass(frozen=True, slots=True)
class SlamResult:
    poses: tuple[Pose, ...]
    landmarks: tuple[Landmark, ...]
    diagnostics: tuple[FrameDiagnostics, ...]
    processing_seconds: float
    status: Literal["completed", "partial"]
    scale: Literal["arbitrary"] = "arbitrary"

    def to_public_dict(self, render_point_limit: int) -> dict[str, Any]:
        if render_point_limit < 1:
            raise ValueError("render_point_limit must be positive")
        total = len(self.landmarks)
        if total <= render_point_limit:
            selected = self.landmarks
        else:
            indices = np.linspace(0, total - 1, render_point_limit, dtype=np.int64)
            selected = tuple(self.landmarks[int(index)] for index in indices)

        return {
            "status": self.status,
            "scale": self.scale,
            "processing_seconds": self.processing_seconds,
            "pose_count": len(self.poses),
            "point_count": total,
            "rendered_point_count": len(selected),
            "points": [
                {
                    "id": landmark.id,
                    "position": landmark.position.tolist(),
                    "observations": landmark.observations,
                    "reprojection_error": landmark.reprojection_error,
                }
                for landmark in selected
            ],
            "poses": [
                {
                    "frame_index": pose.frame_index,
                    "timestamp_seconds": pose.timestamp_seconds,
                    "keyframe": pose.keyframe,
                    "world_to_camera": pose.world_to_camera.tolist(),
                }
                for pose in self.poses
            ],
            "diagnostics": [
                {
                    "frame_index": diagnostic.frame_index,
                    "tracked_features": diagnostic.tracked_features,
                    "inliers": diagnostic.inliers,
                    "reprojection_error": diagnostic.reprojection_error,
                    "stage": diagnostic.stage,
                }
                for diagnostic in self.diagnostics
            ],
        }
