from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from backend.app.config import Settings
from backend.slam.types import (
    CameraIntrinsics,
    FrameDiagnostics,
    Landmark,
    Pose,
    SlamErrorCategory,
    SlamFailure,
    SlamResult,
)


def test_intrinsics_reject_non_physical_values() -> None:
    with pytest.raises(ValueError, match="focal"):
        CameraIntrinsics(fx=0, fy=500, cx=320, cy=240, width=640, height=480)

    with pytest.raises(ValueError, match="principal"):
        CameraIntrinsics(fx=500, fy=500, cx=900, cy=240, width=640, height=480)


def test_pose_requires_finite_rigid_four_by_four_matrix() -> None:
    with pytest.raises(ValueError, match="4x4"):
        Pose(world_to_camera=np.eye(3))

    invalid_bottom_row = np.eye(4)
    invalid_bottom_row[3, 3] = 2
    with pytest.raises(ValueError, match="homogeneous"):
        Pose(world_to_camera=invalid_bottom_row)

    non_finite = np.eye(4)
    non_finite[0, 3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        Pose(world_to_camera=non_finite)


def test_result_serialization_bounds_rendered_points_without_lying_about_total() -> None:
    landmarks = tuple(
        Landmark(id=index, position=np.array([float(index), 0.5, 3.0]), observations=3)
        for index in range(10)
    )
    result = SlamResult(
        poses=(Pose.identity(frame_index=0, timestamp_seconds=0),),
        landmarks=landmarks,
        diagnostics=(FrameDiagnostics(frame_index=0, tracked_features=120, inliers=96),),
        processing_seconds=0.42,
        status="completed",
        scale="arbitrary",
    )

    payload = result.to_public_dict(render_point_limit=4)

    assert payload["point_count"] == 10
    assert payload["rendered_point_count"] == 4
    assert [point["id"] for point in payload["points"]] == [0, 3, 6, 9]
    assert payload["scale"] == "arbitrary"


def test_failure_exposes_only_allowlisted_category_and_safe_message() -> None:
    failure = SlamFailure(
        SlamErrorCategory.TRACKING_LOST,
        "Tracking was lost after frame 18. Try a slower, better-lit video.",
        internal_detail="private video path; database password=secret",
    )

    assert failure.to_public_dict() == {
        "code": "TRACKING_LOST",
        "message": "Tracking was lost after frame 18. Try a slower, better-lit video.",
    }
    assert "private" not in str(failure)
    assert "secret" not in str(failure)


def test_settings_enforce_cost_and_resource_limits() -> None:
    settings = Settings(
        max_video_bytes=10_000,
        max_video_duration_seconds=10,
        max_video_width=1920,
        max_video_height=1080,
        max_video_frames=600,
        processing_timeout_seconds=20,
        max_concurrent_runs=1,
        rate_limit_runs_per_minute=2,
        render_point_limit=1000,
    )
    assert settings.max_total_pixels == 2_073_600

    with pytest.raises(ValidationError):
        Settings(max_concurrent_runs=0)
    with pytest.raises(ValidationError):
        Settings(render_point_limit=100_001)
