from __future__ import annotations

from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.slam.pipeline import SlamPipeline
from backend.slam.types import SlamErrorCategory, SlamFailure
from scripts.generate_test_videos import generate_video


@pytest.fixture(scope="module")
def pipeline_videos(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("pipeline-videos")
    videos = {
        "good": root / "good.mp4",
        "low": root / "low.mp4",
        "lost": root / "lost.mp4",
    }
    generate_video(videos["good"], mode="translation", duration_seconds=2.5, width=640, height=360)
    generate_video(videos["low"], mode="low_texture", duration_seconds=2, width=640, height=360)
    generate_video(videos["lost"], mode="tracking_loss", duration_seconds=4, width=640, height=360)
    return videos


def _settings() -> Settings:
    return Settings(
        processing_long_edge=480,
        processing_fps=8,
        max_processed_frames=80,
        max_features=700,
        min_initial_landmarks=10,
        min_pnp_landmarks=8,
    )


def test_pipeline_reconstructs_real_sparse_geometry(
    pipeline_videos: dict[str, Path],
) -> None:
    events: list[tuple[str, int, int]] = []
    result = SlamPipeline(_settings()).process(
        pipeline_videos["good"],
        progress=lambda stage, completed, total: events.append((stage, completed, total)),
    )

    assert result.status == "completed"
    assert len(result.poses) >= 8
    assert len(result.landmarks) >= 10
    assert result.scale == "arbitrary"
    assert {event[0] for event in events} >= {"decoding", "initializing", "tracking", "optimizing"}
    assert result.processing_seconds > 0


def test_pipeline_fails_honestly_on_low_texture(pipeline_videos: dict[str, Path]) -> None:
    with pytest.raises(SlamFailure) as error:
        SlamPipeline(_settings()).process(pipeline_videos["low"])
    assert error.value.category is SlamErrorCategory.INITIALIZATION_FAILED


def test_pipeline_returns_partial_map_after_tracking_loss(
    pipeline_videos: dict[str, Path],
) -> None:
    result = SlamPipeline(_settings()).process(pipeline_videos["lost"])
    assert result.status == "partial"
    assert len(result.poses) >= 2
    assert len(result.landmarks) >= 10


def test_pipeline_rejects_invalid_video(tmp_path: Path) -> None:
    invalid = tmp_path / "fake.mp4"
    invalid.write_bytes(b"not a video, despite the extension")
    with pytest.raises(SlamFailure) as error:
        SlamPipeline(_settings()).process(invalid)
    assert error.value.category is SlamErrorCategory.VIDEO_INVALID
