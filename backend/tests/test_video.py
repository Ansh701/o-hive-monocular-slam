from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.config import Settings
from backend.slam.types import SlamErrorCategory, SlamFailure
from backend.slam.video import (
    ProcessingProfile,
    iter_sampled_frames,
    sanitize_display_filename,
    validate_and_probe,
)


def _write_video(
    path: Path,
    *,
    codec: str = "mp4v",
    width: int = 160,
    height: int = 120,
    fps: float = 10.0,
    frames: int = 12,
) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    assert writer.isOpened(), f"test codec {codec} is unavailable"
    for index in range(frames):
        image = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.circle(image, (20 + index * 3, 50), 8, (50, 220, 240), -1)
        cv2.putText(image, str(index), (5, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255))
        writer.write(image)
    writer.release()


@pytest.fixture
def valid_video(tmp_path: Path) -> Path:
    path = tmp_path / "valid.mp4"
    _write_video(path)
    return path


def test_valid_mp4_is_probed_from_decoded_content(valid_video: Path) -> None:
    metadata = validate_and_probe(valid_video, Settings())

    assert metadata.width == 160
    assert metadata.height == 120
    assert metadata.frame_count == 12
    assert metadata.fps == pytest.approx(10.0, abs=0.1)
    assert metadata.duration_seconds == pytest.approx(1.2, abs=0.1)
    assert metadata.codec.lower() in {"mp4v", "fmp4"}


def test_wrong_extension_and_fake_video_are_rejected(tmp_path: Path) -> None:
    wrong_extension = tmp_path / "clip.avi"
    _write_video(wrong_extension, codec="MJPG")
    with pytest.raises(SlamFailure) as extension_error:
        validate_and_probe(wrong_extension, Settings())
    assert extension_error.value.category is SlamErrorCategory.VIDEO_INVALID

    fake = tmp_path / "fake.mp4"
    fake.write_bytes(b"<html>not a video</html>")
    with pytest.raises(SlamFailure, match="decode"):
        validate_and_probe(fake, Settings())


def test_unsupported_codec_is_rejected_even_with_allowed_extension(tmp_path: Path) -> None:
    source = tmp_path / "mjpeg.avi"
    disguised = tmp_path / "mjpeg.mp4"
    _write_video(source, codec="MJPG")
    source.replace(disguised)

    with pytest.raises(SlamFailure, match="codec") as error:
        validate_and_probe(disguised, Settings())
    assert error.value.category is SlamErrorCategory.VIDEO_INVALID


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"max_video_bytes": 100}, "size"),
        ({"max_video_duration_seconds": 0.5}, "duration"),
        ({"max_video_width": 100}, "dimensions"),
        ({"max_video_frames": 5}, "frames"),
    ],
)
def test_resource_limits_reject_video(
    valid_video: Path, settings: dict[str, int | float], message: str
) -> None:
    with pytest.raises(SlamFailure, match=message):
        validate_and_probe(valid_video, Settings(**settings))


def test_display_filename_strips_paths_controls_and_overlong_names() -> None:
    cleaned = sanitize_display_filename("../../private\\..\\\x00  card.mov")
    assert cleaned == "card.mov"
    assert sanitize_display_filename("a" * 300 + ".mp4") == "a" * 116 + ".mp4"


def test_sampling_is_bounded_resized_and_monotonic(valid_video: Path) -> None:
    profile = ProcessingProfile(long_edge=80, target_fps=5, max_frames=4)

    frames = list(iter_sampled_frames(valid_video, profile))

    assert len(frames) == 4
    assert all(frame.image.shape == (60, 80, 3) for frame in frames)
    assert [frame.source_index for frame in frames] == sorted(
        frame.source_index for frame in frames
    )
    assert [frame.timestamp_seconds for frame in frames] == sorted(
        frame.timestamp_seconds for frame in frames
    )


def test_decode_failure_after_probe_is_typed(
    valid_video: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class BrokenCapture:
        def isOpened(self) -> bool:
            return True

        def get(self, property_id: int) -> float:
            values = {
                cv2.CAP_PROP_FRAME_WIDTH: 160,
                cv2.CAP_PROP_FRAME_HEIGHT: 120,
                cv2.CAP_PROP_FPS: 10,
                cv2.CAP_PROP_FRAME_COUNT: 12,
            }
            return float(values.get(property_id, 0))

        def set(self, property_id: int, value: float) -> bool:
            return True

        def read(self) -> tuple[bool, None]:
            return False, None

        def release(self) -> None:
            return None

    monkeypatch.setattr(cv2, "VideoCapture", lambda _: BrokenCapture())

    with pytest.raises(SlamFailure, match="decode") as error:
        list(iter_sampled_frames(valid_video, ProcessingProfile()))
    assert error.value.category is SlamErrorCategory.VIDEO_INVALID
