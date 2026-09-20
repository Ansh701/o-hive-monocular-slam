from __future__ import annotations

import math
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import numpy.typing as npt

from backend.app.config import Settings
from backend.slam.types import SlamErrorCategory, SlamFailure

ALLOWED_EXTENSIONS = frozenset({".mp4", ".mov", ".webm"})
ALLOWED_CODECS = frozenset({"mp4v", "fmp4", "avc1", "h264", "vp80", "vp90", "av01"})


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float
    codec: str
    size_bytes: int


@dataclass(frozen=True, slots=True)
class ProcessingProfile:
    long_edge: int = 640
    target_fps: float = 8.0
    max_frames: int = 240

    def __post_init__(self) -> None:
        if self.long_edge < 1 or self.target_fps <= 0 or self.max_frames < 2:
            raise ValueError("processing profile values must be positive and include two frames")


@dataclass(frozen=True, slots=True)
class SampledFrame:
    image: npt.NDArray[np.uint8]
    source_index: int
    timestamp_seconds: float


def _video_invalid(message: str, detail: str | None = None) -> SlamFailure:
    return SlamFailure(SlamErrorCategory.VIDEO_INVALID, message, internal_detail=detail)


def _decode_fourcc(value: float) -> str:
    integer = int(value)
    return "".join(chr((integer >> (8 * index)) & 0xFF) for index in range(4)).strip("\x00")


def sanitize_display_filename(filename: str) -> str:
    normalized = filename.replace("\\", "/")
    basename = normalized.rsplit("/", maxsplit=1)[-1]
    basename = re.sub(r"[\x00-\x1f\x7f]", "", basename).strip()
    basename = basename.lstrip(". ") or "video"
    suffix = Path(basename).suffix[:16]
    stem_limit = max(1, 120 - len(suffix))
    stem = Path(basename).stem[:stem_limit]
    return f"{stem}{suffix}"


def validate_and_probe(path: Path, settings: Settings) -> VideoMetadata:
    suffix = path.suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise _video_invalid("Unsupported video extension. Upload MP4, MOV, or WebM.")
    try:
        size_bytes = path.stat().st_size
    except OSError as exc:
        raise _video_invalid("The uploaded video could not be read.", str(exc)) from exc
    if size_bytes > settings.max_video_bytes:
        raise _video_invalid("The video size exceeds the configured upload limit.")
    if size_bytes < 16:
        raise _video_invalid("The video is too small to decode.")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise _video_invalid("The file could not decode as a video.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        codec = _decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)).lower()
        decoded, frame = capture.read()
        if not decoded or frame is None or frame.size == 0:
            raise _video_invalid("The video container opened but its frames could not decode.")
    finally:
        capture.release()

    if width <= 0 or height <= 0 or not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
        raise _video_invalid("The video metadata is invalid or incomplete.")
    if codec not in ALLOWED_CODECS:
        raise _video_invalid(f"The video codec '{codec or 'unknown'}' is not supported.")
    if width > settings.max_video_width or height > settings.max_video_height:
        raise _video_invalid("The video dimensions exceed the configured limit.")
    if frame_count > settings.max_video_frames:
        raise _video_invalid("The video contains too many frames.")

    duration_seconds = frame_count / fps
    if duration_seconds > settings.max_video_duration_seconds:
        raise _video_invalid("The video duration exceeds the configured limit.")
    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_seconds=duration_seconds,
        codec=codec,
        size_bytes=size_bytes,
    )


def _sample_indices(frame_count: int, source_fps: float, profile: ProcessingProfile) -> list[int]:
    step = max(1, round(source_fps / profile.target_fps))
    candidates = np.arange(0, frame_count, step, dtype=np.int64)
    if len(candidates) > profile.max_frames:
        selected = np.linspace(0, len(candidates) - 1, profile.max_frames, dtype=np.int64)
        candidates = candidates[selected]
    return [int(index) for index in candidates]


def iter_sampled_frames(path: Path, profile: ProcessingProfile) -> Iterator[SampledFrame]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise _video_invalid("The video could not decode for processing.")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(fps) or fps <= 0 or frame_count <= 0:
            raise _video_invalid("The video could not decode valid timing information.")

        for source_index in _sample_indices(frame_count, fps, profile):
            capture.set(cv2.CAP_PROP_POS_FRAMES, float(source_index))
            decoded, image = capture.read()
            if not decoded or image is None or image.size == 0:
                raise _video_invalid("A selected video frame could not decode.")
            height, width = image.shape[:2]
            longest = max(width, height)
            if longest > profile.long_edge:
                scale = profile.long_edge / longest
                image = cv2.resize(
                    image,
                    (max(1, round(width * scale)), max(1, round(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            yield SampledFrame(
                image=cast(npt.NDArray[np.uint8], image),
                source_index=source_index,
                timestamp_seconds=source_index / fps,
            )
    finally:
        capture.release()
