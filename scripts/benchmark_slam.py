from __future__ import annotations

import argparse
import json
import platform
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.app.config import Settings
from backend.slam.pipeline import SlamPipeline
from backend.slam.video import validate_and_probe
from scripts.generate_test_videos import generate_video


@dataclass
class BenchmarkMetrics:
    source: str
    video_duration_seconds: float
    original_resolution: str
    original_fps: float
    original_frame_count: int
    processing_resolution: str
    effective_processing_fps: float
    frames_decoded: int
    frames_processed: int
    processing_seconds: float
    total_end_to_end_seconds: float
    processing_ratio: float
    pose_count: int
    point_count: int
    keyframe_count: int
    median_reprojection_error_px: float | None
    status: str
    environment: str
    generated_at: str


def evaluate_benchmark(
    metrics: BenchmarkMetrics, *, threshold_seconds: float
) -> dict[str, Any]:
    report = asdict(metrics)
    report["threshold_seconds"] = threshold_seconds
    report["passed"] = (
        metrics.status.upper() in {"COMPLETED", "PARTIAL"}
        and metrics.processing_seconds <= threshold_seconds
        and metrics.pose_count >= 2
        and metrics.point_count > 0
    )
    return report


def run_benchmark(video_path: Path, *, environment: str) -> BenchmarkMetrics:
    if not video_path.exists():
        generate_video(video_path, mode="benchmark", duration_seconds=10.0)
    settings = Settings(app_env="benchmark")
    metadata = validate_and_probe(video_path, settings)
    sample_step = max(1, round(metadata.fps / settings.processing_fps))
    sampled_count = min(
        len(range(0, metadata.frame_count, sample_step)), settings.max_processed_frames
    )
    scale = min(1.0, settings.processing_long_edge / max(metadata.width, metadata.height))
    processed_width = max(1, round(metadata.width * scale))
    processed_height = max(1, round(metadata.height * scale))
    started = time.perf_counter()
    result = SlamPipeline(settings).process(video_path)
    elapsed = time.perf_counter() - started
    reprojection_errors = [
        diagnostic.reprojection_error
        for diagnostic in result.diagnostics
        if diagnostic.reprojection_error is not None
        and np.isfinite(diagnostic.reprojection_error)
    ]
    return BenchmarkMetrics(
        source=video_path.name,
        video_duration_seconds=round(metadata.duration_seconds, 6),
        original_resolution=f"{metadata.width}x{metadata.height}",
        original_fps=round(metadata.fps, 3),
        original_frame_count=metadata.frame_count,
        processing_resolution=f"{processed_width}x{processed_height}",
        effective_processing_fps=settings.processing_fps,
        frames_decoded=sampled_count,
        frames_processed=len(result.diagnostics),
        processing_seconds=round(result.processing_seconds, 6),
        total_end_to_end_seconds=round(elapsed, 6),
        processing_ratio=round(result.processing_seconds / metadata.duration_seconds, 6),
        pose_count=len(result.poses),
        point_count=len(result.landmarks),
        keyframe_count=sum(pose.keyframe for pose in result.poses),
        median_reprojection_error_px=(
            round(float(np.median(reprojection_errors)), 6) if reprojection_errors else None
        ),
        status=result.status.upper(),
        environment=f"{environment}; {platform.platform()}; {platform.processor() or 'CPU'}",
        generated_at=datetime.now(UTC).isoformat(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark the bounded sparse SLAM pipeline")
    parser.add_argument(
        "--video",
        type=Path,
        default=Path("test-videos/benchmark-10s.mp4"),
    )
    parser.add_argument("--environment", default="local")
    parser.add_argument("--threshold-seconds", type=float, default=10.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = evaluate_benchmark(
        run_benchmark(args.video, environment=args.environment),
        threshold_seconds=args.threshold_seconds,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
