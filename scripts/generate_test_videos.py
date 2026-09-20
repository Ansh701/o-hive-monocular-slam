from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt


def _scene(seed: int = 701) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.uint8]]:
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        (
            rng.uniform(-5.5, 5.5, 1400),
            rng.uniform(-3.0, 3.0, 1400),
            rng.uniform(5.0, 18.0, 1400),
        )
    )
    colors = rng.integers(55, 255, size=(1400, 3), dtype=np.uint8)
    return points, colors


def _project(
    points: npt.NDArray[np.float64],
    center: npt.NDArray[np.float64],
    yaw: float,
    width: int,
    height: int,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    cosine, sine = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]])
    camera = (rotation @ (points - center).T).T
    valid = camera[:, 2] > 0.5
    focal = 0.9 * width
    pixels = np.column_stack(
        (
            focal * camera[:, 0] / camera[:, 2] + width / 2,
            focal * camera[:, 1] / camera[:, 2] + height / 2,
        )
    )
    valid &= (
        (pixels[:, 0] >= 3)
        & (pixels[:, 0] < width - 3)
        & (pixels[:, 1] >= 3)
        & (pixels[:, 1] < height - 3)
    )
    return pixels, valid


def generate_video(
    path: Path,
    *,
    mode: str,
    duration_seconds: float = 10.0,
    fps: int = 30,
    width: int = 960,
    height: int = 540,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create an MP4 fixture")
    points, colors = _scene()
    total_frames = round(duration_seconds * fps)
    try:
        for frame_index in range(total_frames):
            progress = frame_index / max(1, total_frames - 1)
            image = np.full((height, width, 3), 14, dtype=np.uint8)
            if mode == "low_texture":
                image[:] = round(28 + 2 * progress)
            else:
                if mode == "rotation":
                    center = np.zeros(3, dtype=np.float64)
                    yaw = (progress - 0.5) * 0.7
                else:
                    center = np.array([(progress - 0.5) * 2.2, 0.12 * np.sin(progress * 6), 0.0])
                    yaw = 0.025 * np.sin(progress * 4)
                pixels, visible = _project(points, center, yaw, width, height)
                for point, color in zip(pixels[visible], colors[visible], strict=True):
                    center_pixel = tuple(np.rint(point).astype(int))
                    point_color = tuple(int(value) for value in color)
                    cv2.circle(image, center_pixel, 2, point_color, -1)
                if mode == "tracking_loss" and 0.42 < progress < 0.67:
                    image[:] = 18
            cv2.putText(
                image,
                "O-HIVE synthetic geometry fixture",
                (24, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (215, 220, 230),
                1,
                cv2.LINE_AA,
            )
            writer.write(image)
    finally:
        writer.release()


def generate_suite(output: Path) -> None:
    generate_video(output / "benchmark-10s.mp4", mode="translation")
    generate_video(output / "low-texture.mp4", mode="low_texture", duration_seconds=4)
    generate_video(output / "rotation-heavy.mp4", mode="rotation", duration_seconds=5)
    generate_video(output / "tracking-loss.mp4", mode="tracking_loss", duration_seconds=6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate privacy-safe synthetic SLAM videos")
    parser.add_argument("--output", type=Path, default=Path("test-videos"))
    arguments = parser.parse_args()
    generate_suite(arguments.output)
