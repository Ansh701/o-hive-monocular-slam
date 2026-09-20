from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import cv2
import numpy as np
import numpy.typing as npt

from backend.app.config import Settings
from backend.slam.bundle_adjustment import (
    BundleAdjustmentConfig,
    BundleWindow,
    Observation,
    bundle_adjust,
)
from backend.slam.features import detect_corners, track_lk_forward_backward
from backend.slam.geometry import camera_center
from backend.slam.intrinsics import build_intrinsics
from backend.slam.keyframes import KeyframePolicy, should_insert_keyframe
from backend.slam.map import TriangulationConfig, initialize_two_view, triangulate_filtered
from backend.slam.tracker import estimate_pose_pnp
from backend.slam.types import (
    CameraIntrinsics,
    FrameDiagnostics,
    Landmark,
    Pose,
    SlamErrorCategory,
    SlamFailure,
    SlamResult,
)
from backend.slam.video import (
    ProcessingProfile,
    SampledFrame,
    iter_sampled_frames,
    validate_and_probe,
)

ProgressCallback = Callable[[str, int, int], None]


def _noop_progress(stage: str, completed: int, total: int) -> None:
    del stage, completed, total


class SlamPipeline:
    """Bounded classical sparse monocular SLAM for short uploaded videos."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process(
        self,
        path: Path,
        intrinsics: CameraIntrinsics | None = None,
        progress: ProgressCallback | None = None,
    ) -> SlamResult:
        started = time.perf_counter()
        report = progress or _noop_progress
        metadata = validate_and_probe(path, self.settings)
        profile = ProcessingProfile(
            long_edge=self.settings.processing_long_edge,
            target_fps=self.settings.processing_fps,
            max_frames=self.settings.max_processed_frames,
        )
        report("decoding", 0, min(metadata.frame_count, profile.max_frames))
        frames = list(iter_sampled_frames(path, profile))
        if len(frames) < 3:
            raise SlamFailure(
                SlamErrorCategory.INITIALIZATION_FAILED,
                "The video does not contain enough usable frames.",
            )
        processed_height, processed_width = frames[0].image.shape[:2]
        if intrinsics is None:
            camera = build_intrinsics(width=processed_width, height=processed_height)
        else:
            scale_x = processed_width / intrinsics.width
            scale_y = processed_height / intrinsics.height
            camera = CameraIntrinsics(
                fx=intrinsics.fx * scale_x,
                fy=intrinsics.fy * scale_y,
                cx=intrinsics.cx * scale_x,
                cy=intrinsics.cy * scale_y,
                width=processed_width,
                height=processed_height,
                approximate=intrinsics.approximate,
            )
        return self._reconstruct(frames, camera, report, started)

    def _reconstruct(
        self,
        frames: list[SampledFrame],
        camera: CameraIntrinsics,
        report: ProgressCallback,
        started: float,
    ) -> SlamResult:
        total = len(frames)
        first_gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(frames[0].image, cv2.COLOR_BGR2GRAY))
        current_points = detect_corners(first_gray, max_features=self.settings.max_features)
        if len(current_points) < self.settings.min_initial_landmarks:
            raise SlamFailure(
                SlamErrorCategory.INITIALIZATION_FAILED,
                "This video has too little texture to initialize a 3D map.",
            )
        anchor_points = current_points.copy()
        previous_gray = first_gray
        report("initializing", 0, total)

        initialization = None
        initialized_frame_index = 0
        for frame_index in range(1, min(total, 15)):
            gray = cast(
                npt.NDArray[np.uint8],
                cv2.cvtColor(frames[frame_index].image, cv2.COLOR_BGR2GRAY),
            )
            tracked = track_lk_forward_backward(previous_gray, gray, current_points)
            anchor_points = anchor_points[tracked.source_indices]
            current_points = tracked.current_points
            previous_gray = gray
            if len(current_points) < self.settings.min_initial_landmarks:
                break
            displacement = np.linalg.norm(current_points - anchor_points, axis=1)
            if float(np.median(displacement)) < 2.5:
                continue
            try:
                initialization = initialize_two_view(
                    camera,
                    anchor_points,
                    current_points,
                    min_landmarks=self.settings.min_initial_landmarks,
                    config=TriangulationConfig(
                        min_points=self.settings.min_initial_landmarks,
                        max_reprojection_error=2.5,
                    ),
                )
                initialized_frame_index = frame_index
                break
            except SlamFailure:
                continue
        if initialization is None:
            raise SlamFailure(
                SlamErrorCategory.INITIALIZATION_FAILED,
                "The video does not provide enough translation and parallax for a reliable map.",
            )

        landmark_positions = initialization.triangulation.points.copy()
        landmark_ids = np.full(len(current_points), -1, dtype=np.int64)
        landmark_ids[initialization.triangulation.source_indices] = np.arange(
            len(landmark_positions), dtype=np.int64
        )
        poses: list[np.ndarray] = [initialization.pose_a.copy(), initialization.pose_b.copy()]
        pose_frame_indices = [0, frames[initialized_frame_index].source_index]
        pose_timestamps = [0.0, frames[initialized_frame_index].timestamp_seconds]
        keyframe_positions = {0, 1}
        observation_counts = np.full(len(landmark_positions), 2, dtype=np.int64)
        observations: list[Observation] = []
        for source_index, initial_landmark_id in enumerate(landmark_ids):
            if initial_landmark_id < 0:
                continue
            observations.append(
                Observation(0, int(initial_landmark_id), anchor_points[source_index])
            )
            observations.append(
                Observation(1, int(initial_landmark_id), current_points[source_index])
            )

        diagnostics = [
            FrameDiagnostics(0, len(anchor_points), len(landmark_positions), stage="initialized"),
            FrameDiagnostics(
                frames[initialized_frame_index].source_index,
                len(current_points),
                len(landmark_positions),
                stage="initialized",
            ),
        ]
        last_keyframe_pose = initialization.pose_b.copy()
        last_keyframe_pose_index = 1
        failures = 0
        partial = False

        for frame_index in range(initialized_frame_index + 1, total):
            frame = frames[frame_index]
            gray = cast(npt.NDArray[np.uint8], cv2.cvtColor(frame.image, cv2.COLOR_BGR2GRAY))
            tracked = track_lk_forward_backward(previous_gray, gray, current_points)
            if len(tracked.current_points) == 0:
                partial = True
                break
            anchor_points = anchor_points[tracked.source_indices]
            landmark_ids = landmark_ids[tracked.source_indices]
            current_points = tracked.current_points
            previous_gray = gray
            mapped = landmark_ids >= 0
            if int(np.count_nonzero(mapped)) < self.settings.min_pnp_landmarks:
                failures += 1
                if failures >= self.settings.max_tracking_failures:
                    partial = True
                    break
                continue
            try:
                pnp = estimate_pose_pnp(
                    landmark_positions[landmark_ids[mapped]],
                    current_points[mapped],
                    camera,
                    min_inliers=self.settings.min_pnp_landmarks,
                )
            except SlamFailure:
                failures += 1
                if failures >= self.settings.max_tracking_failures:
                    partial = True
                    break
                continue
            failures = 0
            pose_index = len(poses)
            poses.append(pnp.world_to_camera)
            pose_frame_indices.append(frame.source_index)
            pose_timestamps.append(frame.timestamp_seconds)
            mapped_rows = np.flatnonzero(mapped)[pnp.inlier_indices]
            for row in mapped_rows:
                landmark_id = int(landmark_ids[row])
                observation_counts[landmark_id] += 1
                observations.append(Observation(pose_index, landmark_id, current_points[row]))

            translation = float(
                np.linalg.norm(
                    camera_center(pnp.world_to_camera) - camera_center(last_keyframe_pose)
                )
            )
            relative_rotation = pnp.world_to_camera[:3, :3] @ last_keyframe_pose[:3, :3].T
            rotation_vector, _ = cv2.Rodrigues(relative_rotation)
            rotation_degrees = float(np.degrees(np.linalg.norm(rotation_vector)))
            parallax = float(
                np.median(np.linalg.norm(current_points[mapped] - anchor_points[mapped], axis=1))
            )
            insert_keyframe = should_insert_keyframe(
                translation=translation,
                rotation_degrees=rotation_degrees,
                median_parallax_pixels=parallax,
                tracked_landmarks=int(np.count_nonzero(mapped)),
                frames_since_keyframe=pose_index - last_keyframe_pose_index,
                policy=KeyframePolicy(),
            )
            if insert_keyframe:
                unknown = landmark_ids < 0
                if int(np.count_nonzero(unknown)) >= self.settings.min_initial_landmarks:
                    try:
                        added = triangulate_filtered(
                            camera.matrix,
                            poses[0],
                            pnp.world_to_camera,
                            anchor_points[unknown],
                            current_points[unknown],
                            TriangulationConfig(
                                min_points=self.settings.min_initial_landmarks,
                                max_reprojection_error=2.5,
                            ),
                        )
                        unknown_rows = np.flatnonzero(unknown)[added.source_indices]
                        first_new_id = len(landmark_positions)
                        landmark_positions = np.vstack((landmark_positions, added.points))
                        new_ids = np.arange(
                            first_new_id, first_new_id + len(added.points), dtype=np.int64
                        )
                        landmark_ids[unknown_rows] = new_ids
                        observation_counts = np.concatenate(
                            (observation_counts, np.full(len(new_ids), 2, dtype=np.int64))
                        )
                        for row, new_landmark_id in zip(unknown_rows, new_ids, strict=True):
                            observations.append(
                                Observation(0, int(new_landmark_id), anchor_points[row])
                            )
                            observations.append(
                                Observation(pose_index, int(new_landmark_id), current_points[row])
                            )
                    except SlamFailure:
                        pass
                keyframe_positions.add(pose_index)
                last_keyframe_pose = pnp.world_to_camera.copy()
                last_keyframe_pose_index = pose_index

            diagnostics.append(
                FrameDiagnostics(
                    frame.source_index,
                    len(current_points),
                    pnp.inlier_count,
                    pnp.median_reprojection_error,
                    "tracking",
                )
            )
            report("tracking", frame_index + 1, total)

        report("optimizing", len(poses), len(poses))
        ba_result = bundle_adjust(
            BundleWindow(tuple(poses), landmark_positions),
            observations,
            camera.matrix,
            BundleAdjustmentConfig(max_nfev=25),
        )
        if ba_result.accepted:
            poses = list(ba_result.window.poses)
            landmark_positions = ba_result.window.landmarks

        public_poses = tuple(
            Pose(
                pose,
                frame_index=pose_frame_indices[index],
                timestamp_seconds=pose_timestamps[index],
                keyframe=index in keyframe_positions,
            )
            for index, pose in enumerate(poses)
        )
        landmarks = tuple(
            Landmark(
                id=index,
                position=point,
                observations=int(observation_counts[index]),
            )
            for index, point in enumerate(landmark_positions)
        )
        status: Literal["completed", "partial"] = "partial" if partial else "completed"
        return SlamResult(
            poses=public_poses,
            landmarks=landmarks,
            diagnostics=tuple(diagnostics),
            processing_seconds=time.perf_counter() - started,
            status=status,
        )
