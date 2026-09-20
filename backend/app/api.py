from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from backend.app.config import Settings
from backend.app.models import SlamRun
from backend.app.security import SlidingWindowRateLimiter
from backend.slam.intrinsics import build_intrinsics
from backend.slam.pipeline import SlamPipeline
from backend.slam.types import CameraIntrinsics, SlamErrorCategory, SlamFailure, SlamResult
from backend.slam.video import sanitize_display_filename, validate_and_probe

logger = logging.getLogger("o_hive_slam")
router = APIRouter()
ALLOWED_DECLARED_TYPES = {"video/mp4", "video/quicktime", "video/webm"}


class RunAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: uuid.UUID
    status: str
    stage: str
    source_filename: str


class RunView(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: uuid.UUID
    status: str
    stage: str
    source_filename: str
    width: int
    height: int
    frame_count: int
    duration_seconds: float
    processing_seconds: float | None
    pose_count: int
    point_count: int
    keyframe_count: int
    scale: str = "arbitrary"
    result: dict[str, Any] | None = None
    error: dict[str, str] | None = None


class RunCoordinator:
    def __init__(
        self,
        settings: Settings,
        pipeline: SlamPipeline,
        session_factory: async_sessionmaker[AsyncSession],
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.session_factory = session_factory
        self.semaphore = asyncio.Semaphore(settings.max_concurrent_runs)
        self.limiter = SlidingWindowRateLimiter(settings.rate_limit_runs_per_minute)
        self.results: dict[str, dict[str, Any]] = {}
        self.errors: dict[str, dict[str, str]] = {}
        self.stages: dict[str, str] = {}
        self.tasks: set[asyncio.Task[None]] = set()

    def start(self, run_id: str, path: Path, intrinsics: CameraIntrinsics | None) -> None:
        task = asyncio.create_task(self._process(run_id, path, intrinsics))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _update(self, run_id: str, **values: Any) -> None:
        async with self.session_factory() as session:
            run = await session.get(SlamRun, run_id)
            if run is None:
                return
            for name, value in values.items():
                setattr(run, name, value)
            await session.commit()

    async def _process(self, run_id: str, path: Path, intrinsics: CameraIntrinsics | None) -> None:
        loop = asyncio.get_running_loop()

        def progress(stage_name: str, completed: int, total: int) -> None:
            del completed, total
            loop.call_soon_threadsafe(self.stages.__setitem__, run_id, stage_name)

        try:
            async with self.semaphore:
                self.stages[run_id] = "processing"
                await self._update(run_id, status="PROCESSING", stage="processing")
                result = await asyncio.wait_for(
                    asyncio.to_thread(self.pipeline.process, path, intrinsics, progress),
                    timeout=self.settings.processing_timeout_seconds,
                )
                await self._complete(run_id, result)
        except TimeoutError:
            timeout_failure = SlamFailure(
                SlamErrorCategory.PROCESSING_TIMEOUT,
                "Reconstruction exceeded the processing time limit. Try a shorter video.",
            )
            await self._fail(run_id, timeout_failure)
        except SlamFailure as failure:
            await self._fail(run_id, failure)
        except Exception as exc:
            logger.exception("run_failed run_id=%s category=internal", run_id)
            internal_failure = SlamFailure(
                SlamErrorCategory.INTERNAL_ERROR,
                "The reconstruction could not be completed. Your video was deleted.",
                internal_detail=type(exc).__name__,
            )
            await self._fail(run_id, internal_failure)
        finally:
            try:
                await asyncio.to_thread(path.unlink, missing_ok=True)
            except OSError:
                logger.warning("temporary_cleanup_deferred run_id=%s", run_id)

    async def _complete(self, run_id: str, result: SlamResult) -> None:
        public = result.to_public_dict(self.settings.render_point_limit)
        self.results[run_id] = public
        status_value = "PARTIAL" if result.status == "partial" else "COMPLETED"
        self.stages[run_id] = "complete"
        await self._update(
            run_id,
            status=status_value,
            stage="complete",
            processing_seconds=result.processing_seconds,
            pose_count=len(result.poses),
            point_count=len(result.landmarks),
            keyframe_count=sum(pose.keyframe for pose in result.poses),
            completed_at=datetime.now(UTC),
        )

    async def _fail(self, run_id: str, failure: SlamFailure) -> None:
        public = failure.to_public_dict()
        self.errors[run_id] = public
        self.stages[run_id] = "failed"
        await self._update(
            run_id,
            status="FAILED",
            stage="failed",
            error_code=public["code"],
            error_message=public["message"][:500],
            completed_at=datetime.now(UTC),
        )


async def _write_bounded_upload(upload: UploadFile, target: Path, max_bytes: int) -> int:
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("xb") as destination:
            while chunk := await upload.read(1024 * 1024):
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail={
                            "code": "VIDEO_TOO_LARGE",
                            "message": "The video exceeds the configured upload size limit.",
                        },
                    )
                destination.write(chunk)
    except Exception:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        raise
    finally:
        await upload.close()
    return written


def _client_identity(request: Request) -> str:
    return request.client.host if request.client else "unknown"


@router.post("/api/slam-runs", response_model=RunAccepted, status_code=status.HTTP_202_ACCEPTED)
async def create_run(
    request: Request,
    video: Annotated[UploadFile, File()],
    fx: Annotated[float | None, Form()] = None,
    fy: Annotated[float | None, Form()] = None,
    cx: Annotated[float | None, Form()] = None,
    cy: Annotated[float | None, Form()] = None,
) -> RunAccepted:
    coordinator: RunCoordinator = request.app.state.coordinator
    await coordinator.limiter.check(_client_identity(request))
    if video.content_type not in ALLOWED_DECLARED_TYPES:
        raise HTTPException(
            status_code=415,
            detail={"code": "VIDEO_INVALID", "message": "Upload MP4, MOV, or WebM video."},
        )
    source_filename = sanitize_display_filename(video.filename or "video")
    suffix = Path(source_filename).suffix.lower()
    run_id = str(uuid.uuid4())
    target = coordinator.settings.temp_upload_dir / f"{uuid.uuid4().hex}{suffix}"
    await _write_bounded_upload(video, target, coordinator.settings.max_video_bytes)
    try:
        metadata = validate_and_probe(target, coordinator.settings)
        calibration = None
        if any(value is not None for value in (fx, fy, cx, cy)):
            calibration = build_intrinsics(
                width=metadata.width,
                height=metadata.height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
        async with request.app.state.session_factory() as session:
            session.add(
                SlamRun(
                    id=run_id,
                    status="UPLOADED",
                    stage="queued",
                    source_filename=source_filename,
                    width=metadata.width,
                    height=metadata.height,
                    frame_count=metadata.frame_count,
                    duration_seconds=metadata.duration_seconds,
                    algorithm_version=coordinator.settings.algorithm_version,
                    intrinsics_approximate=calibration is None,
                )
            )
            await session.commit()
    except SlamFailure as failure:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        raise HTTPException(status_code=422, detail=failure.to_public_dict()) from failure
    except ValueError as exc:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        raise HTTPException(
            status_code=422,
            detail={"code": "CALIBRATION_INVALID", "message": str(exc)},
        ) from exc
    except Exception:
        await asyncio.to_thread(target.unlink, missing_ok=True)
        raise
    coordinator.start(run_id, target, calibration)
    return RunAccepted(
        id=uuid.UUID(run_id),
        status="UPLOADED",
        stage="queued",
        source_filename=source_filename,
    )


@router.get("/api/slam-runs/{run_id}", response_model=RunView)
async def get_run(request: Request, run_id: uuid.UUID) -> RunView:
    key = str(run_id)
    async with request.app.state.session_factory() as session:
        run = await session.get(SlamRun, key)
    if run is None:
        raise HTTPException(status_code=404, detail={"code": "RUN_NOT_FOUND"})
    error = None
    if run.error_code:
        error = {"code": run.error_code, "message": run.error_message or "Reconstruction failed."}
    return RunView(
        id=run_id,
        status=run.status,
        stage=request.app.state.coordinator.stages.get(key, run.stage),
        source_filename=run.source_filename,
        width=run.width,
        height=run.height,
        frame_count=run.frame_count,
        duration_seconds=run.duration_seconds,
        processing_seconds=run.processing_seconds,
        pose_count=run.pose_count,
        point_count=run.point_count,
        keyframe_count=run.keyframe_count,
        result=request.app.state.coordinator.results.get(key),
        error=request.app.state.coordinator.errors.get(key, error),
    )


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "alive"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    try:
        async with request.app.state.session_factory() as session:
            await session.execute(text("SELECT 1 FROM slam_runs LIMIT 1"))
    except Exception as exc:
        logger.warning("readiness_failed category=%s", type(exc).__name__)
        raise HTTPException(
            status_code=503,
            detail={"status": "not_ready", "database": "unavailable"},
        ) from exc
    return {"status": "ready", "database": "ready"}
