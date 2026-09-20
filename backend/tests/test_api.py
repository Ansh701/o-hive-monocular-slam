from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from backend.app.config import Settings
from backend.app.db import create_database, session_factory_for
from backend.app.main import create_app
from backend.app.models import SlamRun
from backend.slam.types import (
    FrameDiagnostics,
    Landmark,
    Pose,
    SlamErrorCategory,
    SlamFailure,
    SlamResult,
)
from scripts.generate_test_videos import generate_video


class SuccessfulPipeline:
    def process(self, path: Path, intrinsics=None, progress=None) -> SlamResult:
        assert path.exists()
        if progress:
            progress("tracking", 1, 1)
        return SlamResult(
            poses=(Pose.identity(0, 0.0), Pose.identity(1, 0.1)),
            landmarks=(Landmark(0, [0.0, 0.0, 4.0], 2),),
            diagnostics=(FrameDiagnostics(1, 30, 22),),
            processing_seconds=0.01,
            status="completed",
        )


class FailingPipeline:
    def process(self, path: Path, intrinsics=None, progress=None) -> SlamResult:
        assert path.exists()
        raise SlamFailure(
            SlamErrorCategory.INITIALIZATION_FAILED,
            "The video does not contain enough parallax.",
            internal_detail="private path and secret diagnostic",
        )


class SlowPipeline:
    def process(self, path: Path, intrinsics=None, progress=None) -> SlamResult:
        import time

        time.sleep(0.15)
        return SuccessfulPipeline().process(path, intrinsics, progress)


@pytest.fixture
def upload_video(tmp_path: Path) -> Path:
    path = tmp_path / "upload.mp4"
    generate_video(path, mode="translation", duration_seconds=0.6, width=320, height=180)
    return path


async def _client(tmp_path: Path, pipeline, **overrides):
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'api.db').as_posix()}"
    values = {
        "database_url": database_url,
        "temp_upload_dir": tmp_path / "uploads",
        "processing_timeout_seconds": 1,
        "rate_limit_runs_per_minute": 3,
    }
    values.update(overrides)
    settings = Settings(**values)
    app = create_app(settings=settings, pipeline=pipeline)
    await create_database(app.state.engine)
    return app, AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


async def _wait_terminal(client: AsyncClient, run_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/api/slam-runs/{run_id}")
        body = response.json()
        if body["status"] in {"COMPLETED", "PARTIAL", "FAILED"}:
            return body
        await asyncio.sleep(0.01)
    raise AssertionError("run did not reach a terminal state")


@pytest.mark.asyncio
async def test_upload_persists_run_returns_result_and_deletes_temporary_video(
    tmp_path: Path, upload_video: Path
) -> None:
    app, client = await _client(tmp_path, SuccessfulPipeline())
    async with client:
        with upload_video.open("rb") as handle:
            response = await client.post(
                "/api/slam-runs",
                files={"video": ("../../unsafe.mp4", handle, "video/mp4")},
            )
        assert response.status_code == 202
        result = await _wait_terminal(client, response.json()["id"])

        assert result["status"] == "COMPLETED"
        assert result["source_filename"] == "unsafe.mp4"
        assert result["result"]["point_count"] == 1
        assert list((tmp_path / "uploads").glob("*")) == []

        async with app.state.session_factory() as session:
            persisted = (await session.execute(select(SlamRun))).scalar_one()
            assert persisted.status == "COMPLETED"


@pytest.mark.asyncio
async def test_failure_is_safe_persisted_and_cleans_temporary_video(
    tmp_path: Path, upload_video: Path
) -> None:
    _app, client = await _client(tmp_path, FailingPipeline())
    async with client:
        with upload_video.open("rb") as handle:
            response = await client.post(
                "/api/slam-runs", files={"video": ("card.mp4", handle, "video/mp4")}
            )
        result = await _wait_terminal(client, response.json()["id"])

    assert result["status"] == "FAILED"
    assert result["error"]["code"] == "INITIALIZATION_FAILED"
    assert "secret" not in str(result)
    assert list((tmp_path / "uploads").glob("*")) == []


@pytest.mark.asyncio
async def test_timeout_cleans_file_and_reports_typed_error(
    tmp_path: Path, upload_video: Path
) -> None:
    _, client = await _client(
        tmp_path,
        SlowPipeline(),
        processing_timeout_seconds=0.03,
    )
    async with client:
        with upload_video.open("rb") as handle:
            response = await client.post(
                "/api/slam-runs", files={"video": ("card.mp4", handle, "video/mp4")}
            )
        result = await _wait_terminal(client, response.json()["id"])

    assert result["error"]["code"] == "PROCESSING_TIMEOUT"
    assert list((tmp_path / "uploads").glob("*")) == []


@pytest.mark.asyncio
async def test_rate_limit_and_concurrency_guard_are_enforced(
    tmp_path: Path, upload_video: Path
) -> None:
    _, client = await _client(
        tmp_path,
        SlowPipeline(),
        processing_timeout_seconds=1,
        rate_limit_runs_per_minute=1,
        max_concurrent_runs=1,
    )
    async with client:
        with upload_video.open("rb") as first, upload_video.open("rb") as second:
            accepted = await client.post(
                "/api/slam-runs", files={"video": ("one.mp4", first, "video/mp4")}
            )
            limited = await client.post(
                "/api/slam-runs", files={"video": ("two.mp4", second, "video/mp4")}
            )
        assert accepted.status_code == 202
        assert limited.status_code == 429
        await _wait_terminal(client, accepted.json()["id"])


@pytest.mark.asyncio
async def test_health_readiness_headers_and_bad_ids(tmp_path: Path) -> None:
    _app, client = await _client(tmp_path, SuccessfulPipeline())
    async with client:
        health = await client.get("/health")
        ready = await client.get("/ready")
        malformed = await client.get("/api/slam-runs/not-a-uuid")
        missing = await client.get(f"/api/slam-runs/{uuid.uuid4()}")

    assert health.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["database"] == "ready"
    assert malformed.status_code == 422
    assert missing.status_code == 404
    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert len(health.headers["x-request-id"]) == 32
    assert "default-src 'self'" in health.headers["content-security-policy"]


@pytest.mark.asyncio
async def test_readiness_fails_when_actual_table_is_missing(tmp_path: Path) -> None:
    database_url = f"sqlite+aiosqlite:///{(tmp_path / 'empty.db').as_posix()}"
    factory, engine = session_factory_for(database_url)
    settings = Settings(database_url=database_url, temp_upload_dir=tmp_path / "uploads")
    app = create_app(settings=settings, pipeline=SuccessfulPipeline())
    app.state.engine = engine
    app.state.session_factory = factory
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/ready")
    assert response.status_code == 503
