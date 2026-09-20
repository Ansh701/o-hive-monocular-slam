from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized operational limits and environment configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    database_url: str = "sqlite+aiosqlite:///./slam.db"
    temp_upload_dir: Path = Path("uploads")
    allowed_hosts: str = "localhost,127.0.0.1,test"
    development_cors_origins: str = "http://localhost:5173"

    max_video_bytes: int = Field(default=50 * 1024 * 1024, ge=1, le=250 * 1024 * 1024)
    max_video_duration_seconds: float = Field(default=30.0, gt=0, le=120)
    max_video_width: int = Field(default=3840, ge=1, le=7680)
    max_video_height: int = Field(default=2160, ge=1, le=4320)
    max_video_frames: int = Field(default=1800, ge=2, le=18_000)
    processing_timeout_seconds: float = Field(default=45.0, gt=0, le=300)
    max_concurrent_runs: int = Field(default=1, ge=1, le=4)
    rate_limit_runs_per_minute: int = Field(default=4, ge=1, le=60)
    render_point_limit: int = Field(default=12_000, ge=100, le=100_000)

    processing_long_edge: int = Field(default=640, ge=320, le=1280)
    processing_fps: float = Field(default=8.0, ge=2.0, le=30.0)
    max_processed_frames: int = Field(default=240, ge=8, le=900)
    max_features: int = Field(default=900, ge=100, le=4000)
    min_initial_landmarks: int = Field(default=12, ge=8, le=200)
    min_pnp_landmarks: int = Field(default=10, ge=6, le=200)
    max_tracking_failures: int = Field(default=2, ge=1, le=10)
    algorithm_version: str = "sparse-slam-v1"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_total_pixels(self) -> int:
        return self.max_video_width * self.max_video_height


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
