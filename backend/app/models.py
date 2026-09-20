from __future__ import annotations

import uuid
from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utc_now() -> datetime:
    return datetime.now(UTC)


class SlamRun(Base):
    __tablename__ = "slam_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    status: Mapped[str] = mapped_column(String(24), index=True)
    stage: Mapped[str] = mapped_column(String(40), default="uploaded")
    source_filename: Mapped[str] = mapped_column(String(120))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    frame_count: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[float] = mapped_column(Float)
    processing_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    pose_count: Mapped[int] = mapped_column(Integer, default=0)
    point_count: Mapped[int] = mapped_column(Integer, default=0)
    keyframe_count: Mapped[int] = mapped_column(Integer, default=0)
    algorithm_version: Mapped[str] = mapped_column(String(40))
    intrinsics_approximate: Mapped[bool] = mapped_column(default=True)
    error_code: Mapped[str | None] = mapped_column(String(40), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
