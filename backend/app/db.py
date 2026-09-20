from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.app.models import Base


def session_factory_for(
    database_url: str,
) -> tuple[async_sessionmaker[AsyncSession], AsyncEngine]:
    engine = create_async_engine(database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False), engine


async def create_database(engine: AsyncEngine) -> None:
    """Create tables for isolated tests/local demos; production uses Alembic."""
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def session_dependency(request: Request) -> AsyncIterator[AsyncSession]:
    async with request.app.state.session_factory() as session:
        yield session
