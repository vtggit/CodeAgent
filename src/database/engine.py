"""
Database engine and session management.

Provides async SQLAlchemy engine creation, session factories,
and lifecycle management for the application.
"""

import os
from pathlib import Path
from typing import AsyncGenerator

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.database.models import Base
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Module-level engine and session factory
_engine: AsyncEngine | None = None
AsyncSessionLocal: async_sessionmaker[AsyncSession] | None = None


def _get_database_url() -> str:
    """
    Get the database URL from settings or environment.

    Converts sqlite:// URLs to sqlite+aiosqlite:// for async support.

    Returns:
        str: The async-compatible database URL.
    """
    try:
        from src.config.settings import get_settings
        db_url = get_settings().database_url
    except Exception:
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./data/multi_agent.db")

    # Convert sync SQLite URL to async
    if db_url.startswith("sqlite:///"):
        db_url = db_url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)

    return db_url


def _ensure_db_directory(db_url: str) -> None:
    """
    Ensure the directory for SQLite database file exists.

    Args:
        db_url: The database URL.
    """
    if "sqlite" in db_url:
        # Extract path from URL like sqlite+aiosqlite:///./data/multi_agent.db
        path_part = db_url.split("///")[-1]
        db_path = Path(path_part)
        db_path.parent.mkdir(parents=True, exist_ok=True)


def get_engine() -> AsyncEngine:
    """
    Get or create the global async database engine.

    Returns:
        AsyncEngine: The SQLAlchemy async engine.
    """
    global _engine
    if _engine is None:
        db_url = _get_database_url()
        _ensure_db_directory(db_url)

        connect_args = {}
        if "sqlite" in db_url:
            connect_args["check_same_thread"] = False

        _engine = create_async_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args=connect_args,
        )

        # Enable WAL mode and foreign keys for SQLite
        if "sqlite" in db_url:
            @event.listens_for(_engine.sync_engine, "connect")
            def _set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL")
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """
    Get or create the global async session factory.

    Returns:
        async_sessionmaker: The session factory.
    """
    global AsyncSessionLocal
    if AsyncSessionLocal is None:
        engine = get_engine()
        AsyncSessionLocal = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return AsyncSessionLocal


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that yields an async database session.

    Usage with FastAPI:
        @app.get("/items")
        async def get_items(db: AsyncSession = Depends(get_session)):
            ...

    Yields:
        AsyncSession: An active database session.
    """
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_async_session() -> AsyncSession:
    """
    Get a standalone async session (not as a dependency).

    Caller is responsible for committing/closing.

    Returns:
        AsyncSession: A new database session.
    """
    factory = get_session_factory()
    return factory()


async def init_db() -> None:
    """
    Initialize the database by creating all tables.

    Should be called during application startup.
    """
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("database_initialized")


async def close_db() -> None:
    """
    Close the database engine and clean up connections.

    Should be called during application shutdown.
    """
    global _engine, AsyncSessionLocal
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        AsyncSessionLocal = None
        logger.info("database_connections_closed")
