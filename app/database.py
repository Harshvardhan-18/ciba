"""
Async SQLAlchemy engine, session factory, and the FastAPI `get_db` dependency.

Uses SQLAlchemy 2.0 async style throughout — no sync engine, no run_sync shims.
The session is yielded inside a context manager so any exception auto-rolls back.
"""
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an AsyncSession and closes it on exit."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


def get_session_factory():
    """
    Returns the current session factory.
    Indirected through a function so tests can monkeypatch it to inject
    a different factory (e.g., pointing at ciba_test instead of ciba_db)
    without restarting the engine.
    """
    return AsyncSessionLocal
