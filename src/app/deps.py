"""FastAPI dependencies."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession

from .storage.db import async_session_factory


async def get_db_session() -> AsyncIterator[AsyncSession]:
    """Yield an async SQLAlchemy session."""

    async with async_session_factory() as session:
        yield session


