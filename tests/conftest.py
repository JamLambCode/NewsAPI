"""Pytest fixtures for Holocron tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.app.config import Settings
from src.app.deps import get_db_session
from src.app.main import app
from src.app.storage.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="function")
async def test_db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def test_db_sessionmaker(test_db_engine):
    """Provide a session factory bound to the test engine."""
    return async_sessionmaker(test_db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest.fixture(scope="function")
def test_settings() -> Settings:
    """Provide test-specific settings."""
    return Settings(
        app_env="test",
        log_level="DEBUG",
        database_url="sqlite+aiosqlite:///:memory:",
        scheduler_enabled=False,
        ollama_model="llama3",
        ollama_base_url="http://localhost:11434",
    )


@pytest_asyncio.fixture(scope="function")
async def test_client(test_db_sessionmaker) -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client for testing the FastAPI app."""

    async def override_db_session() -> AsyncIterator[AsyncSession]:
        async with test_db_sessionmaker() as session:
            yield session

    app.dependency_overrides[get_db_session] = override_db_session
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client
    app.dependency_overrides.pop(get_db_session, None)

