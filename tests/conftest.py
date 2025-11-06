"""Pytest fixtures for Holocron tests."""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from src.app.config import Settings, get_settings
from src.app.main import app
from src.app.storage.models import Base


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="function")
async def test_db_engine():
    """Create an in-memory SQLite database for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture(scope="function")
async def test_db_session(test_db_engine) -> AsyncIterator[AsyncSession]:
    """Provide a database session for testing."""
    async_session_factory = sessionmaker(
        test_db_engine, class_=AsyncSession, expire_on_commit=False
    )
    async with async_session_factory() as session:
        yield session


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


@pytest.fixture(scope="function")
async def test_client() -> AsyncIterator[AsyncClient]:
    """Provide an async HTTP client for testing the FastAPI app."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

