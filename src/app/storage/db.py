"""Database engine and initialization utilities."""

from __future__ import annotations

import argparse
import asyncio
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from ..config import get_settings
from .models import Base


settings = get_settings()


def create_engine() -> AsyncEngine:
    """Create an async SQLAlchemy engine based on configuration."""

    return create_async_engine(settings.database_url, echo=False, future=True, pool_pre_ping=True)


engine: AsyncEngine = create_engine()
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def init_db() -> None:
    """Create database tables."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db() -> None:
    """Drop database tables."""

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def get_connection() -> AsyncIterator[AsyncEngine]:
    """Yield the configured engine (useful for scripts)."""

    yield engine


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Holocron DB utilities")
    parser.add_argument("--init", action="store_true", help="Create database tables")
    parser.add_argument("--drop", action="store_true", help="Drop database tables")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.init:
        asyncio.run(init_db())
    elif args.drop:
        asyncio.run(drop_db())
    else:
        raise SystemExit("Specify --init or --drop to manage the schema.")


if __name__ == "__main__":
    main()


