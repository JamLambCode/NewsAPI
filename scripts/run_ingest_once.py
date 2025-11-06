"""Utility script to run a single ingestion cycle for a feed."""

from __future__ import annotations

import argparse
import asyncio

from src.app.ingest.rss import RSSClient
from src.app.main import process_payload, resolve_lang
from src.app.schemas import EntitiesRequest
from src.app.storage.db import async_session_factory


async def ingest_once(feed: str, limit: int) -> None:
    client = RSSClient(feed)
    entries = await client.fetch_entries()
    if not entries:
        print("No entries retrieved for feed", feed)
        return

    async with async_session_factory() as session:
        for entry in entries[:limit]:
            payload = EntitiesRequest(
                title=entry.title,
                url=entry.link,
                feed=feed,
                lang=resolve_lang(feed, None),
                text=entry.summary,
            )
            await process_payload(session, payload)
        await session.commit()
    print(f"Processed {min(limit, len(entries))} entries from {feed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single RSS ingestion cycle")
    parser.add_argument("--feed", default="fr/lemonde", help="Feed slug to ingest")
    parser.add_argument("--limit", type=int, default=5, help="Maximum number of entries to process")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(ingest_once(args.feed, args.limit))


if __name__ == "__main__":
    main()


