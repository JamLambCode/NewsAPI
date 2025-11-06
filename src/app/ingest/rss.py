"""RSS ingestion utilities."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import dateparser
import feedparser
import httpx


FEED_REGISTRY = {
    "fr/lemonde": "https://www.lemonde.fr/rss/en_continu.xml",
}
FEED_LANG_MAP = {
    "fr/lemonde": "fr",
}


@dataclass(slots=True)
class FeedEntry:
    """Normalized RSS entry."""

    title: str
    link: str
    summary: Optional[str]
    published: Optional[datetime]
    feed: str


class RSSClient:
    """Fetch and parse RSS feeds asynchronously."""

    def __init__(
        self,
        feed_slug: str,
        *,
        client: Optional[httpx.AsyncClient] = None,
    ) -> None:
        if feed_slug not in FEED_REGISTRY:
            raise ValueError(f"Unknown feed slug: {feed_slug}")
        self.feed_slug = feed_slug
        self.feed_url = FEED_REGISTRY[feed_slug]
        self._client = client

    async def fetch_entries(self, *, timeout: float = 10.0) -> list[FeedEntry]:
        """Return parsed feed entries."""

        close_client = False
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(timeout))
            close_client = True

        try:
            response = await client.get(self.feed_url)
            response.raise_for_status()
            parsed = feedparser.parse(response.text)
        finally:
            if close_client:
                await client.aclose()

        entries: list[FeedEntry] = []
        for entry in parsed.entries:
            title = entry.get("title", "").strip()
            link = entry.get("link") or entry.get("id")
            summary = entry.get("summary") or entry.get("description")
            published = _parse_datetime(entry)
            if not title or not link:
                continue
            entries.append(
                FeedEntry(
                    title=title,
                    link=link,
                    summary=summary.strip() if summary else None,
                    published=published,
                    feed=self.feed_slug,
                )
            )
        return entries


def _parse_datetime(entry: feedparser.util.FeedParserDict) -> Optional[datetime]:
    """Parse the published timestamp using dateparser."""

    for key in ("published", "updated", "created"):
        raw = entry.get(key)
        if raw:
            with contextlib.suppress(ValueError, OverflowError):
                return dateparser.parse(raw)
    return None



