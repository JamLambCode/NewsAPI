"""Quote reuse detection utilities (Déjà-vu Detector)."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence


WORD_RE = re.compile(r"\w+[’']?\w*|\S")


def shingles(text: str, n: int = 12) -> Iterator[str]:
    """Yield overlapping n-word shingles from the provided text."""

    if not text:
        return iter(())
    tokens = [token for token in WORD_RE.findall(text) if token.strip()]
    if len(tokens) < n:
        return iter(())

    snippets: list[str] = []
    for idx in range(len(tokens) - n + 1):
        snippet = " ".join(tokens[idx : idx + n])
        snippets.append(snippet)
    return iter(snippets)


def fingerprint(snippet: str) -> str:
    """Stable hash for a snippet."""

    normalized = re.sub(r"\s+", " ", snippet.lower()).strip()
    return hashlib.blake2s(normalized.encode("utf-8"), digest_size=16).hexdigest()


@dataclass(slots=True)
class ArticleMeta:
    article_id: int
    feed: str
    title: str
    text_fr: str


@dataclass(slots=True)
class QuoteOccurrence:
    article_id: int
    feed: str
    title: str
    snippet: str


@dataclass(slots=True)
class QuoteCluster:
    fingerprint: str
    snippet: str
    occurrences: list[QuoteOccurrence]

    @property
    def count(self) -> int:
        return len(self.occurrences)

    @property
    def feeds(self) -> set[str]:
        return {occ.feed for occ in self.occurrences}


def collect_reused_quotes(
    articles: Sequence[ArticleMeta],
    *,
    shingle_size: int = 12,
    min_hits: int = 3,
    require_distinct_feeds: bool = True,
) -> list[QuoteCluster]:
    """Aggregate repeated shingles across articles."""

    occurrences: dict[str, list[QuoteOccurrence]] = defaultdict(list)

    for article in articles:
        seen_fingerprints: set[str] = set()
        for snippet in shingles(article.text_fr, shingle_size):
            fp = fingerprint(snippet)
            if fp in seen_fingerprints:
                continue
            seen_fingerprints.add(fp)
            occurrences[fp].append(
                QuoteOccurrence(
                    article_id=article.article_id,
                    feed=article.feed,
                    title=article.title,
                    snippet=snippet,
                )
            )

    clusters: list[QuoteCluster] = []
    for fp, occs in occurrences.items():
        if len(occs) < min_hits:
            continue
        if require_distinct_feeds and len({occ.feed for occ in occs}) < min_hits:
            continue
        clusters.append(QuoteCluster(fingerprint=fp, snippet=occs[0].snippet, occurrences=occs))

    clusters.sort(key=lambda cluster: cluster.count, reverse=True)
    return clusters

