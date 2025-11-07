"""Unit tests for deja-vu analytics utilities."""

from __future__ import annotations

from src.app.analytics.dejavu import ArticleMeta, collect_reused_quotes, shingles


def test_shingles_basic():
    text = "Emmanuel Macron a déclaré que la France soutient l'Ukraine fermement."
    result = list(shingles(text, n=5))
    assert result
    # ensure sliding window
    assert result[0].startswith("Emmanuel Macron")
    assert result[1].split()[0] == "Macron"


def test_collect_reused_quotes_requires_distinct_feeds():
    articles = [
        ArticleMeta(article_id=1, feed="fr/lemonde", title="A", text_fr="La France soutient l'Ukraine fermement."),
        ArticleMeta(article_id=2, feed="fr/france24", title="B", text_fr="La France soutient l'Ukraine fermement."),
        ArticleMeta(article_id=3, feed="fr/rfi", title="C", text_fr="La France soutient l'Ukraine fermement."),
    ]
    clusters = collect_reused_quotes(articles, shingle_size=5, min_hits=3)
    assert clusters
    primary = clusters[0]
    assert primary.count == 3
    assert primary.feeds == {"fr/lemonde", "fr/france24", "fr/rfi"}

