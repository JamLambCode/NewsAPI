"""Unit tests for deterministic NLP components."""

from __future__ import annotations

from src.app.nlp.ner import normalize_text, normalize_texts


def test_normalize_text_lowercase():
    """Test text normalization converts to lowercase."""
    assert normalize_text("HELLO") == "hello"


def test_normalize_text_unicode():
    """Test text normalization handles unicode."""
    assert normalize_text("café") == "café"
    assert normalize_text("élève") == "élève"


def test_normalize_text_whitespace():
    """Test text normalization collapses whitespace."""
    assert normalize_text("hello    world") == "hello world"
    assert normalize_text("  trim  ") == "trim"


def test_normalize_texts_batch():
    """Test batch normalization drops empties."""
    inputs = ["Hello", "", "  World  ", "Test"]
    outputs = normalize_texts(inputs)
    assert len(outputs) == 3
    assert "hello" in outputs
    assert "world" in outputs
    assert "test" in outputs

