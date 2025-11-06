"""Deterministic named entity recognition wrapper."""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Iterable, List


ALLOWED_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "EVENT", "DATE"}


@dataclass(slots=True)
class EntitySpan:
    """Structured entity span produced by the NER model."""

    text: str
    label: str
    start: int
    end: int


class DeterministicNER:
    """Wrapper around a Hugging Face pipeline for deterministic NER."""

    def __init__(self, model_name: str = "numind/NuNER-multilingual-v0.1") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _load_pipeline(self):  # pragma: no cover - lazy load
        from transformers import pipeline

        if self._pipeline is None:
            self._pipeline = pipeline(
                "ner",
                model=self.model_name,
                aggregation_strategy="simple",
            )

    def extract(self, text: str) -> List[EntitySpan]:
        """Return normalized entity spans for the provided text."""

        if not text:
            return []
        self._load_pipeline()
        raw_spans = self._pipeline(text)
        spans: List[EntitySpan] = []
        for span in raw_spans:
            label = span.get("entity_group", "")
            if label in ALLOWED_ENTITY_TYPES:
                spans.append(
                    EntitySpan(
                        text=span.get("word", "").strip(),
                        label=label,
                        start=int(span.get("start", 0)),
                        end=int(span.get("end", 0)),
                    )
                )
        return spans


def normalize_text(value: str) -> str:
    """Normalize entity surface strings for consistent storage."""

    value = unicodedata.normalize("NFKC", value.lower())
    return " ".join(value.split())


def normalize_texts(values: Iterable[str]) -> List[str]:
    """Normalize a collection of values and drop empties."""

    normalized: List[str] = []
    for value in values:
        if not value:
            continue
        normalized.append(normalize_text(value))
    return normalized



