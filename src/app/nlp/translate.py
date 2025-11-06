"""Deterministic translation helpers."""

from __future__ import annotations

from typing import Iterable, List


LANG_MODEL_MAP = {
    "fr": "Helsinki-NLP/opus-mt-fr-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
}


class Translator:
    """Wrap MarianMT translation pipelines."""

    def __init__(self, default_lang: str = "fr") -> None:
        self.default_lang = default_lang
        self._pipelines: dict[str, object] = {}

    def _get_pipeline(self, lang: str):  # pragma: no cover - lazy load
        from transformers import pipeline

        model_name = LANG_MODEL_MAP.get(lang, LANG_MODEL_MAP[self.default_lang])
        if model_name not in self._pipelines:
            self._pipelines[model_name] = pipeline("translation", model=model_name)
        return self._pipelines[model_name]

    def translate(self, values: Iterable[str], lang: str) -> List[str]:
        """Translate a list of surface strings."""

        values = [value for value in values if value]
        if not values:
            return []
        translator = self._get_pipeline(lang)
        outputs = translator(values)
        return [item["translation_text"] for item in outputs]

    def translate_one(self, value: str, lang: str) -> str:
        """Translate a single string."""

        result = self.translate([value], lang)
        return result[0] if result else ""



