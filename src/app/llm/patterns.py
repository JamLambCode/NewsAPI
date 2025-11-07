"""Pattern summarization helpers for analytics features."""

from __future__ import annotations

import logging
from typing import Iterable, Optional


logger = logging.getLogger("holocron.llm")


SUMMARY_SYSTEM_PROMPT = (
    "You are a media analyst. Given a repeated line across French news outlets, "
    "you craft a short (<=12 words) English title that describes the narrative pattern. "
    "Respond with JSON: {\"title\": \"...\"}. Avoid quotation marks around outlet names."
)


class PatternSummarizer:
    """Generate short summaries for repeated quote clusters using an Ollama-first strategy."""

    def __init__(
        self,
        ollama_model: str,
        ollama_base_url: str,
        *,
        openai_model: Optional[str] = None,
        openai_key: Optional[str] = None,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.openai_model = openai_model
        self.openai_key = openai_key

    async def summarize(self, snippet: str, outlets: Iterable[str]) -> str:
        """Return a short title describing the reuse pattern."""

        outlets_list = sorted(set(outlets))
        if not snippet:
            return "Repeated phrasing detected"

        title = await self._summarize_with_ollama(snippet, outlets_list)
        if title:
            return title
        title = await self._summarize_with_openai(snippet, outlets_list)
        if title:
            return title
        return self._fallback(snippet, outlets_list)

    async def _summarize_with_ollama(self, snippet: str, outlets: Iterable[str]) -> Optional[str]:
        try:
            import ollama
        except ImportError:  # pragma: no cover - optional dependency
            logger.debug("Ollama library not available for pattern summarization")
            return None

        client = ollama.AsyncClient(host=self.ollama_base_url)
        messages = self._build_messages(snippet, outlets)
        try:
            response = await client.chat(model=self.ollama_model, messages=messages)
            content = response["message"]["content"]
        except Exception as exc:  # pragma: no cover - network issues
            logger.warning("Ollama summarization failed: %s", exc)
            return None
        return self._parse_title(content)

    async def _summarize_with_openai(self, snippet: str, outlets: Iterable[str]) -> Optional[str]:
        if not self.openai_model or not self.openai_key:
            return None
        try:
            from openai import AsyncOpenAI
        except ImportError:  # pragma: no cover - optional dependency
            return None

        client = AsyncOpenAI(api_key=self.openai_key)
        messages = self._build_messages(snippet, outlets)
        try:
            response = await client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0,
            )
            content = response.choices[0].message.content
        except Exception:  # pragma: no cover - API errors
            return None
        return self._parse_title(content)

    @staticmethod
    def _build_messages(snippet: str, outlets: Iterable[str]) -> list[dict[str, str]]:
        outlets_str = ", ".join(outlets)
        user_prompt = (
            f"Repeated snippet (French): {snippet}\n"
            f"Observed outlets: {outlets_str}\n"
            "Provide a concise English title summarizing how the quote is reused."
        )
        return [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _parse_title(content: str | None) -> Optional[str]:
        import json

        if not content:
            return None
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        title = data.get("title")
        if not title:
            return None
        return title.strip()

    @staticmethod
    def _fallback(snippet: str, outlets: Iterable[str]) -> str:
        outlets_list = ", ".join(outlets)
        preview = snippet[:80].strip()
        return f"Shared line across {outlets_list}: {preview}"

