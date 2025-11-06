"""Relationship classification using LLMs with deterministic guardrails."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml

logger = logging.getLogger("holocron.llm")


@dataclass(slots=True)
class RelationshipDefinition:
    """Supported relationship definition loaded from YAML."""

    name: str
    source_type: str
    target_type: str
    description: str


@dataclass(slots=True)
class RelationshipPrompt:
    """Structured payload for relationship classification."""

    source_text: str
    source_type: str
    target_text: str
    target_type: str
    context: str


def load_relationships(path: Path | str = "relationships.yaml") -> list[RelationshipDefinition]:
    """Load relationship definitions from YAML."""

    file_path = Path(path)
    if not file_path.exists():  # pragma: no cover - optional path
        return []
    data = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    relationships = data.get("relationships", []) if isinstance(data, dict) else []
    return [RelationshipDefinition(**item) for item in relationships]


SYSTEM_PROMPT = (
    "You are a strict classifier for news relationship extraction. "
    "Only respond with JSON: {\"label\": <label or null>} using allowed labels."  # noqa: E501
)


class RelationshipClassifier:
    """Classify relationships with an Ollama-first strategy."""

    def __init__(
        self,
        ollama_model: str,
        ollama_base_url: str,
        *,
        openai_model: Optional[str] = None,
        openai_key: Optional[str] = None,
        openrouter_model: Optional[str] = None,
        openrouter_key: Optional[str] = None,
        schema: Optional[Iterable[RelationshipDefinition]] = None,
    ) -> None:
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.openai_model = openai_model
        self.openai_key = openai_key
        self.openrouter_model = openrouter_model
        self.openrouter_key = openrouter_key
        self.schema = list(schema or [])

    @property
    def allowed_labels(self) -> list[str]:
        if not self.schema:
            return []
        return [definition.name for definition in self.schema]

    def labels_for_types(self, source_type: str, target_type: str) -> list[str]:
        """Return labels compatible with the provided type pairing."""

        matches = [
            definition.name
            for definition in self.schema
            if definition.source_type == source_type and definition.target_type == target_type
        ]
        return matches or self.allowed_labels

    async def classify(self, prompt: RelationshipPrompt, allowed_labels: list[str]) -> Optional[str]:
        """Return a relationship label or None."""

        if not allowed_labels:
            return None

        logger.debug(
            "Classifying relationship: %s (%s) -> %s (%s)",
            prompt.source_text,
            prompt.source_type,
            prompt.target_text,
            prompt.target_type,
        )

        tasks = [self._classify_with_ollama(prompt, allowed_labels)]
        if self.openai_key and self.openai_model:
            tasks.append(self._classify_with_openai(prompt, allowed_labels))
        if self.openrouter_key and self.openrouter_model:
            tasks.append(self._classify_with_openrouter(prompt, allowed_labels))

        for coroutine in tasks:
            label = await coroutine
            if label:
                logger.debug("Classifier returned label: %s", label)
                return label

        label = self._heuristic_label(prompt, allowed_labels)
        if label:
            logger.debug("Heuristic fallback returned label: %s", label)
        else:
            logger.debug("No relationship detected")
        return label

    async def _classify_with_ollama(
        self, prompt: RelationshipPrompt, allowed_labels: list[str]
    ) -> Optional[str]:
        try:
            import ollama
        except ImportError:  # pragma: no cover - optional dependency
            logger.debug("Ollama library not available")
            return None

        messages = self._build_messages(prompt, allowed_labels)
        client = ollama.AsyncClient(host=self.ollama_base_url)
        try:
            logger.debug("Calling Ollama with model: %s", self.ollama_model)
            response = await client.chat(model=self.ollama_model, messages=messages)
            content = response["message"]["content"]
            return self._parse_label(content, allowed_labels)
        except Exception as exc:  # pragma: no cover - network errors
            logger.warning("Ollama classification failed: %s", exc)
            return None

    async def _classify_with_openai(
        self, prompt: RelationshipPrompt, allowed_labels: list[str]
    ) -> Optional[str]:
        if not self.openai_key or not self.openai_model:
            return None
        try:
            from openai import AsyncOpenAI
        except ImportError:  # pragma: no cover - optional dependency
            return None

        client = AsyncOpenAI(api_key=self.openai_key)
        messages = self._build_messages(prompt, allowed_labels)
        try:
            response = await client.chat.completions.create(
                model=self.openai_model,
                messages=messages,
                temperature=0.0,
            )
            content = response.choices[0].message.content
            if not content:
                return None
        except Exception:  # pragma: no cover - API errors
            return None
        return self._parse_label(content, allowed_labels)

    async def _classify_with_openrouter(
        self, prompt: RelationshipPrompt, allowed_labels: list[str]
    ) -> Optional[str]:
        if not self.openrouter_key or not self.openrouter_model:
            return None
        import httpx

        messages = self._build_messages(prompt, allowed_labels)
        payload = {
            "model": self.openrouter_model,
            "messages": messages,
        }
        headers = {
            "Authorization": f"Bearer {self.openrouter_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions", json=payload, headers=headers
                )
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
            except Exception:  # pragma: no cover - network errors
                return None
        return self._parse_label(content, allowed_labels)

    def _heuristic_label(
        self, prompt: RelationshipPrompt, allowed_labels: list[str]
    ) -> Optional[str]:
        context = prompt.context.lower()
        heuristics = {
            "works_for": ["travaille pour", "works for", "président de", "dirige"],
            "met_with": ["a rencontré", "met with", "s'est entretenu"],
            "located_in": ["basé à", "based in", "situé à"],
            "announced": ["a annoncé", "announced"],
            "criticized": ["a critiqué", "criticized"],
        }
        for label, patterns in heuristics.items():
            if label not in allowed_labels:
                continue
            if any(pattern in context for pattern in patterns):
                return label
        return None

    @staticmethod
    def _build_messages(prompt: RelationshipPrompt, allowed_labels: list[str]) -> list[dict[str, str]]:
        allowed = ", ".join(allowed_labels)
        user_content = (
            "Allowed labels: "
            f"{allowed}. Respond with JSON. \n"
            f"Source: {prompt.source_text} ({prompt.source_type}). "
            f"Target: {prompt.target_text} ({prompt.target_type}). "
            f"Context: {prompt.context}"
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    @staticmethod
    def _parse_label(content: str, allowed_labels: list[str]) -> Optional[str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None
        label = data.get("label")
        if label in allowed_labels:
            return label
        return None


def build_classifier_from_settings(settings) -> RelationshipClassifier:
    schema = load_relationships()
    return RelationshipClassifier(
        ollama_model=settings.ollama_model,
        ollama_base_url=settings.ollama_base_url,
        openai_model=settings.openai_model,
        openai_key=settings.openai_api_key,
        openrouter_model=settings.openrouter_model,
        openrouter_key=settings.openrouter_api_key,
        schema=schema,
    )



