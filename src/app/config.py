"""Application configuration handling."""

from functools import lru_cache
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the News Intelligence service."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"

    database_url: str = "sqlite+aiosqlite:///./news.db"

    scheduler_enabled: bool = True
    ingest_feeds: List[str] = ["fr/lemonde"]
    ingest_interval_minutes: int = 15

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"

    openrouter_api_key: Optional[str] = None
    openrouter_model: str = "anthropic/claude-3-haiku"

    api_key: Optional[str] = None


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()


