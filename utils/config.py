"""Project configuration, loaded from environment variables / .env."""
from __future__ import annotations

from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_api_key: Optional[str] = None
    llm_base_url: Optional[str] = None
    llm_model: Optional[str] = None

    database_url: Optional[str] = None

    log_level: str = "INFO"
    log_file: Optional[str] = None


settings = Settings()


def reload_settings() -> Settings:
    """Re-read ``.env`` and return a fresh :class:`Settings` object.

    Also replaces the module-level ``settings``, so code that reads
    ``config.settings`` picks up the new values. Modules that did
    ``from utils.config import settings`` keep their old object.
    """
    global settings
    settings = Settings()
    return settings
