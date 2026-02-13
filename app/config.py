"""Centralised configuration loaded from environment variables / .env file."""

from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Fireflies
    fireflies_api_key: str = ""

    # Apollo.io (contact enrichment)
    apollo_api_key: str = ""

    # Gmail / Google OAuth
    gmail_credentials_path: str = "./credentials.json"
    gmail_token_path: str = "./token.json"
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    # API authentication
    briefing_api_key: str = ""

    # Database
    database_url: str = "sqlite:///./briefing_engine.db"

    # Output
    output_dir: Path = Path("./out")

    # Logging
    log_level: str = "INFO"

    # Retrieval window
    retrieval_window_days: int = 90

    @property
    def is_sqlite(self) -> bool:
        return self.database_url.startswith("sqlite")

    @property
    def effective_database_url(self) -> str:
        """Return a SQLAlchemy-compatible URL.

        Railway injects ``postgres://`` but SQLAlchemy 2.0+ requires
        ``postgresql://``.
        """
        url = self.database_url
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        return url


settings = Settings()


def validate_config() -> list[str]:
    """Check configuration and return a list of warnings.

    Logs each warning. Returns the list so callers can surface them.
    """
    warnings: list[str] = []

    if not settings.openai_api_key:
        warnings.append(
            "OPENAI_API_KEY is not set – brief generation and embeddings will fail"
        )
    if not settings.fireflies_api_key:
        warnings.append(
            "FIREFLIES_API_KEY is not set – Fireflies ingestion will be skipped"
        )

    has_gmail_oauth_file = (
        settings.gmail_credentials_path
        and Path(settings.gmail_credentials_path).exists()
    )
    has_gmail_env = settings.google_client_id and settings.google_refresh_token
    if not has_gmail_oauth_file and not has_gmail_env:
        warnings.append(
            "Gmail not configured – set GOOGLE_CLIENT_ID + GOOGLE_REFRESH_TOKEN "
            "or provide a credentials.json file"
        )

    if not settings.briefing_api_key:
        warnings.append(
            "BRIEFING_API_KEY is not set – API endpoints are unauthenticated"
        )

    for w in warnings:
        logger.warning("Config: %s", w)

    return warnings
