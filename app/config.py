"""Centralised configuration loaded from environment variables / .env file."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


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

    # Gmail / Google OAuth
    gmail_credentials_path: str = "./credentials.json"
    gmail_token_path: str = "./token.json"
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

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
