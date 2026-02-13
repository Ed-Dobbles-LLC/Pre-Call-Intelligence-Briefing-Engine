"""Gmail API client (read-only).

Uses the Google API Python client with OAuth2 credentials.
"""

from __future__ import annotations

import base64
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


def _get_gmail_service():
    """Build and return an authenticated Gmail API service.

    Supports two authentication modes:
    1. **Environment variables** (for cloud/Railway): Set GOOGLE_CLIENT_ID,
       GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN.
    2. **File-based** (for local dev): Provide credentials.json and token.json.
    """
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
    except ImportError:
        logger.error(
            "Google API libraries not installed. "
            "Run: pip install google-auth google-auth-oauthlib google-api-python-client"
        )
        return None

    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    creds = None

    # Mode 1: Build credentials from environment variables
    if settings.google_client_id and settings.google_client_secret and settings.google_refresh_token:
        logger.info("Using env-var-based Google OAuth credentials")
        creds = Credentials(
            token=None,
            refresh_token=settings.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=SCOPES,
        )
        creds.refresh(Request())
        return build("gmail", "v1", credentials=creds)

    # Mode 2: File-based credentials (local dev)
    token_path = Path(settings.gmail_token_path)
    creds_path = Path(settings.gmail_credentials_path)

    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not creds_path.exists():
                logger.warning(
                    "Gmail credentials not configured – set GOOGLE_CLIENT_ID, "
                    "GOOGLE_CLIENT_SECRET, and GOOGLE_REFRESH_TOKEN env vars, "
                    "or provide %s for local OAuth flow",
                    creds_path,
                )
                return None
            flow = InstalledAppFlow.from_client_secrets_file(str(creds_path), SCOPES)
            creds = flow.run_local_server(port=0)
        token_path.write_text(creds.to_json())

    return build("gmail", "v1", credentials=creds)


class GmailClient:
    """Synchronous wrapper around the Gmail read-only API."""

    def __init__(self):
        self.service = _get_gmail_service()
        if not self.service:
            logger.warning("Gmail service not available – client will return empty results")

    def search_messages(
        self,
        query: str,
        max_results: int = 50,
    ) -> list[dict[str, Any]]:
        """Search Gmail with the given query string and return full messages."""
        if not self.service:
            return []

        try:
            response = (
                self.service.users()
                .messages()
                .list(userId="me", q=query, maxResults=max_results)
                .execute()
            )
        except Exception:
            logger.exception("Gmail search failed for query: %s", query)
            return []

        message_ids = [m["id"] for m in response.get("messages", [])]
        messages = []
        for msg_id in message_ids:
            try:
                msg = (
                    self.service.users()
                    .messages()
                    .get(userId="me", id=msg_id, format="full")
                    .execute()
                )
                messages.append(msg)
            except Exception:
                logger.exception("Failed to fetch Gmail message %s", msg_id)
        return messages

    def search_by_person(
        self,
        email: str | None = None,
        name: str | None = None,
        since_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Search for emails involving a person (by email or name) in the last N days."""
        if not self.service:
            return []

        parts = []
        if email:
            parts.append(f"(from:{email} OR to:{email})")
        if name:
            parts.append(f'("{name}")')

        since = datetime.utcnow() - timedelta(days=since_days)
        date_str = since.strftime("%Y/%m/%d")
        parts.append(f"after:{date_str}")

        query = " ".join(parts)
        logger.info("Gmail query: %s", query)
        return self.search_messages(query)

    def search_by_company(
        self,
        domain: str | None = None,
        company_name: str | None = None,
        since_days: int = 90,
    ) -> list[dict[str, Any]]:
        """Search for emails involving a company (by domain or name)."""
        if not self.service:
            return []

        parts = []
        if domain:
            parts.append(f"(from:@{domain} OR to:@{domain})")
        if company_name:
            parts.append(f'("{company_name}")')

        since = datetime.utcnow() - timedelta(days=since_days)
        date_str = since.strftime("%Y/%m/%d")
        parts.append(f"after:{date_str}")

        query = " ".join(parts)
        logger.info("Gmail query: %s", query)
        return self.search_messages(query)

    @staticmethod
    def extract_body(message: dict) -> str:
        """Extract the plain-text body from a Gmail message payload."""
        payload = message.get("payload", {})

        # Simple single-part message
        if payload.get("mimeType") == "text/plain" and payload.get("body", {}).get("data"):
            return base64.urlsafe_b64decode(payload["body"]["data"]).decode("utf-8", errors="replace")

        # Multipart
        for part in payload.get("parts", []):
            if part.get("mimeType") == "text/plain" and part.get("body", {}).get("data"):
                return base64.urlsafe_b64decode(part["body"]["data"]).decode(
                    "utf-8", errors="replace"
                )

        return ""

    @staticmethod
    def extract_headers(message: dict) -> dict[str, str]:
        """Extract common headers from a Gmail message."""
        headers = {}
        for h in message.get("payload", {}).get("headers", []):
            name = h.get("name", "").lower()
            if name in ("from", "to", "cc", "subject", "date"):
                headers[name] = h.get("value", "")
        return headers
