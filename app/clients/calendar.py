"""Google Calendar API client for meeting ingestion.

Fetches calendar events for the next 7 days, normalizes them into
a standard format, and extracts attendee information for contact matching.

Requires Google Calendar read-only scope:
  https://www.googleapis.com/auth/calendar.readonly

Authentication reuses the same OAuth credentials as Gmail
(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_REFRESH_TOKEN).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)

CALENDAR_SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]


def _get_calendar_service():
    """Build a Google Calendar API v3 service.

    Uses environment-based OAuth (Railway) or file-based OAuth (local dev).
    Raises RuntimeError if credentials are not configured.
    """
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError:
        raise RuntimeError(
            "google-api-python-client is required for Calendar integration. "
            "Install with: pip install google-api-python-client google-auth"
        )

    creds = None

    # Method 1: Environment variable-based OAuth (Railway deployment)
    if (
        settings.google_client_id
        and settings.google_client_secret
        and settings.google_refresh_token
    ):
        creds = Credentials(
            token=None,
            refresh_token=settings.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=settings.google_client_id,
            client_secret=settings.google_client_secret,
            scopes=CALENDAR_SCOPES,
        )
    else:
        raise RuntimeError(
            "Google Calendar requires GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, "
            "and GOOGLE_REFRESH_TOKEN environment variables."
        )

    return build("calendar", "v3", credentials=creds)


class CalendarClient:
    """Fetches and normalizes Google Calendar events."""

    def __init__(self) -> None:
        self.service = None

    def _ensure_service(self) -> None:
        if self.service is None:
            self.service = _get_calendar_service()

    def fetch_upcoming_events(
        self,
        days: int = 7,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        """Fetch calendar events from now to now + days.

        Returns a list of normalized event dicts.
        """
        self._ensure_service()

        now = datetime.now(timezone.utc)
        end = now + timedelta(days=days)

        try:
            events_result = self.service.events().list(
                calendarId="primary",
                timeMin=now.isoformat(),
                timeMax=end.isoformat(),
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()
        except Exception:
            logger.exception("Failed to fetch calendar events")
            return []

        raw_events = events_result.get("items", [])
        logger.info("Fetched %d calendar events for next %d days", len(raw_events), days)

        return [self._normalize_event(e) for e in raw_events]

    def _normalize_event(self, event: dict) -> dict[str, Any]:
        """Normalize a raw Google Calendar event into our standard format."""
        start = event.get("start", {})
        end = event.get("end", {})
        start_time = start.get("dateTime") or start.get("date", "")
        end_time = end.get("dateTime") or end.get("date", "")

        attendees = []
        for att in event.get("attendees", []):
            email = att.get("email", "").lower().strip()
            if not email:
                continue
            attendees.append({
                "email": email,
                "name": att.get("displayName", ""),
                "response_status": att.get("responseStatus", "needsAction"),
                "organizer": att.get("organizer", False),
                "self": att.get("self", False),
            })

        organizer = event.get("organizer", {})
        conference_link = ""
        entry_points = event.get("conferenceData", {}).get("entryPoints", [])
        for ep in entry_points:
            if ep.get("entryPointType") == "video":
                conference_link = ep.get("uri", "")
                break

        return {
            "id": event.get("id", ""),
            "recurring_event_id": event.get("recurringEventId", ""),
            "title": event.get("summary", "Untitled"),
            "description": (event.get("description") or "")[:2000],
            "start_time": start_time,
            "end_time": end_time,
            "attendees": attendees,
            "organizer_email": organizer.get("email", "").lower(),
            "organizer_name": organizer.get("displayName", ""),
            "conference_link": conference_link,
            "location": event.get("location", ""),
            "html_link": event.get("htmlLink", ""),
            "status": event.get("status", "confirmed"),
            "created": event.get("created", ""),
            "updated": event.get("updated", ""),
        }


def normalize_event_for_storage(event: dict) -> dict:
    """Convert a normalized event dict to a storage-ready format."""
    return {
        "calendar_event_id": event["id"],
        "recurring_event_id": event.get("recurring_event_id", ""),
        "title": event["title"],
        "description": event.get("description", "")[:500],
        "start_time": event["start_time"],
        "end_time": event["end_time"],
        "attendee_emails": [a["email"] for a in event.get("attendees", [])],
        "organizer_email": event.get("organizer_email", ""),
        "conference_link": event.get("conference_link", ""),
        "location": event.get("location", ""),
        "status": "UPCOMING",
    }
