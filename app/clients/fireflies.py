"""Fireflies.ai GraphQL API client.

Thin wrapper around the Fireflies API.  All methods return raw dicts;
normalisation is handled by the ingest layer.

API docs: https://docs.fireflies.ai/graphql-api/query/transcripts
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

FIREFLIES_GQL_URL = "https://api.fireflies.ai/graphql"

# ---------------------------------------------------------------------------
# GraphQL fragments – expanded to match full Fireflies schema
# ---------------------------------------------------------------------------

_TRANSCRIPT_FIELDS = """
    id
    title
    date
    duration
    organizer_email
    participants
    transcript_url
    meeting_attendees {
        displayName
        email
        name
    }
    summary {
        overview
        shorthand_bullet
        action_items
        short_summary
        keywords
    }
    sentences {
        index
        speaker_name
        speaker_id
        text
        raw_text
        start_time
        end_time
    }
"""

# Server-side filtered query using Fireflies native parameters
QUERY_TRANSCRIPTS_FILTERED = """
query TranscriptsFiltered(
    $limit: Int,
    $skip: Int,
    $fromDate: DateTime,
    $toDate: DateTime
) {
    transcripts(
        limit: $limit,
        skip: $skip,
        fromDate: $fromDate,
        toDate: $toDate
    ) {
        %s
    }
}
""" % _TRANSCRIPT_FIELDS

QUERY_TRANSCRIPTS = """
query Transcripts($limit: Int, $skip: Int) {
    transcripts(limit: $limit, skip: $skip) {
        %s
    }
}
""" % _TRANSCRIPT_FIELDS

QUERY_TRANSCRIPT_BY_ID = """
query Transcript($id: String!) {
    transcript(id: $id) {
        %s
    }
}
""" % _TRANSCRIPT_FIELDS


class FirefliesClient:
    """Async client for the Fireflies.ai GraphQL API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.fireflies_api_key
        if not self.api_key:
            logger.warning("Fireflies API key not configured – client will return empty results")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def _post(self, query: str, variables: dict | None = None) -> dict[str, Any]:
        if not self.api_key:
            return {}
        payload: dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(FIREFLIES_GQL_URL, json=payload, headers=self.headers)
            resp.raise_for_status()
            data = resp.json()
            if "errors" in data:
                error_msgs = [e.get("message", str(e)) for e in data["errors"]]
                logger.error("Fireflies GraphQL errors: %s", error_msgs)
                raise RuntimeError(f"Fireflies API error: {'; '.join(error_msgs)}")
            return data.get("data", {})

    async def list_transcripts(self, limit: int = 50, skip: int = 0) -> list[dict]:
        """List recent transcripts (no filters)."""
        data = await self._post(QUERY_TRANSCRIPTS, {"limit": limit, "skip": skip})
        return data.get("transcripts", [])

    async def get_transcript(self, transcript_id: str) -> dict | None:
        """Fetch a single transcript by ID."""
        data = await self._post(QUERY_TRANSCRIPT_BY_ID, {"id": transcript_id})
        return data.get("transcript")

    async def search_transcripts(
        self,
        participant_email: str | None = None,
        participant_name: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        keyword: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Search transcripts using server-side filters where possible,
        then refine client-side for name-based matching.

        Server-side filters (fast, handled by Fireflies API):
          - participants: filter by attendee email
          - fromDate / toDate: date range
          - keyword: search titles and spoken words

        Client-side filters (applied after fetch):
          - participant_name: match against speaker_names, meeting_attendees, title
        """
        variables: dict[str, Any] = {"limit": min(limit, 50), "skip": 0}

        # Server-side: date range (Fireflies expects ISO 8601: YYYY-MM-DDTHH:mm:ss.sssZ)
        if since:
            variables["fromDate"] = since.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if until:
            variables["toDate"] = until.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        all_transcripts: list[dict] = []
        skip = 0
        while len(all_transcripts) < limit:
            variables["skip"] = skip
            variables["limit"] = min(50, limit - len(all_transcripts))
            data = await self._post(QUERY_TRANSCRIPTS_FILTERED, variables)
            batch = data.get("transcripts") or []
            if not batch:
                break
            all_transcripts.extend(batch)
            skip += len(batch)

        # Client-side: filter by participant email
        if participant_email:
            email_lower = participant_email.lower()
            all_transcripts = [
                t for t in all_transcripts
                if email_lower in [
                    (a.get("email") or "").lower()
                    for a in (t.get("meeting_attendees") or [])
                ] or email_lower in [
                    (p or "").lower() for p in (t.get("participants") or [])
                ]
            ]

        # Client-side: name-based matching
        if participant_name:
            name_lower = participant_name.lower()
            all_transcripts = [
                t for t in all_transcripts
                if _transcript_mentions_name(t, name_lower)
            ]

        return all_transcripts[:limit]


def _transcript_mentions_name(transcript: dict, name_lower: str) -> bool:
    """Check if a transcript mentions a person by name."""
    # Check meeting_attendees (structured, reliable)
    for attendee in transcript.get("meeting_attendees") or []:
        for field in ("displayName", "name", "email"):
            val = (attendee.get(field) or "").lower()
            if name_lower in val:
                return True

    # Check speaker names in sentences
    for sentence in transcript.get("sentences") or []:
        speaker = (sentence.get("speaker_name") or "").lower()
        if name_lower in speaker:
            return True

    # Check title
    if name_lower in (transcript.get("title") or "").lower():
        return True

    return False
