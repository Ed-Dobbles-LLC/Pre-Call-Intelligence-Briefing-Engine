"""Fireflies.ai GraphQL API client.

Thin wrapper around the Fireflies API.  All methods return raw dicts;
normalisation is handled by the ingest layer.
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
# GraphQL fragments
# ---------------------------------------------------------------------------

_TRANSCRIPT_FIELDS = """
    id
    title
    date
    duration
    organizer_email
    participants
    summary {
        overview
        shorthand_bullet
        action_items
    }
    sentences {
        speaker_name
        text
        start_time
        end_time
    }
"""

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
                logger.error("Fireflies GraphQL errors: %s", data["errors"])
            return data.get("data", {})

    async def list_transcripts(self, limit: int = 50, skip: int = 0) -> list[dict]:
        data = await self._post(QUERY_TRANSCRIPTS, {"limit": limit, "skip": skip})
        return data.get("transcripts", [])

    async def get_transcript(self, transcript_id: str) -> dict | None:
        data = await self._post(QUERY_TRANSCRIPT_BY_ID, {"id": transcript_id})
        return data.get("transcript")

    async def search_transcripts(
        self,
        participant_email: str | None = None,
        participant_name: str | None = None,
        since: datetime | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Fetch transcripts and filter client-side by participant or date.

        The Fireflies API doesn't expose rich server-side search, so we pull
        recent transcripts and filter locally.
        """
        all_transcripts: list[dict] = []
        skip = 0
        batch = 50
        while len(all_transcripts) < limit:
            batch_data = await self.list_transcripts(limit=batch, skip=skip)
            if not batch_data:
                break
            all_transcripts.extend(batch_data)
            skip += batch

        results: list[dict] = []
        for t in all_transcripts:
            # Date filter
            t_date = t.get("date")
            if since and t_date:
                try:
                    # Fireflies returns epoch ms or ISO string
                    if isinstance(t_date, (int, float)):
                        t_dt = datetime.utcfromtimestamp(t_date / 1000)
                    else:
                        t_dt = datetime.fromisoformat(str(t_date))
                    if t_dt < since:
                        continue
                except (ValueError, TypeError):
                    pass

            # Participant filter
            participants = t.get("participants", []) or []
            if participant_email:
                if participant_email.lower() not in [
                    p.lower() for p in participants if isinstance(p, str)
                ]:
                    continue
            if participant_name:
                # Check participant names in sentences
                sentences = t.get("sentences", []) or []
                speaker_names = {
                    s.get("speaker_name", "").lower()
                    for s in sentences
                    if s.get("speaker_name")
                }
                if participant_name.lower() not in speaker_names:
                    # Also check title
                    title = (t.get("title") or "").lower()
                    if participant_name.lower() not in title:
                        continue

            results.append(t)
            if len(results) >= limit:
                break

        return results
