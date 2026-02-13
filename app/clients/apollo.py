"""Apollo.io People Enrichment API client.

Enriches contacts with photo URLs, LinkedIn profiles, job titles,
company data, and more.

API docs: https://docs.apollo.io/reference/people-enrichment
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

APOLLO_ENRICH_URL = "https://api.apollo.io/api/v1/people/match"
APOLLO_BULK_ENRICH_URL = "https://api.apollo.io/api/v1/people/bulk_match"


class ApolloClient:
    """Async client for the Apollo.io People Enrichment API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.apollo_api_key
        if not self.api_key:
            logger.warning("Apollo API key not configured – enrichment disabled")
        self.headers = {
            "x-api-key": self.api_key or "",
            "Content-Type": "application/json",
        }

    async def enrich_person(
        self,
        email: str | None = None,
        first_name: str | None = None,
        last_name: str | None = None,
        name: str | None = None,
        domain: str | None = None,
        organization_name: str | None = None,
    ) -> dict[str, Any] | None:
        """Enrich a single person. Returns the person dict or None."""
        if not self.api_key:
            return None

        payload: dict[str, Any] = {
            "reveal_personal_emails": False,
            "reveal_phone_number": False,
        }
        if email:
            payload["email"] = email
        if first_name:
            payload["first_name"] = first_name
        if last_name:
            payload["last_name"] = last_name
        if name and not (first_name or last_name):
            payload["name"] = name
        if domain:
            payload["domain"] = domain
        if organization_name:
            payload["organization_name"] = organization_name

        if not email and not name and not (first_name and last_name):
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    APOLLO_ENRICH_URL, json=payload, headers=self.headers
                )
                if resp.status_code == 429:
                    logger.warning("Apollo rate limit hit")
                    return None
                resp.raise_for_status()
                data = resp.json()
                return data.get("person")
        except Exception:
            logger.exception("Apollo enrichment failed for %s", email or name)
            return None

    async def enrich_bulk(
        self, details: list[dict[str, str]]
    ) -> list[dict[str, Any] | None]:
        """Enrich up to 10 people at once. Returns list of person dicts."""
        if not self.api_key or not details:
            return [None] * len(details)

        results: list[dict[str, Any] | None] = []

        # Apollo bulk endpoint accepts max 10 per call
        for i in range(0, len(details), 10):
            batch = details[i : i + 10]
            try:
                async with httpx.AsyncClient(timeout=30) as client:
                    resp = await client.post(
                        APOLLO_BULK_ENRICH_URL,
                        json={
                            "details": batch,
                            "reveal_personal_emails": False,
                            "reveal_phone_number": False,
                        },
                        headers=self.headers,
                    )
                    if resp.status_code == 429:
                        logger.warning("Apollo rate limit hit on bulk enrichment")
                        results.extend([None] * len(batch))
                        continue
                    resp.raise_for_status()
                    data = resp.json()
                    matches = data.get("matches") or []
                    # Pad with None if fewer matches than requested
                    results.extend(matches)
                    results.extend([None] * (len(batch) - len(matches)))
            except Exception:
                logger.exception("Apollo bulk enrichment failed")
                results.extend([None] * len(batch))

        return results[: len(details)]


def normalize_enrichment(person: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the fields we care about from an Apollo person record."""
    if not person:
        return {}

    org = person.get("organization") or {}

    return {
        "photo_url": person.get("photo_url") or "",
        "linkedin_url": person.get("linkedin_url") or "",
        "title": person.get("title") or "",
        "headline": person.get("headline") or "",
        "seniority": person.get("seniority") or "",
        "city": person.get("city") or "",
        "state": person.get("state") or "",
        "country": person.get("country") or "",
        "company_name": org.get("name") or "",
        "company_industry": org.get("industry") or "",
        "company_size": org.get("estimated_num_employees"),
        "company_domain": org.get("primary_domain") or "",
        "company_linkedin": org.get("linkedin_url") or "",
    }
