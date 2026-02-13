"""Apollo.io People Enrichment API client.

Enriches contacts with photo URLs, LinkedIn profiles, job titles,
company data, and more.

API docs: https://docs.apollo.io/reference/people-enrichment
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

APOLLO_ENRICH_URL = "https://api.apollo.io/api/v1/people/match"


class ApolloClient:
    """Async client for the Apollo.io People Enrichment API."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.apollo_api_key
        if not self.api_key:
            logger.warning("Apollo API key not configured – enrichment disabled")
        self.headers = {
            "x-api-key": self.api_key or "",
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
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
            logger.debug("Apollo: skipping enrichment – no email or name provided")
            return None

        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    APOLLO_ENRICH_URL, json=payload, headers=self.headers
                )
                if resp.status_code == 429:
                    logger.warning("Apollo rate limit hit")
                    return None
                if resp.status_code != 200:
                    logger.warning(
                        "Apollo returned %d for %s: %s",
                        resp.status_code,
                        email or name,
                        resp.text[:200],
                    )
                    return None
                data = resp.json()
                person = data.get("person")
                if person:
                    logger.info(
                        "Apollo matched: %s -> %s (%s)",
                        email or name,
                        person.get("name", "?"),
                        person.get("title", "?"),
                    )
                else:
                    logger.info("Apollo: no match for %s", email or name)
                return person
        except Exception:
            logger.exception("Apollo enrichment failed for %s", email or name)
            return None

    async def enrich_many(
        self, contacts: list[dict[str, str | None]],
        name_variants: dict[str, list[str]] | None = None,
    ) -> list[dict[str, Any] | None]:
        """Enrich multiple contacts sequentially with rate-limit awareness.

        Each contact dict should have: email, name, company (all optional).
        Uses single-person endpoint for reliability.
        If a lookup fails and name_variants are provided, tries expanded names.
        """
        if not self.api_key or not contacts:
            return [None] * len(contacts)

        name_variants = name_variants or {}
        results: list[dict[str, Any] | None] = []

        for i, contact in enumerate(contacts):
            email = contact.get("email")
            name = contact.get("name") or ""
            company = contact.get("company")

            # Build the best possible lookup
            parts = name.split(None, 1) if name else []
            first = parts[0] if parts else None
            last = parts[1] if len(parts) > 1 else None

            # Extract domain from email for company matching
            domain = None
            if email and "@" in email:
                domain = email.split("@")[1]
                if domain in {
                    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
                    "me.com", "icloud.com", "protonmail.com",
                }:
                    domain = None

            person = await self.enrich_person(
                email=email,
                first_name=first,
                last_name=last,
                domain=domain,
                organization_name=company,
            )

            # If no match and we have name variants, try expanded names
            if not person and first and last:
                first_lower = first.lower()
                variants = name_variants.get(first_lower, [])
                for variant in variants:
                    logger.info(
                        "Apollo: retrying %s %s as %s %s",
                        first, last, variant.title(), last,
                    )
                    await asyncio.sleep(0.3)
                    person = await self.enrich_person(
                        first_name=variant.title(),
                        last_name=last,
                        domain=domain,
                        organization_name=company,
                    )
                    if person:
                        break

            results.append(person)

            # Brief pause between requests to respect rate limits
            if i < len(contacts) - 1:
                await asyncio.sleep(0.5)

        return results


def normalize_enrichment(person: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the fields we care about from an Apollo person record."""
    if not person:
        return {}

    org = person.get("organization") or {}

    enriched = {
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

    # Only count as a real enrichment if we got at least some useful data
    has_data = any([
        enriched["photo_url"],
        enriched["linkedin_url"],
        enriched["title"],
    ])
    if not has_data:
        return {}

    return enriched
