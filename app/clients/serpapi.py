"""SerpAPI web search client for fetching public data about contacts.

Used by the deep profiler to gather real web results (news articles,
LinkedIn posts, company pages, conference talks) before LLM synthesis.
Gracefully degrades when the API key is not configured.

API docs: https://serpapi.com/search-api
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"


class SerpAPIClient:
    """Async client for SerpAPI Google search."""

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or settings.serpapi_api_key
        if not self.api_key:
            logger.warning("SerpAPI key not configured – web search disabled")

    async def search(
        self,
        query: str,
        num: int = 10,
        engine: str = "google",
    ) -> list[dict[str, Any]]:
        """Run a web search. Returns a list of organic result dicts."""
        if not self.api_key:
            return []

        params = {
            "q": query,
            "api_key": self.api_key,
            "engine": engine,
            "num": num,
        }

        try:
            async with httpx.AsyncClient(timeout=20) as client:
                resp = await client.get(SERPAPI_URL, params=params)
                if resp.status_code == 403:
                    logger.warning("SerpAPI auth failed – check API key")
                    return []
                if resp.status_code == 429:
                    logger.warning("SerpAPI rate limited")
                    return []
                if resp.status_code != 200:
                    logger.warning(
                        "SerpAPI error %d: %s", resp.status_code, resp.text[:200]
                    )
                    return []
                data = resp.json()
                return data.get("organic_results", [])
        except Exception:
            logger.exception("SerpAPI search failed for: %s", query)
            return []

    async def search_person(
        self,
        name: str,
        company: str = "",
        title: str = "",
        linkedin_url: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        """Run multiple targeted searches for a person and return categorised results.

        Returns a dict with keys: general, linkedin, news, talks
        """
        if not self.api_key:
            return {"general": [], "linkedin": [], "news": [], "talks": []}

        queries: list[tuple[str, str]] = []

        # Primary identity search
        q_parts = [f'"{name}"']
        if company:
            q_parts.append(f'"{company}"')
        queries.append(("general", " ".join(q_parts)))

        # LinkedIn-specific search
        linkedin_q = f'"{name}" site:linkedin.com'
        if company:
            linkedin_q += f' "{company}"'
        queries.append(("linkedin", linkedin_q))

        # News coverage
        news_parts = [f'"{name}"']
        if company:
            news_parts.append(f'"{company}"')
        queries.append(("news", " ".join(news_parts) + " news OR interview OR article"))

        # Conference talks / podcasts
        talks_parts = [f'"{name}"']
        if company:
            talks_parts.append(company)
        queries.append(
            ("talks", " ".join(talks_parts) + " conference OR podcast OR keynote OR talk")
        )

        results: dict[str, list[dict[str, Any]]] = {}
        for category, query in queries:
            hits = await self.search(query, num=8)
            results[category] = [_normalize_result(r) for r in hits]
            # Brief pause between searches to respect rate limits
            await asyncio.sleep(0.3)

        return results


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract the fields we need from a SerpAPI organic result."""
    return {
        "title": result.get("title", ""),
        "link": result.get("link", ""),
        "snippet": result.get("snippet", ""),
        "source": result.get("source", ""),
        "date": result.get("date", ""),
    }


def format_web_results_for_prompt(
    results: dict[str, list[dict[str, Any]]],
) -> str:
    """Format categorised search results into a text block for the LLM prompt.

    Returns an empty string if no results were found.
    """
    sections: list[str] = []

    total = sum(len(v) for v in results.values())
    if total == 0:
        return ""

    sections.append(
        "The following web search results were retrieved in real-time. "
        "Use these as primary sources for your analysis. "
        "Cite the source URL when referencing a specific result."
    )

    category_labels = {
        "general": "General Search Results",
        "linkedin": "LinkedIn Results",
        "news": "News & Articles",
        "talks": "Conference Talks & Podcasts",
    }

    for category, label in category_labels.items():
        items = results.get(category, [])
        if not items:
            continue
        sections.append(f"\n**{label}:**")
        for i, item in enumerate(items, 1):
            line = f"{i}. **{item['title']}**"
            if item.get("source"):
                line += f" ({item['source']})"
            if item.get("date"):
                line += f" [{item['date']}]"
            line += f"\n   URL: {item['link']}"
            if item.get("snippet"):
                line += f"\n   > {item['snippet']}"
            sections.append(line)

    return "\n".join(sections)
