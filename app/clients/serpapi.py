"""SerpAPI web search client with tiered retrieval and source classification.

Implements:
- Targeted query generation with search operators
- Source tier classification (Primary > Secondary > Low-quality)
- Identity disambiguation search battery
- Company registry and corporate searches
- Recency-aware result handling

API docs: https://serpapi.com/search-api
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search"

# The 10 mandatory public visibility categories for the sweep
VISIBILITY_CATEGORIES: list[str] = [
    "ted", "tedx", "keynote", "conference", "summit",
    "podcast", "webinar", "youtube_talk", "panel", "interview_video",
]


# ---------------------------------------------------------------------------
# Source Tier Classification
# ---------------------------------------------------------------------------

class SourceTier:
    """Source quality classification."""
    PRIMARY = 1    # Company site, registry, direct talks, official PR
    SECONDARY = 2  # Major news, conference pages, reputable blogs
    LOW = 3        # Forums, unverified sources, SEO content

# Domain-to-tier mapping
_TIER_MAP: dict[str, int] = {
    # Primary: company sites, registries, direct platforms
    "linkedin.com": SourceTier.PRIMARY,
    "companieshouse.gov.uk": SourceTier.PRIMARY,
    "find-and-update.company-information.service.gov.uk": SourceTier.PRIMARY,
    "sec.gov": SourceTier.PRIMARY,
    "crunchbase.com": SourceTier.PRIMARY,
    "businesswire.com": SourceTier.PRIMARY,
    "prnewswire.com": SourceTier.PRIMARY,
    "globenewswire.com": SourceTier.PRIMARY,
    # Secondary: major news, conferences
    "techcrunch.com": SourceTier.SECONDARY,
    "reuters.com": SourceTier.SECONDARY,
    "bloomberg.com": SourceTier.SECONDARY,
    "wsj.com": SourceTier.SECONDARY,
    "ft.com": SourceTier.SECONDARY,
    "forbes.com": SourceTier.SECONDARY,
    "bbc.co.uk": SourceTier.SECONDARY,
    "theguardian.com": SourceTier.SECONDARY,
    "venturebeat.com": SourceTier.SECONDARY,
    "zdnet.com": SourceTier.SECONDARY,
    "wired.com": SourceTier.SECONDARY,
    "youtube.com": SourceTier.SECONDARY,
    "github.com": SourceTier.SECONDARY,
    "medium.com": SourceTier.SECONDARY,
}


def classify_source_tier(url: str) -> int:
    """Classify a URL into a source tier (1=Primary, 2=Secondary, 3=Low)."""
    if not url:
        return SourceTier.LOW
    url_lower = url.lower()
    for domain, tier in _TIER_MAP.items():
        if domain in url_lower:
            return tier
    # Check if it's a company domain (not social media / generic)
    # Company domains are primary when they match the subject's company
    return SourceTier.SECONDARY  # Default to secondary for unknown domains


def _normalize_result(result: dict[str, Any]) -> dict[str, Any]:
    """Extract fields from a SerpAPI organic result with tier classification."""
    link = result.get("link", "")
    return {
        "title": result.get("title", ""),
        "link": link,
        "snippet": result.get("snippet", ""),
        "source": result.get("source", ""),
        "date": result.get("date", ""),
        "tier": classify_source_tier(link),
    }


class SerpAPIClient:
    """Async client for SerpAPI with targeted query generation."""

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

        Returns a dict with keys: general, linkedin, news, talks, company_site, registry
        """
        if not self.api_key:
            return {
                "general": [], "linkedin": [], "news": [],
                "talks": [], "company_site": [], "registry": [],
            }

        queries: list[tuple[str, str]] = []

        # 1. Primary identity search
        q_parts = [f'"{name}"']
        if company:
            q_parts.append(f'"{company}"')
        if title:
            q_parts.append(title)
        queries.append(("general", " ".join(q_parts)))

        # 2. LinkedIn-specific search
        linkedin_q = f'"{name}" site:linkedin.com'
        if company:
            linkedin_q += f' "{company}"'
        queries.append(("linkedin", linkedin_q))

        # 3. Company site bio search
        if company:
            # Try to find them on their company's website
            company_slug = re.sub(r"[^a-zA-Z0-9]", "", company.lower())
            queries.append((
                "company_site",
                f'"{name}" site:{company_slug}.com OR site:{company_slug}.io '
                f'OR site:{company_slug}.ai'
            ))

        # 4. News and interviews
        news_parts = [f'"{name}"']
        if company:
            news_parts.append(f'"{company}"')
        queries.append((
            "news",
            " ".join(news_parts) + " interview OR article OR profile OR announcement"
        ))

        # 5. Conference talks / podcasts / webinars
        talks_parts = [f'"{name}"']
        if company:
            talks_parts.append(company)
        queries.append((
            "talks",
            " ".join(talks_parts) + " podcast OR webinar OR conference OR keynote OR panel"
        ))

        # 6. Corporate registry (UK Companies House)
        queries.append((
            "registry",
            f'"{name}" site:find-and-update.company-information.service.gov.uk'
        ))

        results: dict[str, list[dict[str, Any]]] = {}
        for category, query in queries:
            hits = await self.search(query, num=8)
            results[category] = [_normalize_result(r) for r in hits]
            await asyncio.sleep(0.3)

        return results

    async def search_public_visibility(
        self,
        name: str,
        company: str = "",
    ) -> dict[str, list[dict[str, Any]]]:
        """Run 10 targeted searches for public speaking/visibility artifacts.

        Returns a dict keyed by visibility category:
        ted, tedx, keynote, conference, summit, podcast, webinar,
        youtube_talk, panel, interview_video
        """
        if not self.api_key:
            return {cat: [] for cat in VISIBILITY_CATEGORIES}

        base = f'"{name}"'
        if company:
            base += f' "{company}"'

        queries: list[tuple[str, str]] = [
            ("ted", f'{base} site:ted.com OR "TED talk"'),
            ("tedx", f'{base} "TEDx" OR site:tedx.com'),
            ("keynote", f'{base} "keynote" "speaker" OR "keynote address"'),
            ("conference", f'{base} "conference" "speaker" OR "spoke at" OR "presented at"'),
            ("summit", f'{base} "summit" "speaker" OR "summit keynote"'),
            ("podcast", f'{base} "podcast" "guest" OR "episode" OR "hosted by"'),
            ("webinar", f'{base} "webinar" OR "virtual event" "speaker"'),
            ("youtube_talk", f'{base} site:youtube.com "talk" OR "presentation" OR "keynote"'),
            ("panel", f'{base} "panel discussion" OR "panelist" OR "panel moderator"'),
            ("interview_video", f'{base} "interview" "video" OR site:youtube.com "interview"'),
        ]

        results: dict[str, list[dict[str, Any]]] = {}
        for category, query in queries:
            hits = await self.search(query, num=5)
            results[category] = [_normalize_result(r) for r in hits]
            await asyncio.sleep(0.3)

        return results

    async def search_visibility_sweep_with_ledger(
        self,
        name: str,
        company: str = "",
        graph: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run the full 16+ query visibility sweep with retrieval ledger logging.

        This is the fail-closed version: every query is logged to the Evidence Graph's
        retrieval ledger, even if it returns 0 results. If no graph is provided,
        results are still returned but not logged.
        """
        from app.brief.evidence_graph import build_visibility_queries

        queries = build_visibility_queries(name, company)
        results: dict[str, list[dict[str, Any]]] = {}

        for i, (query, intent) in enumerate(queries):
            category = f"visibility_{i}"
            if self.api_key:
                hits = await self.search(query, num=5)
                normalized = [_normalize_result(r) for r in hits]
            else:
                hits = []
                normalized = []

            results[category] = normalized

            # Log to retrieval ledger (always, even with 0 results)
            if graph is not None:
                graph.log_retrieval(
                    query=query,
                    intent=intent,
                    results=normalized,
                )

            if self.api_key:
                await asyncio.sleep(0.3)

        # Also populate the standard VISIBILITY_CATEGORIES dict
        category_results: dict[str, list[dict[str, Any]]] = {}
        for cat in VISIBILITY_CATEGORIES:
            category_results[cat] = []

        # Map queries back to categories based on keywords
        category_keywords = {
            "ted": ["ted.com", '"TED"', "TED talk"],
            "tedx": ["TEDx", "tedx.com"],
            "keynote": ["keynote"],
            "conference": ["conference"],
            "summit": ["summit"],
            "podcast": ["podcast"],
            "webinar": ["webinar"],
            "youtube_talk": ["YouTube talk", "youtube.com"],
            "panel": ["panel"],
            "interview_video": ["interview video", "fireside"],
        }
        for key, items in results.items():
            query_text = queries[int(key.split("_")[1])][0] if key.startswith("visibility_") else ""
            for cat, keywords in category_keywords.items():
                if any(kw.lower() in query_text.lower() for kw in keywords):
                    category_results[cat].extend(items)
                    break

        return category_results

    async def search_person_with_ledger(
        self,
        name: str,
        company: str = "",
        title: str = "",
        linkedin_url: str = "",
        graph: Any = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run person search queries with retrieval ledger logging."""
        results = await self.search_person(
            name=name, company=company, title=title, linkedin_url=linkedin_url,
        )

        if graph is not None:
            intent_map = {
                "general": "bio",
                "linkedin": "entity_lock",
                "news": "press",
                "talks": "talks",
                "company_site": "bio",
                "registry": "registry",
            }
            for category, items in results.items():
                intent = intent_map.get(category, "bio")
                q_parts = [f'"{name}"']
                if company:
                    q_parts.append(f'"{company}"')
                graph.log_retrieval(
                    query=f"{' '.join(q_parts)} [{category}]",
                    intent=intent,
                    results=items,
                )

        return results

    async def search_targeted(
        self,
        name: str,
        company: str = "",
        queries_override: list[tuple[str, str]] | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Run custom targeted queries. Use for follow-up disambiguation.

        queries_override: list of (category, query) tuples.
        """
        if not self.api_key:
            return {}

        queries = queries_override or []
        results: dict[str, list[dict[str, Any]]] = {}
        for category, query in queries:
            hits = await self.search(query, num=5)
            results[category] = [_normalize_result(r) for r in hits]
            await asyncio.sleep(0.3)

        return results


# ---------------------------------------------------------------------------
# Source formatting for LLM prompts
# ---------------------------------------------------------------------------

_TIER_LABELS = {
    SourceTier.PRIMARY: "PRIMARY",
    SourceTier.SECONDARY: "SECONDARY",
    SourceTier.LOW: "LOW-QUALITY",
}


def format_web_results_for_prompt(
    results: dict[str, list[dict[str, Any]]],
) -> str:
    """Format categorised search results into a text block for the LLM prompt.

    Includes source tier classification. Returns an empty string if no results.
    """
    sections: list[str] = []

    total = sum(len(v) for v in results.values())
    if total == 0:
        return ""

    sections.append(
        "The following web search results were retrieved in real-time. "
        "Use these as primary sources for your analysis. "
        "Cite the source URL when referencing a specific result. "
        "Source quality tier is indicated in brackets."
    )

    category_labels = {
        "general": "General Search Results",
        "linkedin": "LinkedIn Results",
        "news": "News & Articles",
        "talks": "Conference Talks & Podcasts",
        "company_site": "Company Website Results",
        "registry": "Corporate Registry Results",
    }

    for category, label in category_labels.items():
        items = results.get(category, [])
        if not items:
            continue
        sections.append(f"\n**{label}:**")
        for i, item in enumerate(items, 1):
            tier_label = _TIER_LABELS.get(item.get("tier", 3), "LOW-QUALITY")
            line = f"{i}. [{tier_label}] **{item['title']}**"
            if item.get("source"):
                line += f" ({item['source']})"
            if item.get("date"):
                line += f" [{item['date']}]"
            line += f"\n   URL: {item['link']}"
            if item.get("snippet"):
                line += f"\n   > {item['snippet']}"
            sections.append(line)

    return "\n".join(sections)


# Visibility category labels for prompt formatting
_VISIBILITY_LABELS: dict[str, str] = {
    "ted": "TED Talks",
    "tedx": "TEDx Talks",
    "keynote": "Keynote Speeches",
    "conference": "Conference Presentations",
    "summit": "Summit Appearances",
    "podcast": "Podcast Appearances",
    "webinar": "Webinars",
    "youtube_talk": "YouTube Talks/Presentations",
    "panel": "Panel Discussions",
    "interview_video": "Video Interviews",
}


def format_visibility_results_for_prompt(
    results: dict[str, list[dict[str, Any]]],
) -> str:
    """Format public visibility sweep results for the LLM prompt.

    Returns an empty string if no results.
    """
    sections: list[str] = []
    total = sum(len(v) for v in results.values())

    sections.append(
        "## PUBLIC VISIBILITY SWEEP RESULTS\n"
        "The following 10 targeted searches were executed. "
        "Use these to populate the Public Visibility Report section."
    )

    for category in VISIBILITY_CATEGORIES:
        label = _VISIBILITY_LABELS.get(category, category)
        items = results.get(category, [])
        status = f"{len(items)} results" if items else "NO RESULTS"
        sections.append(f"\n**{label}** ({status}):")
        if items:
            for i, item in enumerate(items, 1):
                tier_label = _TIER_LABELS.get(item.get("tier", 3), "LOW-QUALITY")
                line = f"  {i}. [{tier_label}] {item['title']}"
                if item.get("date"):
                    line += f" [{item['date']}]"
                line += f"\n     URL: {item['link']}"
                if item.get("snippet"):
                    line += f"\n     > {item['snippet']}"
                sections.append(line)
        else:
            sections.append("  (No results found)")

    sections.append(f"\n**Total visibility artifacts found:** {total}")
    return "\n".join(sections)


def generate_search_plan(
    name: str,
    company: str = "",
    title: str = "",
    linkedin_url: str = "",
    location: str = "",
) -> list[dict[str, str]]:
    """Generate a deterministic search plan for a person.

    Returns a list of dicts: {"query": str, "category": str, "rationale": str}
    This is useful for audit trails and for showing the user what would be searched.
    """
    plan = []

    # Core identity
    q_parts = [f'"{name}"']
    if company:
        q_parts.append(f'"{company}"')
    if title:
        q_parts.append(title)
    plan.append({
        "query": " ".join(q_parts),
        "category": "identity",
        "rationale": "Primary identity confirmation — name + company + title",
    })

    # LinkedIn
    linkedin_q = f'"{name}" site:linkedin.com'
    if company:
        linkedin_q += f' "{company}"'
    plan.append({
        "query": linkedin_q,
        "category": "linkedin",
        "rationale": "LinkedIn profile — role confirmation, career history, connections",
    })

    # Company website
    if company:
        company_slug = re.sub(r"[^a-zA-Z0-9]", "", company.lower())
        plan.append({
            "query": (
                f'"{name}" site:{company_slug}.com OR site:{company_slug}.io '
                f'OR site:{company_slug}.ai'
            ),
            "category": "company_site",
            "rationale": "Company website bio — official title, team page, about page",
        })

    # News and interviews
    news_parts = [f'"{name}"']
    if company:
        news_parts.append(f'"{company}"')
    plan.append({
        "query": " ".join(news_parts) + " interview OR article OR profile",
        "category": "news",
        "rationale": "News coverage — public statements, company announcements, press quotes",
    })

    # Podcasts and conferences
    talks_parts = [f'"{name}"']
    if company:
        talks_parts.append(company)
    plan.append({
        "query": " ".join(talks_parts) + " podcast OR webinar OR conference OR keynote",
        "category": "talks",
        "rationale": "Public speaking — rhetorical patterns, stated positions, audience context",
    })

    # Corporate registry (UK)
    plan.append({
        "query": f'"{name}" site:find-and-update.company-information.service.gov.uk',
        "category": "registry",
        "rationale": "UK Companies House — directorships, company filings, registered address",
    })

    # Corporate registry (US)
    plan.append({
        "query": f'"{name}" site:sec.gov OR site:opencorporates.com',
        "category": "registry_us",
        "rationale": "US corporate filings — SEC filings, officer listings",
    })

    # Thought leadership / blog
    plan.append({
        "query": f'"{name}" blog OR "written by" OR "authored by"',
        "category": "authored",
        "rationale": "Authored content — reveals thinking style, priorities, expertise claims",
    })

    # Public visibility sweep (10 queries)
    base = f'"{name}"'
    if company:
        base += f' "{company}"'
    visibility_queries = [
        ("ted", f'{base} site:ted.com OR "TED talk"',
         "TED talk — high-visibility public thought leadership"),
        ("tedx", f'{base} "TEDx" OR site:tedx.com',
         "TEDx talk — regional/topical thought leadership"),
        ("keynote", f'{base} "keynote" "speaker" OR "keynote address"',
         "Keynote speaking — conference headliner status"),
        ("conference", f'{base} "conference" "speaker" OR "spoke at"',
         "Conference presentations — industry engagement"),
        ("summit", f'{base} "summit" "speaker" OR "summit keynote"',
         "Summit appearances — executive-level visibility"),
        ("podcast", f'{base} "podcast" "guest" OR "episode"',
         "Podcast appearances — public positions, messaging patterns"),
        ("webinar", f'{base} "webinar" OR "virtual event" "speaker"',
         "Webinar appearances — topical expertise signals"),
        ("youtube_talk", f'{base} site:youtube.com "talk" OR "presentation"',
         "YouTube talks — public presentations, rhetorical style"),
        ("panel", f'{base} "panel discussion" OR "panelist"',
         "Panel discussions — collaborative positioning, peer network"),
        ("interview_video", f'{base} "interview" "video" OR site:youtube.com "interview"',
         "Video interviews — unscripted positions, body language context"),
    ]
    for cat, query, rationale in visibility_queries:
        plan.append({"query": query, "category": f"visibility_{cat}", "rationale": rationale})

    return plan
