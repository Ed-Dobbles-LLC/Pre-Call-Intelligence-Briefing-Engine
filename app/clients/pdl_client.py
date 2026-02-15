"""People Data Labs (PDL) API client.

Enriches contacts via https://api.peopledatalabs.com/v5/person/enrich.
Auth: X-Api-Key header.
Rate limited to PDL_MAX_REQUESTS_PER_MIN.
Retries 2x on 429/5xx with exponential backoff.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

PDL_ENRICH_URL = "https://api.peopledatalabs.com/v5/person/enrich"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PDLPersonFields:
    """Structured fields extracted from PDL response."""
    name: str = ""
    title: str = ""
    company: str = ""
    location: str = ""
    linkedin_url: str = ""
    photo_url: str = ""


@dataclass
class PDLEnrichResult:
    """Structured result from PDL enrichment."""
    status: str = ""  # "success", "no_match", "error"
    person_id: str = ""
    match_confidence: float = 0.0
    fields: PDLPersonFields = field(default_factory=PDLPersonFields)
    raw_response: dict = field(default_factory=dict)
    error: str = ""
    http_status: int = 0


# ---------------------------------------------------------------------------
# In-memory rate limiter
# ---------------------------------------------------------------------------


class RateLimiter:
    """Sliding-window rate limiter enforcing max_requests per window."""

    def __init__(self, max_requests: int, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def acquire(self) -> bool:
        """Try to acquire a request slot. Returns True if allowed."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self.max_requests:
            return False
        self._timestamps.append(now)
        return True

    def wait_time(self) -> float:
        """Seconds until a slot opens. Returns 0 if available now."""
        now = time.monotonic()
        cutoff = now - self.window_seconds
        active = [t for t in self._timestamps if t > cutoff]
        if len(active) < self.max_requests:
            return 0.0
        oldest = min(active)
        return max(0.0, oldest + self.window_seconds - now)

    @property
    def current_count(self) -> int:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        return len([t for t in self._timestamps if t > cutoff])

    @property
    def state(self) -> dict:
        return {
            "max_requests": self.max_requests,
            "window_seconds": self.window_seconds,
            "current_count": self.current_count,
            "remaining": max(0, self.max_requests - self.current_count),
        }


# ---------------------------------------------------------------------------
# PDL Client
# ---------------------------------------------------------------------------

# Module-level rate limiter (shared across requests)
_rate_limiter: RateLimiter | None = None


def get_rate_limiter() -> RateLimiter:
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            max_requests=settings.pdl_max_requests_per_min,
            window_seconds=60.0,
        )
    return _rate_limiter


# Track enrichment attempts for debug visibility
_enrichment_log: list[dict] = []
MAX_LOG_SIZE = 100


def get_enrichment_log() -> list[dict]:
    return _enrichment_log


def _log_attempt(entry: dict) -> None:
    _enrichment_log.append(entry)
    if len(_enrichment_log) > MAX_LOG_SIZE:
        _enrichment_log.pop(0)


class PDLClient:
    """People Data Labs API client with rate limiting and retries."""

    def __init__(self):
        self.api_key = settings.pdl_api_key
        self.enabled = settings.pdl_enabled
        self.timeout_ms = settings.pdl_timeout_ms
        self.max_retries = 2
        self.rate_limiter = get_rate_limiter()

    async def enrich_person(
        self,
        email: str | None = None,
        linkedin_url: str | None = None,
        name: str | None = None,
        company: str | None = None,
        location: str | None = None,
    ) -> PDLEnrichResult:
        """Enrich a person using PDL's person/enrich endpoint.

        Identifier priority: email > linkedin_url > name+company > name+location.
        Returns structured result. Never raises — errors returned in result.
        """
        if not self.enabled or not self.api_key:
            return PDLEnrichResult(
                status="error",
                error="PDL not enabled or API key not configured",
            )

        # Build query params
        params: dict[str, str] = {}
        if email:
            params["email"] = email
        if linkedin_url:
            params["profile"] = linkedin_url
        if name:
            # Split name into first/last for PDL
            parts = name.split(None, 1)
            if len(parts) >= 1:
                params["first_name"] = parts[0]
            if len(parts) >= 2:
                params["last_name"] = parts[1]
        if company:
            params["company"] = company
        if location:
            params["location"] = location

        if not params:
            return PDLEnrichResult(
                status="error",
                error="No identifiers provided",
            )

        # Rate limit check
        if not self.rate_limiter.acquire():
            wait = self.rate_limiter.wait_time()
            logger.warning(
                "PDL rate limit reached (%d/%d). Wait %.1fs",
                self.rate_limiter.current_count,
                self.rate_limiter.max_requests,
                wait,
            )
            # Wait and retry once
            if wait > 0:
                await asyncio.sleep(min(wait, 10.0))
                if not self.rate_limiter.acquire():
                    return PDLEnrichResult(
                        status="error",
                        error="Rate limit exceeded",
                    )

        # Execute request with retries
        log_params = {k: v for k, v in params.items()}
        # Never log the API key
        logger.info("PDL enrich request: params=%s", list(log_params.keys()))

        result = await self._execute_with_retries(params)

        # Log attempt
        _log_attempt({
            "timestamp": time.time(),
            "params_keys": list(log_params.keys()),
            "status": result.status,
            "http_status": result.http_status,
            "match_confidence": result.match_confidence,
            "person_id": result.person_id,
            "error": result.error,
            "fields_returned": [
                k for k, v in (
                    ("name", result.fields.name),
                    ("title", result.fields.title),
                    ("company", result.fields.company),
                    ("location", result.fields.location),
                    ("linkedin_url", result.fields.linkedin_url),
                    ("photo_url", result.fields.photo_url),
                )
                if v
            ],
        })

        return result

    async def _execute_with_retries(self, params: dict) -> PDLEnrichResult:
        """Execute PDL API call with retry logic for 429/5xx."""
        timeout_s = self.timeout_ms / 1000.0
        headers = {
            "X-Api-Key": self.api_key,
            "Accept": "application/json",
        }

        last_error = ""
        last_status = 0

        for attempt in range(1 + self.max_retries):
            try:
                logger.info(
                    "PDL request attempt %d/%d",
                    attempt + 1,
                    1 + self.max_retries,
                )
                async with httpx.AsyncClient(timeout=timeout_s) as client:
                    resp = await client.get(
                        PDL_ENRICH_URL,
                        params=params,
                        headers=headers,
                    )

                last_status = resp.status_code
                logger.info("PDL response: status=%d", resp.status_code)

                if resp.status_code == 200:
                    return self._parse_success(resp.json(), resp.status_code)

                if resp.status_code == 404:
                    return PDLEnrichResult(
                        status="no_match",
                        http_status=404,
                        error="No matching person found",
                    )

                # Retry on 429 (rate limit) or 5xx (server error)
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_error = f"HTTP {resp.status_code}"
                    if attempt < self.max_retries:
                        wait = 2 ** (attempt + 1)  # 2s, 4s
                        logger.warning(
                            "PDL returned %d, retrying in %ds (attempt %d/%d)",
                            resp.status_code, wait, attempt + 1, 1 + self.max_retries,
                        )
                        await asyncio.sleep(wait)
                        continue
                else:
                    # Non-retryable error
                    return PDLEnrichResult(
                        status="error",
                        http_status=resp.status_code,
                        error=f"PDL API error: HTTP {resp.status_code}",
                    )

            except httpx.TimeoutException:
                last_error = "Timeout"
                last_status = 0
                if attempt < self.max_retries:
                    wait = 2 ** (attempt + 1)
                    logger.warning(
                        "PDL timeout, retrying in %ds (attempt %d/%d)",
                        wait, attempt + 1, 1 + self.max_retries,
                    )
                    await asyncio.sleep(wait)
                    continue

            except Exception as exc:
                last_error = str(exc)
                last_status = 0
                logger.exception("PDL request failed")
                break

        return PDLEnrichResult(
            status="error",
            http_status=last_status,
            error=f"All retries exhausted: {last_error}",
        )

    def _parse_success(self, data: dict, status_code: int) -> PDLEnrichResult:
        """Parse a successful PDL response into structured fields."""
        fields = PDLPersonFields(
            name=data.get("full_name", ""),
            title=data.get("job_title", ""),
            company=data.get("job_company_name", ""),
            location=data.get("location_name", ""),
            linkedin_url=data.get("linkedin_url", ""),
            photo_url=data.get("profile_pic_url", "") or data.get("github_avatar_url", ""),
        )

        return PDLEnrichResult(
            status="success",
            person_id=data.get("id", ""),
            match_confidence=float(data.get("likelihood", 0) or 0),
            fields=fields,
            raw_response=data,
            http_status=status_code,
        )
