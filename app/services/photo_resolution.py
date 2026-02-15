"""PhotoResolutionService — decision-tree photo resolver with caching.

Root cause of current failure:
Apollo.io returns LinkedIn CDN URLs for photo_url. These URLs:
1. Expire after hours/days (signed URLs with TTL)
2. Get auth-blocked by LinkedIn (403/401)
3. Are hotlinked by the frontend — blocked by CORS / referrer policy

We do NOT scrape LinkedIn profile HTML for images.
We do NOT hotlink linkedin.com CDN URLs in the frontend.

Decision tree:
1) uploaded → stored photo (user-uploaded, never expires)
2) cached_proxy → internal cached URL (previously resolved + cached)
3) enrichment_provider → Apollo/Clearbit person API → cache locally
4) gravatar → MD5(email) → cache locally
5) company_logo → Clearbit logo API → cache locally (fallback avatar)
6) initials → CSS-generated initials (final fallback)
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class PhotoSource(str, Enum):
    """How the photo was obtained."""
    UPLOADED = "uploaded"
    CACHED_PROXY = "cached_proxy"
    ENRICHMENT_PROVIDER = "enrichment_provider"
    LINKEDIN_PDF_CROP = "linkedin_pdf_crop"
    GRAVATAR = "gravatar"
    COMPANY_LOGO = "company_logo"
    INITIALS = "initials"


class PhotoStatus(str, Enum):
    """Current state of photo resolution."""
    RESOLVED = "RESOLVED"
    MISSING = "MISSING"
    FAILED = "FAILED"
    FAILED_RENDER = "FAILED_RENDER"
    BLOCKED = "BLOCKED"
    EXPIRED = "EXPIRED"
    UNKNOWN = "UNKNOWN"


@dataclass
class PhotoResolutionResult:
    """Result of the photo resolution decision tree."""
    photo_url: str = ""
    photo_source: str = PhotoSource.INITIALS
    photo_status: str = PhotoStatus.MISSING
    provider_response_code: int | None = None
    error: str = ""
    resolved_at: str = ""
    cache_key: str = ""


@dataclass
class PhotoResolutionLog:
    """Instrumentation log for a single resolution attempt."""
    contact_id: int = 0
    contact_name: str = ""
    attempted_sources: list[str] = field(default_factory=list)
    resolved_source: str = ""
    provider_response_code: int | None = None
    error: str = ""
    duration_ms: int = 0
    timestamp: str = ""


def gravatar_url(email: str, size: int = 200) -> str:
    """Generate Gravatar URL from email address."""
    if not email:
        return ""
    email_hash = hashlib.md5(
        email.strip().lower().encode("utf-8")
    ).hexdigest()
    return f"https://www.gravatar.com/avatar/{email_hash}?s={size}&d=404"


def clearbit_logo_url(domain: str, size: int = 128) -> str:
    """Generate Clearbit company logo URL from domain."""
    if not domain:
        return ""
    return f"https://logo.clearbit.com/{domain}?size={size}"


def extract_domain_from_email(email: str) -> str:
    """Extract domain from email address."""
    if not email or "@" not in email:
        return ""
    domain = email.split("@")[1].lower()
    # Skip common free email providers
    free_providers = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "live.com", "icloud.com", "me.com", "aol.com",
        "protonmail.com", "mail.com",
    }
    if domain in free_providers:
        return ""
    return domain


class PhotoResolutionService:
    """Resolves contact photos through a priority fallback chain.

    Decision tree (stops at first success):
    1. uploaded → return stored photo
    2. cached_proxy → return cached internal URL
    3. enrichment_provider → Apollo photo_url (if not a LinkedIn CDN URL)
    4. gravatar → MD5(email) lookup
    5. company_logo → Clearbit logo API
    6. initials → CSS-generated (final fallback, always works)
    """

    # LinkedIn CDN patterns we refuse to hotlink
    BLOCKED_URL_PATTERNS = [
        "media.licdn.com",
        "media-exp1.licdn.com",
        "static.licdn.com",
        "platform-lookaside.fbsbx.com",
    ]

    def __init__(self) -> None:
        self._resolution_logs: list[PhotoResolutionLog] = []

    @property
    def resolution_logs(self) -> list[PhotoResolutionLog]:
        return list(self._resolution_logs)

    def resolve(
        self,
        contact_id: int = 0,
        contact_name: str = "",
        email: str = "",
        linkedin_url: str = "",
        company_domain: str = "",
        existing_photo_url: str = "",
        existing_photo_source: str = "",
        existing_photo_status: str = "",
    ) -> PhotoResolutionResult:
        """Run the photo resolution decision tree.

        RULE: Never regress. If an existing photo_url is present and has not
        been marked FAILED_RENDER by the client, preserve it. Only attempt
        re-resolution when photo_url is empty or explicitly failed.

        Returns PhotoResolutionResult with the best available photo.
        """
        log = PhotoResolutionLog(
            contact_id=contact_id,
            contact_name=contact_name,
            timestamp=datetime.utcnow().isoformat(),
        )

        result = PhotoResolutionResult(
            resolved_at=datetime.utcnow().isoformat(),
        )

        # Step 0: PRESERVE existing photo if present and not failed
        # This is the anti-regression guard — never wipe a working photo.
        if existing_photo_url and existing_photo_status != PhotoStatus.FAILED_RENDER:
            log.attempted_sources.append("existing_preserved")
            log.resolved_source = "existing_preserved"
            # Determine the source category
            source = existing_photo_source or PhotoSource.ENRICHMENT_PROVIDER
            if existing_photo_source == PhotoSource.UPLOADED:
                source = PhotoSource.UPLOADED
            elif existing_photo_source == PhotoSource.CACHED_PROXY:
                source = PhotoSource.CACHED_PROXY
            elif existing_photo_source == PhotoSource.LINKEDIN_PDF_CROP:
                source = PhotoSource.LINKEDIN_PDF_CROP
            elif existing_photo_source == PhotoSource.GRAVATAR:
                source = PhotoSource.GRAVATAR
            elif existing_photo_source == PhotoSource.COMPANY_LOGO:
                source = PhotoSource.COMPANY_LOGO
            else:
                source = PhotoSource.ENRICHMENT_PROVIDER

            result.photo_url = existing_photo_url
            result.photo_source = source
            result.photo_status = PhotoStatus.RESOLVED
            if self._is_blocked_url(existing_photo_url):
                log.attempted_sources.append("linkedin_cdn_preserved")
                logger.info(
                    "Photo for %s is LinkedIn CDN — preserving (client will fallback on error)",
                    contact_name,
                )
            self._resolution_logs.append(log)
            return result

        # --- Below here: photo_url is empty OR photo_status == FAILED_RENDER ---

        # Step 1: Gravatar (email-based)
        if email:
            log.attempted_sources.append("gravatar")
            grav_url = gravatar_url(email)
            if grav_url:
                result.photo_url = grav_url
                result.photo_source = PhotoSource.GRAVATAR
                result.photo_status = PhotoStatus.RESOLVED
                result.cache_key = f"gravatar:{email.lower()}"
                log.resolved_source = "gravatar"
                self._resolution_logs.append(log)
                return result

        # Step 2: Company logo (domain-based)
        domain = company_domain or extract_domain_from_email(email)
        if domain:
            log.attempted_sources.append("company_logo")
            logo_url = clearbit_logo_url(domain)
            if logo_url:
                result.photo_url = logo_url
                result.photo_source = PhotoSource.COMPANY_LOGO
                result.photo_status = PhotoStatus.RESOLVED
                result.cache_key = f"logo:{domain}"
                log.resolved_source = "company_logo"
                self._resolution_logs.append(log)
                return result

        # Step 3: Initials fallback (always works)
        log.attempted_sources.append("initials")
        log.resolved_source = "initials"
        result.photo_url = ""
        result.photo_source = PhotoSource.INITIALS
        result.photo_status = PhotoStatus.MISSING
        self._resolution_logs.append(log)
        return result

    def _is_blocked_url(self, url: str) -> bool:
        """Check if URL is a LinkedIn CDN URL we refuse to hotlink."""
        if not url:
            return False
        url_lower = url.lower()
        return any(pattern in url_lower for pattern in self.BLOCKED_URL_PATTERNS)

    def get_debug_stats(self, profiles: list[dict]) -> dict:
        """Generate debug stats for /debug/photos endpoint."""
        total = len(profiles)
        resolved = 0
        missing = 0
        failed = 0
        blocked = 0
        source_breakdown: dict[str, int] = {}
        error_breakdown: dict[str, int] = {}

        for p in profiles:
            photo_status = p.get("photo_status", "MISSING")
            photo_source = p.get("photo_source", "initials")

            if photo_status == PhotoStatus.RESOLVED:
                resolved += 1
            elif photo_status == PhotoStatus.FAILED:
                failed += 1
            elif photo_status == PhotoStatus.BLOCKED:
                blocked += 1
            else:
                missing += 1

            source_breakdown[photo_source] = (
                source_breakdown.get(photo_source, 0) + 1
            )

            photo_error = p.get("photo_last_error", "")
            if photo_error:
                error_breakdown[photo_error[:80]] = (
                    error_breakdown.get(photo_error[:80], 0) + 1
                )

        return {
            "total_contacts": total,
            "resolved_photos": resolved,
            "missing_photos": missing,
            "failed_resolution": failed,
            "blocked_urls": blocked,
            "source_breakdown": source_breakdown,
            "last_error_breakdown": error_breakdown,
        }


def resolve_photo_for_profile(profile_data: dict) -> dict:
    """Convenience: run photo resolution for a single profile dict.

    Updates the profile_data dict in place and returns the resolution result.
    RULE: Never clear an existing photo_url unless photo_status == FAILED_RENDER.
    """
    service = PhotoResolutionService()
    result = service.resolve(
        contact_name=profile_data.get("name", ""),
        email=profile_data.get("email", ""),
        linkedin_url=profile_data.get("linkedin_url", ""),
        company_domain=profile_data.get("company_domain", ""),
        existing_photo_url=profile_data.get("photo_url", ""),
        existing_photo_source=profile_data.get("photo_source", ""),
        existing_photo_status=profile_data.get("photo_status", ""),
    )

    profile_data["photo_url"] = result.photo_url
    profile_data["photo_source"] = result.photo_source
    profile_data["photo_status"] = result.photo_status
    profile_data["photo_last_checked_at"] = result.resolved_at
    if result.error:
        profile_data["photo_last_error"] = result.error

    return profile_data


def backfill_photo_status(profile_data: dict) -> dict:
    """Backfill: if photo_url exists but status was set to MISSING, restore to UNKNOWN.

    This fixes profiles that had their status overwritten by a prior buggy resolver
    that blocked LinkedIn CDN URLs and set status to MISSING.
    """
    photo_url = profile_data.get("photo_url", "")
    photo_status = profile_data.get("photo_status", "")
    if photo_url and photo_status in (PhotoStatus.MISSING, "MISSING", ""):
        profile_data["photo_status"] = PhotoStatus.UNKNOWN
    return profile_data
