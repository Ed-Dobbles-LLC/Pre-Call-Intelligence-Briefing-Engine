"""Tests for the PhotoResolutionService decision tree.

Key regression tests:
- NEVER wipe an existing photo_url unless photo_status == FAILED_RENDER
- LinkedIn CDN URLs are preserved (rendered as-is; client handles fallback)
- Photo refresh does not clear photo_url
"""

from __future__ import annotations

from app.services.photo_resolution import (
    PhotoResolutionService,
    PhotoSource,
    PhotoStatus,
    backfill_photo_status,
    clearbit_logo_url,
    extract_domain_from_email,
    gravatar_url,
    resolve_photo_for_profile,
)


class TestGravatarUrl:
    def test_basic_email(self):
        url = gravatar_url("test@example.com")
        assert "gravatar.com/avatar/" in url
        assert "d=404" in url

    def test_empty_email(self):
        assert gravatar_url("") == ""

    def test_strips_and_lowercases(self):
        url1 = gravatar_url("Test@Example.com")
        url2 = gravatar_url("  test@example.com  ")
        assert url1 == url2

    def test_custom_size(self):
        url = gravatar_url("test@example.com", size=128)
        assert "s=128" in url


class TestClearbitLogoUrl:
    def test_basic_domain(self):
        url = clearbit_logo_url("acme.com")
        assert "logo.clearbit.com/acme.com" in url

    def test_empty_domain(self):
        assert clearbit_logo_url("") == ""

    def test_custom_size(self):
        url = clearbit_logo_url("acme.com", size=256)
        assert "size=256" in url


class TestExtractDomain:
    def test_company_email(self):
        assert extract_domain_from_email("john@acme.com") == "acme.com"

    def test_free_email_returns_empty(self):
        assert extract_domain_from_email("john@gmail.com") == ""

    def test_outlook_returns_empty(self):
        assert extract_domain_from_email("john@outlook.com") == ""

    def test_no_at_sign(self):
        assert extract_domain_from_email("notanemail") == ""

    def test_empty(self):
        assert extract_domain_from_email("") == ""


class TestPhotoResolutionDecisionTree:
    """Tests for the full decision tree."""

    def test_uploaded_photo_highest_priority(self):
        """Step 1: user-uploaded photo always wins."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="https://internal.cdn/photo.jpg",
            existing_photo_source=PhotoSource.UPLOADED,
        )
        assert result.photo_source == PhotoSource.UPLOADED
        assert result.photo_status == PhotoStatus.RESOLVED
        assert result.photo_url == "https://internal.cdn/photo.jpg"

    def test_cached_proxy_preserved(self):
        """Previously cached proxy is preserved."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="https://internal.cdn/cached.jpg",
            existing_photo_source=PhotoSource.CACHED_PROXY,
        )
        assert result.photo_source == PhotoSource.CACHED_PROXY
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_non_linkedin_enrichment_url_preserved(self):
        """Existing enrichment photo URL is preserved."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="https://api.apollo.io/photos/test.jpg",
            existing_photo_source="",
        )
        assert result.photo_source == PhotoSource.ENRICHMENT_PROVIDER
        assert result.photo_status == PhotoStatus.RESOLVED
        assert result.photo_url == "https://api.apollo.io/photos/test.jpg"

    def test_linkedin_cdn_url_preserved_not_blocked(self):
        """REGRESSION FIX: LinkedIn CDN URLs are PRESERVED, not blocked."""
        service = PhotoResolutionService()
        licdn_url = "https://media.licdn.com/dms/image/v2/abc123"
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url=licdn_url,
            existing_photo_source="",
        )
        # MUST preserve the LinkedIn CDN URL (client handles fallback)
        assert result.photo_url == licdn_url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_licdn_exp1_also_preserved(self):
        """REGRESSION FIX: media-exp1.licdn.com URLs also preserved."""
        service = PhotoResolutionService()
        url = "https://media-exp1.licdn.com/photo.jpg"
        result = service.resolve(
            contact_name="Test",
            email="test@example.com",
            existing_photo_url=url,
        )
        assert result.photo_url == url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_failed_render_triggers_re_resolution(self):
        """When photo_status == FAILED_RENDER, resolver tries gravatar/logo."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="https://media.licdn.com/expired.jpg",
            existing_photo_status=PhotoStatus.FAILED_RENDER,
        )
        # Should NOT use the failed URL — should resolve to gravatar
        assert "licdn.com" not in result.photo_url
        assert result.photo_source in (PhotoSource.GRAVATAR, PhotoSource.COMPANY_LOGO)
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_gravatar_when_no_photo(self):
        """Gravatar when no existing photo URL."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@example.com",
        )
        assert result.photo_source == PhotoSource.GRAVATAR
        assert "gravatar.com" in result.photo_url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_company_logo_fallback(self):
        """Company logo when no email for gravatar."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            company_domain="acme.com",
        )
        assert result.photo_source == PhotoSource.COMPANY_LOGO
        assert "clearbit.com" in result.photo_url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_company_logo_from_email_domain(self):
        """Company domain extracted from company email."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="john@bigcorp.io",
        )
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_initials_when_nothing_available(self):
        """Initials fallback when nothing else works."""
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test")
        assert result.photo_source == PhotoSource.INITIALS
        assert result.photo_status == PhotoStatus.MISSING
        assert result.photo_url == ""

    def test_no_silent_null_photo_url(self):
        """photo_url must never be None — always string."""
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test")
        assert result.photo_url is not None
        assert isinstance(result.photo_url, str)

    def test_resolution_log_recorded(self):
        """Each resolution attempt creates a log entry."""
        service = PhotoResolutionService()
        service.resolve(contact_name="Test User", email="test@acme.com")
        assert len(service.resolution_logs) == 1
        log = service.resolution_logs[0]
        assert log.contact_name == "Test User"
        assert len(log.attempted_sources) > 0

    def test_linkedin_cdn_preserved_creates_log(self):
        """LinkedIn CDN URL preservation should be logged."""
        service = PhotoResolutionService()
        service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="https://media.licdn.com/photo.jpg",
        )
        log = service.resolution_logs[0]
        assert "existing_preserved" in log.attempted_sources
        assert "linkedin_cdn_preserved" in log.attempted_sources

    def test_cache_key_set_for_gravatar(self):
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test", email="test@example.com")
        assert result.cache_key.startswith("gravatar:")

    def test_cache_key_set_for_company_logo(self):
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test", company_domain="acme.com")
        assert result.cache_key.startswith("logo:")


# ---------------------------------------------------------------------------
# REGRESSION PREVENTION TESTS — These would have caught the photo regression
# ---------------------------------------------------------------------------


class TestPhotoRegressionPrevention:
    """Tests that would have caught the licdn.com photo disappearance."""

    def test_existing_licdn_photo_with_resolved_status_kept(self):
        """REGRESSION: contact with photo_url=licdn.com + status=RESOLVED must keep it."""
        service = PhotoResolutionService()
        licdn_url = "https://media.licdn.com/dms/image/v2/abc123"
        result = service.resolve(
            contact_name="Una Fox",
            email="una@example.com",
            existing_photo_url=licdn_url,
            existing_photo_source=PhotoSource.ENRICHMENT_PROVIDER,
            existing_photo_status=PhotoStatus.RESOLVED,
        )
        assert result.photo_url == licdn_url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_photo_refresh_does_not_clear_existing_url(self):
        """REGRESSION: resolve_photo_for_profile must not wipe photo_url."""
        profile = {
            "name": "Ben Titmus",
            "email": "ben@acme.com",
            "photo_url": "https://media.licdn.com/dms/image/v2/ben123",
            "photo_source": "enrichment_provider",
            "photo_status": "RESOLVED",
        }
        resolve_photo_for_profile(profile)
        # photo_url MUST NOT be cleared or replaced
        assert profile["photo_url"] == "https://media.licdn.com/dms/image/v2/ben123"
        assert profile["photo_status"] == PhotoStatus.RESOLVED

    def test_photo_refresh_does_not_downgrade_to_gravatar(self):
        """REGRESSION: resolver must not replace licdn URL with gravatar."""
        service = PhotoResolutionService()
        licdn_url = "https://media.licdn.com/dms/image/v2/photo.jpg"
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url=licdn_url,
            existing_photo_source="",
            existing_photo_status="RESOLVED",
        )
        assert result.photo_url == licdn_url
        assert "gravatar.com" not in result.photo_url

    def test_failed_render_allows_re_resolution(self):
        """Only FAILED_RENDER status triggers finding a new photo."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="https://media.licdn.com/expired.jpg",
            existing_photo_source="enrichment_provider",
            existing_photo_status=PhotoStatus.FAILED_RENDER,
        )
        # Now it SHOULD try to find a better photo
        assert "licdn.com" not in result.photo_url
        assert result.photo_source in (PhotoSource.GRAVATAR, PhotoSource.COMPANY_LOGO)

    def test_empty_photo_url_triggers_resolution(self):
        """When photo_url is empty, resolver tries to find one."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="",
            existing_photo_status="",
        )
        assert result.photo_url != ""
        assert result.photo_source == PhotoSource.GRAVATAR


class TestBackfillPhotoStatus:
    """Tests for the status backfill function."""

    def test_backfill_missing_to_unknown(self):
        """If photo_url exists but status=MISSING, restore to UNKNOWN."""
        profile = {
            "photo_url": "https://media.licdn.com/photo.jpg",
            "photo_status": "MISSING",
        }
        backfill_photo_status(profile)
        assert profile["photo_status"] == PhotoStatus.UNKNOWN

    def test_backfill_empty_status_to_unknown(self):
        """If photo_url exists but status is empty, set UNKNOWN."""
        profile = {
            "photo_url": "https://example.com/photo.jpg",
            "photo_status": "",
        }
        backfill_photo_status(profile)
        assert profile["photo_status"] == PhotoStatus.UNKNOWN

    def test_no_backfill_when_already_resolved(self):
        """If status is RESOLVED, don't change it."""
        profile = {
            "photo_url": "https://example.com/photo.jpg",
            "photo_status": "RESOLVED",
        }
        backfill_photo_status(profile)
        assert profile["photo_status"] == "RESOLVED"

    def test_no_backfill_when_no_url(self):
        """If no photo_url, don't change status."""
        profile = {
            "photo_url": "",
            "photo_status": "MISSING",
        }
        backfill_photo_status(profile)
        assert profile["photo_status"] == "MISSING"


class TestResolvePhotoForProfile:
    def test_updates_profile_data_in_place(self):
        profile = {"name": "Test", "email": "test@acme.com"}
        resolve_photo_for_profile(profile)
        assert "photo_source" in profile
        assert "photo_status" in profile
        assert "photo_last_checked_at" in profile

    def test_sets_photo_url(self):
        profile = {"name": "Test", "email": "test@acme.com"}
        resolve_photo_for_profile(profile)
        assert profile["photo_url"] != ""

    def test_preserves_existing_photo(self):
        """resolve_photo_for_profile must not wipe an existing photo."""
        profile = {
            "name": "Test",
            "email": "test@acme.com",
            "photo_url": "https://media.licdn.com/photo.jpg",
            "photo_source": "enrichment_provider",
            "photo_status": "RESOLVED",
        }
        resolve_photo_for_profile(profile)
        assert profile["photo_url"] == "https://media.licdn.com/photo.jpg"


class TestPhotoDebugStats:
    def test_basic_stats(self):
        service = PhotoResolutionService()
        profiles = [
            {"photo_status": "RESOLVED", "photo_source": "gravatar"},
            {"photo_status": "MISSING", "photo_source": "initials"},
            {"photo_status": "BLOCKED", "photo_source": "enrichment_provider", "photo_last_error": "LinkedIn CDN blocked"},
        ]
        stats = service.get_debug_stats(profiles)
        assert stats["total_contacts"] == 3
        assert stats["resolved_photos"] == 1
        assert stats["missing_photos"] == 1
        assert stats["blocked_urls"] == 1
        assert "gravatar" in stats["source_breakdown"]
        assert "LinkedIn CDN blocked" in stats["last_error_breakdown"]
