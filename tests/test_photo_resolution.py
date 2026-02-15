"""Tests for the PhotoResolutionService decision tree."""

from __future__ import annotations

from app.services.photo_resolution import (
    PhotoResolutionService,
    PhotoSource,
    PhotoStatus,
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

    def test_cached_proxy_second_priority(self):
        """Step 2: previously cached proxy."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="https://internal.cdn/cached.jpg",
            existing_photo_source=PhotoSource.CACHED_PROXY,
        )
        assert result.photo_source == PhotoSource.CACHED_PROXY
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_non_linkedin_enrichment_url_accepted(self):
        """Step 3: enrichment photo that is NOT LinkedIn CDN → accepted."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            existing_photo_url="https://api.apollo.io/photos/test.jpg",
            existing_photo_source="",
        )
        assert result.photo_source == PhotoSource.ENRICHMENT_PROVIDER
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_linkedin_cdn_url_blocked(self):
        """LinkedIn CDN URLs are BLOCKED — falls through to gravatar."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="https://media.licdn.com/dms/image/v2/abc123",
            existing_photo_source="",
        )
        # Should NOT use the LinkedIn CDN URL
        assert "licdn.com" not in result.photo_url
        # Should fall through to gravatar or company logo
        assert result.photo_source in (
            PhotoSource.GRAVATAR, PhotoSource.COMPANY_LOGO
        )
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_licdn_exp1_also_blocked(self):
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@example.com",
            existing_photo_url="https://media-exp1.licdn.com/photo.jpg",
        )
        assert "licdn.com" not in result.photo_url

    def test_gravatar_fallback_with_email(self):
        """Step 4: gravatar when no enrichment photo."""
        service = PhotoResolutionService()
        result = service.resolve(
            contact_name="Test",
            email="test@example.com",
        )
        assert result.photo_source == PhotoSource.GRAVATAR
        assert "gravatar.com" in result.photo_url
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_company_logo_fallback(self):
        """Step 5: company logo when no email for gravatar."""
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
            existing_photo_url="https://media.licdn.com/blocked.jpg",
        )
        # Gravatar comes first, then company logo would be fallback
        assert result.photo_status == PhotoStatus.RESOLVED

    def test_initials_when_nothing_available(self):
        """Step 6: initials fallback when nothing else works."""
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
        assert log.resolved_source in ("gravatar", "company_logo", "initials")
        assert len(log.attempted_sources) > 0

    def test_blocked_url_creates_error_log(self):
        """Blocked LinkedIn URL should record error in log."""
        service = PhotoResolutionService()
        service.resolve(
            contact_name="Test",
            email="test@acme.com",
            existing_photo_url="https://media.licdn.com/photo.jpg",
        )
        log = service.resolution_logs[0]
        assert "enrichment_provider_blocked" in log.attempted_sources

    def test_cache_key_set_for_gravatar(self):
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test", email="test@example.com")
        assert result.cache_key.startswith("gravatar:")

    def test_cache_key_set_for_company_logo(self):
        service = PhotoResolutionService()
        result = service.resolve(contact_name="Test", company_domain="acme.com")
        assert result.cache_key.startswith("logo:")


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
