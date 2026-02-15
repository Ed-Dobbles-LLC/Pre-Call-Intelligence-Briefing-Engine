"""Tests for People Data Labs (PDL) enrichment integration.

Covers:
1. Successful enrichment updates fields
2. No-match does not wipe fields
3. Photo download success replaces photo
4. Photo download failure does NOT replace existing photo
5. Rate limiter enforces max requests/min
6. Retry logic works for 429
7. API endpoint returns expected JSON
"""

from __future__ import annotations

import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

from app.clients.pdl_client import (
    PDLClient,
    PDLEnrichResult,
    PDLPersonFields,
    RateLimiter,
)
from app.services.enrichment_service import _download_and_store_photo, enrich_contact
from app.store.database import EntityRecord, get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_pdl_success(
    name="Una Fox",
    title="CEO",
    company="Fox Corp",
    location="San Francisco, CA",
    linkedin_url="https://linkedin.com/in/unafox",
    photo_url="https://example.com/photo.jpg",
    person_id="pdl-123",
    confidence=0.95,
) -> PDLEnrichResult:
    """Create a successful PDL enrichment result."""
    return PDLEnrichResult(
        status="success",
        person_id=person_id,
        match_confidence=confidence,
        fields=PDLPersonFields(
            name=name,
            title=title,
            company=company,
            location=location,
            linkedin_url=linkedin_url,
            photo_url=photo_url,
        ),
        raw_response={
            "id": person_id,
            "full_name": name,
            "job_title": title,
            "job_company_name": company,
            "location_name": location,
            "linkedin_url": linkedin_url,
            "profile_pic_url": photo_url,
            "likelihood": confidence,
        },
        http_status=200,
    )


def _make_pdl_no_match() -> PDLEnrichResult:
    return PDLEnrichResult(
        status="no_match",
        http_status=404,
        error="No matching person found",
    )


def _make_pdl_error(msg="API error") -> PDLEnrichResult:
    return PDLEnrichResult(
        status="error",
        error=msg,
    )


# ---------------------------------------------------------------------------
# 1. Successful enrichment updates fields
# ---------------------------------------------------------------------------


class TestSuccessfulEnrichment:
    @pytest.mark.asyncio
    async def test_enrich_updates_empty_fields(self):
        """When profile fields are empty, PDL data should fill them in."""
        profile_data = {"emails": ["una@fox.com"], "name": "Una Fox"}
        pdl_result = _make_pdl_success()

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = pdl_result
            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=1,
                contact_name="Una Fox",
            )

        assert result["success"] is True
        assert result["match_confidence"] == 0.95
        assert result["pdl_person_id"] == "pdl-123"
        assert "title" in result["fields_updated"]
        assert "company" in result["fields_updated"]
        assert "location" in result["fields_updated"]
        assert "linkedin_url" in result["fields_updated"]
        assert profile_data["title"] == "CEO"
        assert profile_data["company"] == "Fox Corp"
        assert profile_data["location"] == "San Francisco, CA"
        assert profile_data["linkedin_url"] == "https://linkedin.com/in/unafox"
        assert profile_data["pdl_person_id"] == "pdl-123"
        assert profile_data["pdl_match_confidence"] == 0.95
        assert "enriched_at" in profile_data

    @pytest.mark.asyncio
    async def test_enrich_does_not_overwrite_existing_fields(self):
        """Existing profile fields should NOT be overwritten by PDL data."""
        profile_data = {
            "emails": ["una@fox.com"],
            "name": "Una Fox",
            "title": "Existing Title",
            "company": "Existing Company",
            "location": "Existing Location",
            "linkedin_url": "https://linkedin.com/in/existing",
        }
        pdl_result = _make_pdl_success()

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = pdl_result
            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=1,
                contact_name="Una Fox",
            )

        assert result["success"] is True
        # No fields should have been updated (all were already set)
        assert "title" not in result["fields_updated"]
        assert "company" not in result["fields_updated"]
        assert "location" not in result["fields_updated"]
        assert "linkedin_url" not in result["fields_updated"]
        # Original values preserved
        assert profile_data["title"] == "Existing Title"
        assert profile_data["company"] == "Existing Company"

    @pytest.mark.asyncio
    async def test_enrich_stores_enrichment_metadata(self):
        """Should store PDL metadata (person_id, confidence, raw response, timestamp)."""
        profile_data = {"emails": ["una@fox.com"]}
        pdl_result = _make_pdl_success()

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = pdl_result
            await enrich_contact(profile_data=profile_data, contact_id=1, contact_name="Una Fox")

        assert profile_data["pdl_person_id"] == "pdl-123"
        assert profile_data["pdl_match_confidence"] == 0.95
        assert profile_data["enrichment_json"]["id"] == "pdl-123"
        assert profile_data["enriched_at"]  # ISO timestamp present

    @pytest.mark.asyncio
    async def test_enrich_uses_email_as_primary_identifier(self):
        """Email should be passed as the primary identifier."""
        profile_data = {"emails": ["una@fox.com"], "linkedin_url": "https://linkedin.com/in/una"}

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = _make_pdl_success()
            await enrich_contact(profile_data=profile_data, contact_id=1)

        call_kwargs = mock_enrich.call_args[1]
        assert call_kwargs["email"] == "una@fox.com"

    @pytest.mark.asyncio
    async def test_enrich_falls_back_to_linkedin_url(self):
        """When no email, should use linkedin_url."""
        profile_data = {"linkedin_url": "https://linkedin.com/in/una"}

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = _make_pdl_success()
            await enrich_contact(
                profile_data=profile_data,
                contact_id=1,
                contact_name="Una Fox",
            )

        call_kwargs = mock_enrich.call_args[1]
        assert call_kwargs["linkedin_url"] == "https://linkedin.com/in/una"

    @pytest.mark.asyncio
    async def test_enrich_requires_some_identifier(self):
        """With no email, no linkedin_url, no name — should return error."""
        profile_data = {}

        result = await enrich_contact(profile_data=profile_data, contact_id=1)

        assert result["success"] is False
        assert "No identifiers" in result["error"]


# ---------------------------------------------------------------------------
# 2. No-match does not wipe fields
# ---------------------------------------------------------------------------


class TestNoMatch:
    @pytest.mark.asyncio
    async def test_no_match_preserves_all_fields(self):
        """PDL no_match should not modify any existing profile fields."""
        profile_data = {
            "emails": ["una@fox.com"],
            "title": "CEO",
            "company": "Fox Corp",
            "photo_url": "https://example.com/existing.jpg",
            "photo_status": "RESOLVED",
            "photo_source": "uploaded",
        }
        original = dict(profile_data)

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = _make_pdl_no_match()
            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=1,
                contact_name="Una Fox",
            )

        assert result["success"] is False
        assert "No matching person" in result["error"]
        # All original fields must be untouched
        assert profile_data["title"] == original["title"]
        assert profile_data["company"] == original["company"]
        assert profile_data["photo_url"] == original["photo_url"]
        assert profile_data["photo_status"] == original["photo_status"]

    @pytest.mark.asyncio
    async def test_error_does_not_wipe_fields(self):
        """PDL error should not modify any existing profile fields."""
        profile_data = {
            "emails": ["una@fox.com"],
            "title": "CEO",
        }
        original = dict(profile_data)

        with patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich:
            mock_enrich.return_value = _make_pdl_error("API rate limit")
            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=1,
                contact_name="Una Fox",
            )

        assert result["success"] is False
        assert profile_data["title"] == original["title"]


# ---------------------------------------------------------------------------
# 3. Photo download success replaces photo
# ---------------------------------------------------------------------------


class TestPhotoDownloadSuccess:
    @pytest.mark.asyncio
    async def test_photo_download_stores_locally(self):
        """Successful photo download should store the image and update profile."""
        profile_data = {"emails": ["una@fox.com"]}
        pdl_result = _make_pdl_success(photo_url="https://example.com/photo.jpg")

        # Mock both the PDL client and the photo download
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"\xff\xd8\xff" + b"\x00" * 500  # Fake JPEG > 100 bytes

        with (
            patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich,
            patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_enrich.return_value = pdl_result
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=42,
                contact_name="Una Fox",
            )

        assert result["success"] is True
        assert result["photo_updated"] is True
        assert "photo_url" in result["fields_updated"]
        assert profile_data["photo_source"] == "enrichment_provider"
        assert profile_data["photo_status"] == "RESOLVED"
        assert "/api/local-image/" in profile_data["photo_url"]

    @pytest.mark.asyncio
    async def test_photo_replaces_low_priority_source(self):
        """Photo from PDL should replace gravatar/initials/company_logo."""
        for low_source in ["gravatar", "company_logo", "initials"]:
            profile_data = {
                "emails": ["una@fox.com"],
                "photo_url": "https://gravatar.com/old.jpg",
                "photo_status": "RESOLVED",
                "photo_source": low_source,
            }

            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.headers = {"content-type": "image/jpeg"}
            mock_response.content = b"\xff\xd8\xff" + b"\x00" * 500

            pdl_result = _make_pdl_success()

            with (
                patch.object(
                    PDLClient, "enrich_person", new_callable=AsyncMock
                ) as mock_enrich,
                patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls,
            ):
                mock_enrich.return_value = pdl_result
                mock_client = AsyncMock()
                mock_client.__aenter__ = AsyncMock(return_value=mock_client)
                mock_client.__aexit__ = AsyncMock(return_value=False)
                mock_client.get = AsyncMock(return_value=mock_response)
                mock_client_cls.return_value = mock_client

                result = await enrich_contact(
                    profile_data=profile_data,
                    contact_id=42,
                    contact_name="Una Fox",
                )

            assert result["photo_updated"] is True, f"Should replace {low_source}"


# ---------------------------------------------------------------------------
# 4. Photo download failure does NOT replace existing photo
# ---------------------------------------------------------------------------


class TestPhotoDownloadFailure:
    @pytest.mark.asyncio
    async def test_failed_download_preserves_existing_photo(self):
        """If photo download fails, existing RESOLVED photo must not be wiped."""
        profile_data = {
            "emails": ["una@fox.com"],
            "photo_url": "https://example.com/existing.jpg",
            "photo_status": "RESOLVED",
            "photo_source": "uploaded",
        }
        original_photo = profile_data["photo_url"]

        # PDL returns a photo URL but download will fail
        pdl_result = _make_pdl_success(photo_url="https://example.com/broken.jpg")

        mock_response = MagicMock()
        mock_response.status_code = 404  # download fails

        with (
            patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich,
            patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_enrich.return_value = pdl_result
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=42,
                contact_name="Una Fox",
            )

        # Enrichment succeeded for text fields, but photo should NOT be updated
        assert result["success"] is True
        assert result["photo_updated"] is False
        assert profile_data["photo_url"] == original_photo
        assert profile_data["photo_status"] == "RESOLVED"
        assert profile_data["photo_source"] == "uploaded"

    @pytest.mark.asyncio
    async def test_timeout_preserves_existing_photo(self):
        """If photo download times out, existing photo must not be wiped."""
        import httpx

        profile_data = {
            "emails": ["una@fox.com"],
            "photo_url": "https://example.com/existing.jpg",
            "photo_status": "RESOLVED",
            "photo_source": "cached_proxy",
        }
        original_photo = profile_data["photo_url"]

        pdl_result = _make_pdl_success(photo_url="https://example.com/slow.jpg")

        with (
            patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich,
            patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_enrich.return_value = pdl_result
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(side_effect=httpx.TimeoutException("timeout"))
            mock_client_cls.return_value = mock_client

            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=42,
                contact_name="Una Fox",
            )

        assert result["success"] is True
        assert result["photo_updated"] is False
        assert profile_data["photo_url"] == original_photo

    @pytest.mark.asyncio
    async def test_invalid_content_type_preserves_photo(self):
        """Non-image content type should not replace existing photo."""
        profile_data = {"emails": ["una@fox.com"]}
        pdl_result = _make_pdl_success(photo_url="https://example.com/notanimage")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "text/html"}

        with (
            patch.object(PDLClient, "enrich_person", new_callable=AsyncMock) as mock_enrich,
            patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls,
        ):
            mock_enrich.return_value = pdl_result
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await enrich_contact(
                profile_data=profile_data,
                contact_id=42,
                contact_name="Una Fox",
            )

        assert result["photo_updated"] is False

    @pytest.mark.asyncio
    async def test_too_small_image_not_stored(self):
        """Image smaller than 100 bytes should be rejected as broken."""
        result = await _download_and_store_photo(
            photo_url="https://example.com/tiny.jpg",
            contact_id=1,
        )
        # We can't easily mock just this one without the full httpx setup,
        # but we can test the low-level function directly
        # with a mocked response returning tiny content
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.content = b"\xff\xd8" * 5  # 10 bytes, way too small

        with patch("app.services.enrichment_service.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            result = await _download_and_store_photo(
                photo_url="https://example.com/tiny.jpg",
                contact_id=1,
            )

        assert result["stored"] is False
        assert "too small" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_url_rejected(self):
        """Invalid photo URLs should be rejected without making requests."""
        for bad_url in ["", "not-a-url", "ftp://example.com/photo.jpg"]:
            result = await _download_and_store_photo(
                photo_url=bad_url,
                contact_id=1,
            )
            assert result["stored"] is False
            assert "Invalid" in result["error"]


# ---------------------------------------------------------------------------
# 5. Rate limiter enforces max requests/min
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_up_to_max_requests(self):
        """Rate limiter should allow max_requests within the window."""
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        for _ in range(5):
            assert limiter.acquire() is True
        # 6th should be denied
        assert limiter.acquire() is False

    def test_denies_after_max_reached(self):
        """After max_requests, further requests should be denied."""
        limiter = RateLimiter(max_requests=3, window_seconds=60.0)
        for _ in range(3):
            limiter.acquire()
        assert limiter.acquire() is False
        assert limiter.current_count == 3

    def test_wait_time_returns_positive_when_full(self):
        """When full, wait_time should return > 0."""
        limiter = RateLimiter(max_requests=2, window_seconds=60.0)
        limiter.acquire()
        limiter.acquire()
        wait = limiter.wait_time()
        assert wait > 0

    def test_wait_time_returns_zero_when_available(self):
        """When slots available, wait_time should return 0."""
        limiter = RateLimiter(max_requests=5, window_seconds=60.0)
        assert limiter.wait_time() == 0.0

    def test_state_reports_correctly(self):
        """State dict should report current count and remaining."""
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)
        limiter.acquire()
        limiter.acquire()
        state = limiter.state
        assert state["max_requests"] == 10
        assert state["current_count"] == 2
        assert state["remaining"] == 8

    def test_sliding_window_expires_old_requests(self):
        """Requests outside the window should be expired."""
        limiter = RateLimiter(max_requests=2, window_seconds=0.1)
        limiter.acquire()
        limiter.acquire()
        assert limiter.acquire() is False
        # Wait for window to expire
        time.sleep(0.15)
        assert limiter.acquire() is True

    def test_current_count_reflects_active_requests(self):
        limiter = RateLimiter(max_requests=10, window_seconds=60.0)
        assert limiter.current_count == 0
        limiter.acquire()
        assert limiter.current_count == 1
        limiter.acquire()
        assert limiter.current_count == 2


# ---------------------------------------------------------------------------
# 6. Retry logic works for 429
# ---------------------------------------------------------------------------


class TestRetryLogic:
    @pytest.mark.asyncio
    async def test_retries_on_429(self):
        """Client should retry on 429 status with backoff."""
        # Reset rate limiter for test isolation
        from app.clients import pdl_client

        pdl_client._rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)

        mock_429_response = MagicMock()
        mock_429_response.status_code = 429

        mock_200_response = MagicMock()
        mock_200_response.status_code = 200
        mock_200_response.json.return_value = {
            "id": "pdl-retry",
            "full_name": "Test Person",
            "job_title": "Engineer",
            "job_company_name": "TestCo",
            "location_name": "NYC",
            "linkedin_url": "",
            "profile_pic_url": "",
            "likelihood": 0.8,
        }

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return mock_429_response
            return mock_200_response

        with (
            patch.object(pdl_client, "settings") as mock_settings,
            patch("app.clients.pdl_client.httpx.AsyncClient") as mock_client_cls,
            patch("app.clients.pdl_client.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_settings.pdl_api_key = "test-key"
            mock_settings.pdl_enabled = True
            mock_settings.pdl_timeout_ms = 5000
            mock_settings.pdl_max_requests_per_min = 100

            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            client = PDLClient()
            client.api_key = "test-key"
            client.enabled = True
            client.rate_limiter = RateLimiter(max_requests=100)

            result = await client._execute_with_retries({"email": "test@example.com"})

        assert call_count == 2  # 1 failure + 1 success
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_retries_on_500(self):
        """Client should retry on 5xx server errors."""
        mock_500_response = MagicMock()
        mock_500_response.status_code = 500

        mock_200_response = MagicMock()
        mock_200_response.status_code = 200
        mock_200_response.json.return_value = {
            "id": "pdl-500",
            "full_name": "Test",
            "job_title": "",
            "job_company_name": "",
            "location_name": "",
            "linkedin_url": "",
            "profile_pic_url": "",
            "likelihood": 0.5,
        }

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return mock_500_response
            return mock_200_response

        with (
            patch("app.clients.pdl_client.httpx.AsyncClient") as mock_client_cls,
            patch("app.clients.pdl_client.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            client = PDLClient()
            client.api_key = "test-key"
            client.enabled = True
            client.timeout_ms = 5000

            result = await client._execute_with_retries({"email": "test@example.com"})

        assert call_count == 3  # 2 failures + 1 success
        assert result.status == "success"

    @pytest.mark.asyncio
    async def test_exhausted_retries_returns_error(self):
        """After all retries exhausted, should return error result."""
        mock_429_response = MagicMock()
        mock_429_response.status_code = 429

        with (
            patch("app.clients.pdl_client.httpx.AsyncClient") as mock_client_cls,
            patch("app.clients.pdl_client.asyncio.sleep", new_callable=AsyncMock),
        ):
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_429_response)
            mock_client_cls.return_value = mock_client

            client = PDLClient()
            client.api_key = "test-key"
            client.enabled = True
            client.timeout_ms = 5000

            result = await client._execute_with_retries({"email": "test@example.com"})

        assert result.status == "error"
        assert "retries exhausted" in result.error.lower()

    @pytest.mark.asyncio
    async def test_non_retryable_error_returns_immediately(self):
        """Non-retryable errors (e.g., 401, 403) should not retry."""
        mock_401_response = MagicMock()
        mock_401_response.status_code = 401

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_401_response

        with patch("app.clients.pdl_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            client = PDLClient()
            client.api_key = "test-key"
            client.enabled = True
            client.timeout_ms = 5000

            result = await client._execute_with_retries({"email": "test@example.com"})

        assert call_count == 1  # No retries
        assert result.status == "error"
        assert "401" in result.error

    @pytest.mark.asyncio
    async def test_404_returns_no_match(self):
        """PDL 404 should be returned as no_match, not retried."""
        mock_404_response = MagicMock()
        mock_404_response.status_code = 404

        call_count = 0

        async def mock_get(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return mock_404_response

        with patch("app.clients.pdl_client.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = mock_get
            mock_client_cls.return_value = mock_client

            client = PDLClient()
            client.api_key = "test-key"
            client.enabled = True
            client.timeout_ms = 5000

            result = await client._execute_with_retries({"email": "test@example.com"})

        assert call_count == 1
        assert result.status == "no_match"


# ---------------------------------------------------------------------------
# 7. API endpoint returns expected JSON
# ---------------------------------------------------------------------------


class TestEnrichEndpoint:
    def _create_test_entity(self, name="Una Fox", email="una@fox.com"):
        """Create a test entity in the DB and return its ID."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = EntityRecord(name=name, entity_type="person")
        entity.set_emails([email])
        entity.domains = json.dumps({"emails": [email], "name": name})
        session.add(entity)
        session.commit()
        eid = entity.id
        session.close()
        return eid

    def test_enrich_endpoint_returns_result(self):
        """POST /profiles/{id}/enrich should return enrichment result."""
        from fastapi.testclient import TestClient

        from app.api import app

        test_client = TestClient(app)

        eid = self._create_test_entity()

        with (
            patch("app.api.settings") as mock_settings,
            patch(
                "app.services.enrichment_service.enrich_contact",
                new_callable=AsyncMock,
            ) as mock_enrich,
        ):
            mock_settings.pdl_enabled = True
            mock_settings.pdl_api_key = "test-key"
            mock_settings.briefing_api_key = ""
            mock_enrich.return_value = {
                "success": True,
                "fields_updated": ["title", "company"],
                "photo_updated": False,
                "match_confidence": 0.95,
                "error": "",
                "pdl_person_id": "pdl-123",
            }

            response = test_client.post(f"/profiles/{eid}/enrich")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["match_confidence"] == 0.95
        assert "title" in data["fields_updated"]

    def test_enrich_endpoint_404_for_missing_profile(self):
        from fastapi.testclient import TestClient

        from app.api import app

        test_client = TestClient(app)

        with patch("app.api.settings") as mock_settings:
            mock_settings.pdl_enabled = True
            mock_settings.pdl_api_key = "test-key"
            mock_settings.briefing_api_key = ""

            response = test_client.post("/profiles/99999/enrich")

        assert response.status_code == 404

    def test_enrich_endpoint_400_when_pdl_disabled(self):
        from fastapi.testclient import TestClient

        from app.api import app

        test_client = TestClient(app)

        with patch("app.api.settings") as mock_settings:
            mock_settings.pdl_enabled = False
            mock_settings.briefing_api_key = ""

            response = test_client.post("/profiles/1/enrich")

        assert response.status_code == 400
        assert "not enabled" in response.json()["detail"]


class TestDebugEnrichmentEndpoint:
    def test_debug_enrichment_returns_stats(self):
        from fastapi.testclient import TestClient

        from app.api import app

        test_client = TestClient(app)

        response = test_client.get("/debug/enrichment")

        assert response.status_code == 200
        data = response.json()
        assert "pdl_enabled" in data
        assert "pdl_configured" in data
        assert "total_enriched" in data
        assert "total_failed" in data
        assert "last_10_attempts" in data
        assert "rate_limiter_state" in data

    def test_debug_enrichment_shows_rate_limiter_state(self):
        from fastapi.testclient import TestClient

        from app.api import app

        test_client = TestClient(app)

        response = test_client.get("/debug/enrichment")
        data = response.json()

        rl_state = data["rate_limiter_state"]
        assert "max_requests" in rl_state
        assert "current_count" in rl_state
        assert "remaining" in rl_state


# ---------------------------------------------------------------------------
# PDL Client unit tests
# ---------------------------------------------------------------------------


class TestPDLClientUnit:
    @pytest.mark.asyncio
    async def test_disabled_client_returns_error(self):
        """When PDL is disabled, enrich_person should return error immediately."""
        from app.clients import pdl_client

        with patch.object(pdl_client, "settings") as mock_settings:
            mock_settings.pdl_api_key = ""
            mock_settings.pdl_enabled = False
            mock_settings.pdl_max_requests_per_min = 10

            client = PDLClient()
            result = await client.enrich_person(email="test@example.com")

        assert result.status == "error"
        assert "not enabled" in result.error.lower()

    @pytest.mark.asyncio
    async def test_no_params_returns_error(self):
        """With no identifiers provided, should return error."""
        from app.clients import pdl_client

        pdl_client._rate_limiter = RateLimiter(max_requests=100, window_seconds=60.0)

        with patch.object(pdl_client, "settings") as mock_settings:
            mock_settings.pdl_api_key = "test-key"
            mock_settings.pdl_enabled = True
            mock_settings.pdl_max_requests_per_min = 100
            mock_settings.pdl_timeout_ms = 5000

            client = PDLClient()
            result = await client.enrich_person()

        assert result.status == "error"
        assert "No identifiers" in result.error

    def test_parse_success_extracts_fields(self):
        """_parse_success should extract all expected fields."""
        client = PDLClient()
        data = {
            "id": "abc",
            "full_name": "Test User",
            "job_title": "Engineer",
            "job_company_name": "TestCo",
            "location_name": "NYC",
            "linkedin_url": "https://linkedin.com/in/test",
            "profile_pic_url": "https://example.com/pic.jpg",
            "likelihood": 0.92,
        }
        result = client._parse_success(data, 200)
        assert result.status == "success"
        assert result.person_id == "abc"
        assert result.match_confidence == 0.92
        assert result.fields.name == "Test User"
        assert result.fields.title == "Engineer"
        assert result.fields.company == "TestCo"
        assert result.fields.photo_url == "https://example.com/pic.jpg"

    def test_parse_success_handles_missing_fields(self):
        """_parse_success should handle missing/null fields gracefully."""
        client = PDLClient()
        data = {"id": "xyz", "likelihood": None}
        result = client._parse_success(data, 200)
        assert result.status == "success"
        assert result.person_id == "xyz"
        assert result.match_confidence == 0.0
        assert result.fields.name == ""
        assert result.fields.photo_url == ""


# ---------------------------------------------------------------------------
# Enrichment log tests
# ---------------------------------------------------------------------------


class TestEnrichmentLog:
    def test_log_tracks_attempts(self):
        """The enrichment log should track API call attempts."""
        from app.clients.pdl_client import _log_attempt, get_enrichment_log

        initial_len = len(get_enrichment_log())
        _log_attempt({
            "timestamp": time.time(),
            "status": "success",
            "params_keys": ["email"],
        })
        assert len(get_enrichment_log()) == initial_len + 1

    def test_log_capped_at_max_size(self):
        """Log should not grow beyond MAX_LOG_SIZE."""
        from app.clients.pdl_client import MAX_LOG_SIZE, _log_attempt, get_enrichment_log

        # Fill beyond max
        for i in range(MAX_LOG_SIZE + 10):
            _log_attempt({"i": i})
        assert len(get_enrichment_log()) <= MAX_LOG_SIZE
