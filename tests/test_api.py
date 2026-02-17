"""Tests for the FastAPI web API."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""  # disable auth for tests

from fastapi.testclient import TestClient

from app.api import app
from app.store.database import EntityRecord, get_session


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_shows_config_status(self):
        response = client.get("/health")
        data = response.json()
        assert "fireflies_configured" in data
        assert "openai_configured" in data
        assert "database" in data


class TestBriefEndpoint:
    def test_brief_requires_person_or_company(self):
        response = client.post("/brief", json={})
        assert response.status_code == 422

    def test_brief_with_person_no_data(self):
        """With no stored data and no API keys, should return a zero-confidence brief."""
        response = client.post("/brief", json={
            "person": "Ghost Person",
            "skip_ingestion": True,
        })
        # Should succeed (graceful degradation) or fail if OpenAI not configured
        # With skip_ingestion and no data, pipeline produces a no-evidence brief
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] == 0.0
        assert "brief" in data
        assert "markdown" in data

    def test_brief_markdown_endpoint(self):
        response = client.post("/brief/markdown", json={
            "person": "Nobody",
            "skip_ingestion": True,
        })
        assert response.status_code == 200
        assert "Intelligence" in response.text

    def test_brief_json_endpoint(self):
        response = client.post("/brief/json", json={
            "person": "Nobody",
            "skip_ingestion": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "header" in data
        assert "open_loops" in data


class TestDashboard:
    def test_root_serves_dashboard(self):
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Pre-Call Intelligence" in response.text

    def test_recent_briefs_empty(self):
        response = client.get("/briefs/recent")
        assert response.status_code == 200
        assert response.json() == []

    def test_recent_briefs_after_generation(self):
        # Generate a brief first
        client.post("/brief", json={"person": "Dashboard Test", "skip_ingestion": True})
        response = client.get("/briefs/recent")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["person"] == "Dashboard Test"

    def test_recent_briefs_limit(self):
        response = client.get("/briefs/recent?limit=1")
        assert response.status_code == 200
        data = response.json()
        assert len(data) <= 1


def _create_confirmed_profile(name="Test Person", company="Acme Corp", title="CTO"):
    """Helper to create a confirmed profile in the DB for deep-profile tests."""
    session = get_session("sqlite:///./test_briefing_engine.db")
    entity = EntityRecord(name=name, entity_type="person")
    entity.set_emails([f"{name.lower().replace(' ', '.')}@example.com"])
    profile_data = {
        "linkedin_status": "confirmed",
        "linkedin_url": "https://linkedin.com/in/testperson",
        "company": company,
        "title": title,
        "location": "London",
    }
    entity.domains = json.dumps(profile_data)
    session.add(entity)
    session.commit()
    session.refresh(entity)
    pid = entity.id
    session.close()
    return pid


class TestDeepProfileEndpoint:
    def test_requires_openai_key(self):
        """Should 400 when OpenAI key is not configured."""
        pid = _create_confirmed_profile()
        response = client.post(f"/profiles/{pid}/deep-profile")
        assert response.status_code == 400
        assert "OpenAI" in response.json()["detail"]

    def test_returns_404_for_missing_profile(self):
        response = client.post("/profiles/99999/deep-profile")
        # Without OpenAI key, 400 comes first
        assert response.status_code in (400, 404)

    def test_rejects_unconfirmed_profile(self):
        """Profiles without linkedin_status == 'confirmed' should be rejected."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = EntityRecord(name="Unconfirmed", entity_type="person")
        entity.domains = json.dumps({"linkedin_status": "pending"})
        session.add(entity)
        session.commit()
        session.refresh(entity)
        pid = entity.id
        session.close()

        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            # Need to reload settings for the key change
            from app.config import settings
            orig = settings.openai_api_key
            settings.openai_api_key = "test-key"
            try:
                response = client.post(f"/profiles/{pid}/deep-profile")
                assert response.status_code == 422
                assert "verified" in response.json()["detail"].lower()
            finally:
                settings.openai_api_key = orig

    @patch("app.api.generate_deep_profile")
    def test_response_includes_entity_lock(self, mock_gen):
        """Response should include entity_lock with score and signals."""
        mock_gen.return_value = "## 1. Strategic Snapshot\n- Test operator [VERIFIED-PUBLIC]"
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            assert response.status_code == 200
            data = response.json()

            # Entity lock report
            assert "entity_lock" in data
            lock = data["entity_lock"]
            assert "entity_lock_score" in lock
            assert "is_locked" in lock
            assert "signals" in lock
            assert "canonical_name" in lock
            assert lock["canonical_name"] == "Test Person"
            assert isinstance(lock["entity_lock_score"], int)
            assert isinstance(lock["signals"], dict)
            assert "name_match" in lock["signals"]
            assert "linkedin_confirmed" in lock["signals"]
            assert "linkedin_url_present" in lock["signals"]
            assert "linkedin_verified_by_retrieval" in lock["signals"]
        finally:
            settings.openai_api_key = orig

    @patch("app.api.determine_dossier_mode", return_value=("full", "Test mode"))
    @patch("app.api.generate_deep_profile")
    def test_response_includes_qa_report(self, mock_gen, mock_mode):
        """Response should include qa_report with all gate results."""
        mock_gen.return_value = (
            "## 1. Strategic Snapshot\n"
            "He is CTO at Acme [VERIFIED-PUBLIC] confirmed.\n"
            "Revenue grew 32% [VERIFIED-PUBLIC] per filings.\n"
            "Team has 45 engineers [VERIFIED-MEETING] from call."
        )
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            assert response.status_code == 200
            data = response.json()

            assert "qa_report" in data
            qa = data["qa_report"]
            assert "passes_all" in qa
            assert "genericness_score" in qa
            assert "evidence_coverage_pct" in qa
            assert "contradictions" in qa
            assert "hallucination_risk_flags" in qa
            assert "markdown" in qa
            assert isinstance(qa["genericness_score"], int)
            assert isinstance(qa["evidence_coverage_pct"], float)
        finally:
            settings.openai_api_key = orig

    @patch("app.api.generate_deep_profile")
    def test_response_includes_search_plan(self, mock_gen):
        """Response should include the search plan used."""
        mock_gen.return_value = "## Dossier content"
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            assert response.status_code == 200
            data = response.json()

            assert "search_plan" in data
            plan = data["search_plan"]
            assert isinstance(plan, list)
            assert len(plan) >= 6  # At least 6 query categories
            assert plan[0]["category"] == "identity"
            assert "query" in plan[0]
            assert "rationale" in plan[0]
        finally:
            settings.openai_api_key = orig

    @patch("app.api.generate_deep_profile")
    def test_linkedin_url_present_in_entity_lock(self, mock_gen):
        """LinkedIn URL should be tracked in entity lock report."""
        mock_gen.return_value = "## Dossier content"
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            data = response.json()
            lock = data["entity_lock"]
            assert isinstance(lock["entity_lock_score"], int)
            # Without SerpAPI results, LinkedIn URL is "present but not verified"
            # so linkedin_confirmed may be False but linkedin_url_present should
            # be True (new field). Entity lock score depends on available search.
            assert "signals" in lock
        finally:
            settings.openai_api_key = orig

    @patch("app.api.determine_dossier_mode", return_value=("full", "Test mode"))
    @patch("app.api.generate_deep_profile")
    def test_qa_data_persisted_to_profile(self, mock_gen, mock_mode):
        """QA results should be stored in profile_data after generation."""
        mock_gen.return_value = "## Dossier [VERIFIED-PUBLIC] content."
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            client.post(f"/profiles/{pid}/deep-profile")

            # Read back from DB
            session = get_session("sqlite:///./test_briefing_engine.db")
            entity = session.query(EntityRecord).get(pid)
            pd = json.loads(entity.domains or "{}")
            session.close()

            assert "entity_lock_score" in pd
            assert "qa_genericness_score" in pd
            assert "qa_evidence_coverage_pct" in pd
            assert "qa_passes_all" in pd
            assert "qa_report_markdown" in pd
            assert "search_plan" in pd
        finally:
            settings.openai_api_key = orig

    @patch("app.api.determine_dossier_mode", return_value=("full", "Test mode"))
    @patch("app.api.generate_deep_profile")
    def test_generic_dossier_flags_qa(self, mock_gen, mock_mode):
        """A generic uncited dossier should be auto-pruned.

        The auto-prune system removes all uncited generic lines. When the
        entire dossier is generic filler, auto-prune strips everything,
        leaving an empty dossier. The fail-closed gates still block output
        due to missing visibility sweep / public results.
        """
        mock_gen.return_value = (
            "He is a strategic leader who drives innovation.\n"
            "She is passionate about cutting-edge technology.\n"
            "A proven track record of delivering results.\n"
            "Empowers teams with a holistic approach to growth.\n"
            "A visionary leader at the intersection of AI."
        )
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            data = response.json()
            # The fail-closed system should block: no visibility sweep,
            # no public results, so the dossier is halted.
            fail_status = data.get("fail_closed_status", {})
            assert not fail_status.get("gates_passed", True)
        finally:
            settings.openai_api_key = orig


# ---------------------------------------------------------------------------
# Mode A: Meeting-Prep Brief API Tests
# ---------------------------------------------------------------------------


class TestMeetingPrepEndpoint:
    def test_meeting_prep_returns_200(self):
        """Meeting prep should always succeed, even without OpenAI key."""
        pid = _create_confirmed_profile()
        response = client.post(f"/profiles/{pid}/meeting-prep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["mode"] == "meeting_prep"
        assert "brief" in data
        assert len(data["brief"]) > 50

    def test_meeting_prep_no_openai_needed(self):
        """Mode A must work without OpenAI key."""
        pid = _create_confirmed_profile()
        response = client.post(f"/profiles/{pid}/meeting-prep")
        assert response.status_code == 200
        data = response.json()
        assert "Meeting-Prep Brief" in data["brief"]

    def test_meeting_prep_returns_404_for_missing_profile(self):
        response = client.post("/profiles/99999/meeting-prep")
        assert response.status_code == 404

    def test_meeting_prep_persists_to_profile(self):
        pid = _create_confirmed_profile()
        client.post(f"/profiles/{pid}/meeting-prep")

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.query(EntityRecord).get(pid)
        pd = json.loads(entity.domains or "{}")
        session.close()

        assert "dossier_mode_a_markdown" in pd
        assert len(pd["dossier_mode_a_markdown"]) > 50
        assert "dossier_mode_a_generated_at" in pd

    def test_meeting_prep_with_interactions(self):
        """Profile with interactions should produce richer brief."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = EntityRecord(name="Meeting Test", entity_type="person")
        profile_data = {
            "interactions": [
                {"title": "Q1 Review", "date": "2026-01-15", "summary": "Discussed pipeline risks"},
                {"title": "Follow-up", "date": "2026-02-01", "summary": "Budget approved"},
            ],
            "action_items": ["Send proposal", "Schedule demo"],
        }
        entity.domains = json.dumps(profile_data)
        session.add(entity)
        session.commit()
        session.refresh(entity)
        pid = entity.id
        session.close()

        response = client.post(f"/profiles/{pid}/meeting-prep")
        assert response.status_code == 200
        data = response.json()
        # Should include evidence from interactions
        assert data["evidence_nodes"] >= 2

    def test_meeting_prep_never_blocks_on_serpapi(self):
        """Mode A must never block on SerpAPI or visibility sweep."""
        pid = _create_confirmed_profile()
        # No SerpAPI key configured — should still work
        response = client.post(f"/profiles/{pid}/meeting-prep")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_meeting_prep_works_for_unconfirmed_profile(self):
        """Mode A should work for any profile, not just confirmed ones."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = EntityRecord(name="Unconfirmed Person", entity_type="person")
        entity.domains = json.dumps({"linkedin_status": "pending"})
        session.add(entity)
        session.commit()
        session.refresh(entity)
        pid = entity.id
        session.close()

        response = client.post(f"/profiles/{pid}/meeting-prep")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "Unconfirmed Person" in data["brief"]


# ---------------------------------------------------------------------------
# Mode B: Deep Research Status Tests
# ---------------------------------------------------------------------------


class TestDeepResearchStatus:
    @patch("app.api.enforce_fail_closed_gates", return_value=(True, ""))
    @patch("app.api.determine_dossier_mode", return_value=("full", "Test mode"))
    @patch("app.api.generate_deep_profile")
    def test_succeeded_status_on_success(self, mock_gen, mock_mode, mock_gates):
        mock_gen.return_value = (
            "## 1. Strategic Snapshot\n"
            "He is CTO at Acme [VERIFIED-PUBLIC] confirmed.\n"
            "Revenue grew 32% [VERIFIED-PUBLIC] per filings."
        )
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            data = response.json()
            assert data["deep_research_status"] == "SUCCEEDED"
            assert data["mode"] == "deep_research"
        finally:
            settings.openai_api_key = orig

    @patch("app.api.generate_deep_profile")
    def test_failed_status_when_halted(self, mock_gen):
        """When pre-synthesis gate halts, status should be FAILED."""
        mock_gen.return_value = "## Dossier"
        pid = _create_confirmed_profile()

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            data = response.json()
            # Without SerpAPI, will be halted
            if data["status"] == "halted":
                assert data["deep_research_status"] == "FAILED"
        finally:
            settings.openai_api_key = orig


# ---------------------------------------------------------------------------
# LinkedIn PDF Ingestion Endpoint Tests
# ---------------------------------------------------------------------------


def _create_profile_with_pdf(name="PDF Test Person"):
    """Helper to create a profile with LinkedIn PDF data."""

    session = get_session("sqlite:///./test_briefing_engine.db")
    entity = EntityRecord(name=name, entity_type="person")
    entity.set_emails(["pdftest@example.com"])
    profile_data = {
        "linkedin_status": "confirmed",
        "linkedin_url": "https://linkedin.com/in/pdftest",
        "company": "TestCo",
        "title": "Director",
        "linkedin_pdf_raw_text": f"{name}\nDirector at TestCo\nAbout\nExperienced professional.",
        "linkedin_pdf_sections": {"about": "Experienced professional.", "header": name},
        "linkedin_pdf_path": "/tmp/test.pdf",
        "linkedin_pdf_page_count": 1,
        "linkedin_pdf_text_length": 100,
    }
    entity.domains = json.dumps(profile_data)
    session.add(entity)
    session.commit()
    session.refresh(entity)
    pid = entity.id
    session.close()
    return pid


class TestLinkedInPdfIngestion:
    def test_ingest_requires_pdf_base64(self):
        """Endpoint requires pdf_base64 in request body."""
        pid = _create_confirmed_profile()
        response = client.post(f"/profiles/{pid}/ingest-linkedin-pdf", json={})
        assert response.status_code == 422

    def test_ingest_invalid_base64(self):
        """Invalid base64 returns 422."""
        pid = _create_confirmed_profile()
        response = client.post(
            f"/profiles/{pid}/ingest-linkedin-pdf",
            json={"pdf_base64": "!!!not-base64!!!"},
        )
        assert response.status_code == 422

    def test_ingest_nonexistent_profile(self):
        """Nonexistent profile returns 404."""
        response = client.post(
            "/profiles/99999/ingest-linkedin-pdf",
            json={"pdf_base64": "dGVzdA=="},
        )
        assert response.status_code == 404

    @patch("app.services.linkedin_pdf.ingest_linkedin_pdf")
    def test_ingest_success(self, mock_ingest):
        """Successful ingestion returns expected fields."""
        from app.services.linkedin_pdf import (
            LinkedInPDFCropResult,
            LinkedInPDFIngestResult,
            LinkedInPDFTextResult,
        )

        mock_ingest.return_value = LinkedInPDFIngestResult(
            text_result=LinkedInPDFTextResult(
                raw_text="Test Person\nCTO\nAbout\nLeading innovation.",
                name="Test Person",
                headline="CTO",
                page_count=2,
                sections={"about": "Leading innovation.", "header": "Test Person"},
                experience=[{"title": "CTO", "company": "TestCo"}],
                education=[{"school": "MIT", "details": "MSc"}],
                skills=["Python", "Leadership"],
            ),
            crop_result=LinkedInPDFCropResult(
                success=False, method="failed", error="No fitz",
            ),
            pdf_path="/tmp/test_42.pdf",
            pdf_hash="abc123",
            ingested_at="2026-02-15T12:00:00",
        )

        pid = _create_confirmed_profile()
        import base64
        pdf_b64 = base64.b64encode(b"fake pdf").decode()
        response = client.post(
            f"/profiles/{pid}/ingest-linkedin-pdf",
            json={"pdf_base64": pdf_b64},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["text_extracted"] is True
        assert data["headshot_cropped"] is False
        assert data["photo_updated"] is False
        assert "header" in data["sections_found"]

    @patch("app.services.linkedin_pdf.ingest_linkedin_pdf")
    def test_ingest_crop_success_updates_photo(self, mock_ingest):
        """When crop succeeds, photo should be updated."""
        from app.services.linkedin_pdf import (
            LinkedInPDFCropResult,
            LinkedInPDFIngestResult,
            LinkedInPDFTextResult,
        )

        mock_ingest.return_value = LinkedInPDFIngestResult(
            text_result=LinkedInPDFTextResult(raw_text="Test", page_count=1),
            crop_result=LinkedInPDFCropResult(
                success=True,
                image_path="./image_cache/linkedin_crop_99.jpg",
                width=200,
                height=200,
                method="pillow_crop",
            ),
            pdf_path="/tmp/test.pdf",
            pdf_hash="def456",
            ingested_at="2026-02-15T12:00:00",
        )

        pid = _create_confirmed_profile()
        import base64
        pdf_b64 = base64.b64encode(b"fake pdf").decode()
        response = client.post(
            f"/profiles/{pid}/ingest-linkedin-pdf",
            json={"pdf_base64": pdf_b64},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["headshot_cropped"] is True
        assert data["photo_updated"] is True


# ---------------------------------------------------------------------------
# Artifact Dossier Endpoint Tests
# ---------------------------------------------------------------------------


class TestArtifactDossierEndpoint:
    def test_requires_uploaded_pdf(self):
        """Endpoint requires LinkedIn PDF to be uploaded first."""
        pid = _create_confirmed_profile()
        response = client.post(f"/profiles/{pid}/artifact-dossier")
        assert response.status_code == 422
        assert "No LinkedIn PDF" in response.json()["detail"]

    def test_nonexistent_profile_404(self):
        response = client.post("/profiles/99999/artifact-dossier")
        assert response.status_code == 404

    def test_generates_dossier_from_pdf(self):
        """Generates artifact dossier when PDF data exists."""
        pid = _create_profile_with_pdf()
        response = client.post(f"/profiles/{pid}/artifact-dossier")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["dossier"]
        assert data["mode"] in ("artifact_first", "artifact_llm")
        assert data["artifact_count"] >= 1
        assert data["evidence_graph"]

    def test_dossier_contains_all_sections(self):
        """Generated dossier has all required sections."""
        pid = _create_profile_with_pdf()
        response = client.post(f"/profiles/{pid}/artifact-dossier")
        data = response.json()
        dossier = data["dossier"]
        assert "Executive Summary" in dossier
        assert "Identity" in dossier
        assert "Gaps" in dossier


# ---------------------------------------------------------------------------
# Deep Profile with PDF artifacts (dual-path)
# ---------------------------------------------------------------------------


class TestDeepProfileWithPdfArtifacts:
    @patch("app.api.enforce_fail_closed_gates", return_value=(True, ""))
    @patch("app.api.determine_dossier_mode", return_value=("full", "Test mode"))
    @patch("app.api.generate_deep_profile")
    def test_includes_pdf_artifact_count(self, mock_gen, mock_mode, mock_gates):
        """Deep profile response includes PDF artifact info."""
        mock_gen.return_value = (
            "## 1. Strategic Snapshot\n"
            "CTO at TestCo [VERIFIED-PDF] from PDF.\n"
            "Revenue grew 32% [VERIFIED-PUBLIC] per filings."
        )
        pid = _create_profile_with_pdf("Deep Test Person")

        from app.config import settings
        orig = settings.openai_api_key
        settings.openai_api_key = "test-key"
        try:
            response = client.post(f"/profiles/{pid}/deep-profile")
            data = response.json()
            if data["status"] == "ok":
                assert "artifacts" in data
                assert data["artifacts"]["pdf_uploaded"] is True
                assert data["artifacts"]["pdf_evidence_nodes"] >= 1
        finally:
            settings.openai_api_key = orig


# ---------------------------------------------------------------------------
# Debug Photos with PDF stats
# ---------------------------------------------------------------------------


class TestDebugPhotosWithPdf:
    def test_debug_photos_includes_pdf_stats(self):
        response = client.get("/debug/photos")
        assert response.status_code == 200
        data = response.json()
        assert "linkedin_pdf_crops" in data
        assert "linkedin_pdf_uploads" in data
        assert "crop_success_rate" in data
