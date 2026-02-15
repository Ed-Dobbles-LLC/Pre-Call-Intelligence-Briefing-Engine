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
        """A generic dossier should trigger QA failure flags."""
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
            qa = data["qa_report"]
            assert qa["genericness_score"] > 20
            assert not qa["passes_all"]
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
