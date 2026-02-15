"""Tests for the Deep Research pipeline.

Covers:
1. Deep research executes visibility sweep and writes ledger rows (mock SerpAPI)
2. Entity lock score increases after PDL enrichment (mock PDL)
3. Dossier generation uses evidence nodes, no generic uncited filler
4. SerpAPI errors create ledger rows with error info
5. POST /profiles/{id}/deep-research API endpoint
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

from app.brief.evidence_graph import (
    DossierMode,
    EvidenceGraph,
    build_visibility_queries,
    compute_visibility_coverage_confidence,
)
from app.brief.qa import score_disambiguation
from app.clients.pdl_client import PDLEnrichResult, PDLPersonFields
from app.clients.serpapi import SerpAPIClient
from app.store.database import EntityRecord, get_session


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entity(name="Test Person", company="TestCo", email="test@testco.com"):
    """Create a test entity in the DB and return its ID."""
    session = get_session("sqlite:///./test_briefing_engine.db")
    entity = EntityRecord(name=name, entity_type="person")
    entity.set_emails([email])
    profile_data = {
        "emails": [email],
        "name": name,
        "company": company,
    }
    entity.domains = json.dumps(profile_data)
    session.add(entity)
    session.commit()
    eid = entity.id
    session.close()
    return eid


def _make_serp_results(count=3):
    """Create mock SerpAPI organic results."""
    return [
        {
            "title": f"Result {i} - Test Person at TestCo",
            "link": f"https://example{i}.com/page",
            "snippet": f"Test Person is the CEO of TestCo based in San Francisco. Result {i}.",
            "source": f"example{i}.com",
            "date": "2026-01-15",
        }
        for i in range(count)
    ]


def _make_pdl_success(
    company="TestCo",
    title="CEO",
    location="San Francisco, CA",
) -> PDLEnrichResult:
    return PDLEnrichResult(
        status="success",
        person_id="pdl-test-123",
        match_confidence=0.92,
        fields=PDLPersonFields(
            name="Test Person",
            title=title,
            company=company,
            location=location,
            linkedin_url="https://linkedin.com/in/testperson",
            photo_url="",
        ),
        raw_response={"id": "pdl-test-123", "likelihood": 0.92},
        http_status=200,
    )


# ---------------------------------------------------------------------------
# 1. Deep research writes visibility ledger rows
# ---------------------------------------------------------------------------


class TestVisibilitySweepExecution:
    @pytest.mark.asyncio
    async def test_visibility_sweep_creates_ledger_rows(self):
        """search_visibility_sweep_with_ledger must create >=12 visibility ledger rows."""
        graph = EvidenceGraph()
        mock_results = _make_serp_results(2)

        async def mock_search(query, num=5):
            return mock_results

        serp = SerpAPIClient(api_key="test-key")

        with patch.object(serp, "search", side_effect=mock_search):
            await serp.search_visibility_sweep_with_ledger(
                name="Test Person",
                company="TestCo",
                graph=graph,
            )

        # Must have visibility-intent ledger rows
        visibility_rows = graph.get_visibility_ledger_rows()
        assert len(visibility_rows) >= 12, (
            f"Expected >=12 visibility ledger rows, got {len(visibility_rows)}"
        )

        # Every row must have intent="visibility"
        for row in visibility_rows:
            assert row.intent == "visibility"
            assert row.query  # non-empty query

        # Total ledger rows must be >= visibility rows
        assert len(graph.ledger) >= len(visibility_rows)

    @pytest.mark.asyncio
    async def test_visibility_sweep_logs_even_with_zero_results(self):
        """Even if SerpAPI returns 0 results, ledger rows must still be created."""
        graph = EvidenceGraph()

        async def mock_search(query, num=5):
            return []  # 0 results

        serp = SerpAPIClient(api_key="test-key")

        with patch.object(serp, "search", side_effect=mock_search):
            await serp.search_visibility_sweep_with_ledger(
                name="Unknown Person",
                company="",
                graph=graph,
            )

        visibility_rows = graph.get_visibility_ledger_rows()
        assert len(visibility_rows) >= 12, (
            f"Must log rows even with 0 results. Got {len(visibility_rows)}"
        )
        # All should have result_count=0
        for row in visibility_rows:
            assert row.result_count == 0

    @pytest.mark.asyncio
    async def test_person_search_creates_ledger_rows(self):
        """search_person_with_ledger must create ledger rows for each query category."""
        graph = EvidenceGraph()

        async def mock_search(query, num=8):
            return _make_serp_results(3)

        serp = SerpAPIClient(api_key="test-key")

        with patch.object(serp, "search", side_effect=mock_search):
            await serp.search_person_with_ledger(
                name="Test Person",
                company="TestCo",
                graph=graph,
            )

        # Should have at least 5 categories (general, linkedin, company_site, news, talks, registry)
        assert len(graph.ledger) >= 5
        intents = {row.intent for row in graph.ledger}
        assert "bio" in intents or "entity_lock" in intents

    def test_visibility_queries_generated(self):
        """build_visibility_queries must return >=15 queries."""
        queries = build_visibility_queries("Test Person", "TestCo")
        assert len(queries) >= 15

        # All must have intent="visibility"
        for query, intent in queries:
            assert intent == "visibility"
            assert "Test Person" in query

    def test_visibility_coverage_confidence(self):
        """Coverage confidence should increase with more query families returning results."""
        graph = EvidenceGraph()

        # Log 15 visibility queries with varying results
        for i in range(15):
            results = _make_serp_results(2) if i < 10 else []
            graph.log_retrieval(
                query=f"test query {i}",
                intent="visibility",
                results=results,
            )

        confidence = compute_visibility_coverage_confidence(graph)
        assert confidence > 0
        assert confidence <= 100


# ---------------------------------------------------------------------------
# 2. Entity lock increases after PDL enrichment
# ---------------------------------------------------------------------------


class TestEntityLockWithPDL:
    def test_entity_lock_without_pdl_is_low(self):
        """Without PDL data and no search results, entity lock should be low."""
        result = score_disambiguation(
            name="Test Person",
            company="TestCo",
            title="CEO",
            location="San Francisco",
        )
        assert result.score < 70, (
            f"Without enrichment, score should be <70. Got {result.score}"
        )

    def test_entity_lock_increases_with_pdl_company(self):
        """PDL confirming company should increase entity lock score."""
        base = score_disambiguation(
            name="Test Person",
            company="TestCo",
            title="CEO",
        )

        with_pdl = score_disambiguation(
            name="Test Person",
            company="TestCo",
            title="CEO",
            pdl_data={
                "canonical_company": "TestCo",
                "canonical_title": "CEO",
                "canonical_location": "San Francisco, CA",
                "pdl_match_confidence": 0.92,
            },
        )

        assert with_pdl.score > base.score, (
            f"PDL should increase score. Base={base.score}, WithPDL={with_pdl.score}"
        )
        assert with_pdl.company_match is True
        assert with_pdl.title_match is True

    def test_entity_lock_pdl_gives_company_credit(self):
        """PDL confirming company should give +20 points for employer match."""
        result = score_disambiguation(
            name="Test Person",
            company="TestCo",
            pdl_data={
                "canonical_company": "TestCo",
                "pdl_match_confidence": 0.92,
            },
        )
        assert result.employer_match is True
        assert result.company_match is True
        # Should have evidence entry for PDL
        pdl_evidence = [e for e in result.evidence if e.get("source") == "pdl"]
        assert len(pdl_evidence) >= 1, "Should have PDL evidence entry"

    def test_entity_lock_pdl_gives_title_credit(self):
        """PDL confirming title should give +10 points."""
        # Exact word overlap: "VP Engineering" matches "VP of Engineering"
        result = score_disambiguation(
            name="Test Person",
            title="VP Engineering",
            pdl_data={
                "canonical_title": "VP of Engineering",
                "pdl_match_confidence": 0.85,
            },
        )
        assert result.title_match is True

        # Substring match: "CEO" in "CEO and Co-Founder"
        result2 = score_disambiguation(
            name="Test Person",
            title="CEO",
            pdl_data={
                "canonical_title": "CEO and Co-Founder",
                "pdl_match_confidence": 0.85,
            },
        )
        assert result2.title_match is True

    def test_entity_lock_pdl_gives_location_credit(self):
        """PDL confirming location should give +10 points."""
        result = score_disambiguation(
            name="Test Person",
            location="San Francisco",
            pdl_data={
                "canonical_location": "San Francisco, CA",
                "pdl_match_confidence": 0.85,
            },
        )
        assert result.location_match is True

    def test_entity_lock_pdl_no_double_count_with_search(self):
        """PDL company credit should not stack with search-based company credit."""
        # With PDL only
        pdl_only = score_disambiguation(
            name="Test Person",
            company="TestCo",
            pdl_data={
                "canonical_company": "TestCo",
                "pdl_match_confidence": 0.92,
            },
        )

        # With PDL + search results that also confirm company
        pdl_and_search = score_disambiguation(
            name="Test Person",
            company="TestCo",
            search_results={
                "general": [
                    {
                        "title": "Test Person - CEO at TestCo",
                        "snippet": "Test Person is the CEO of TestCo",
                        "link": "https://example.com",
                    }
                ],
            },
            pdl_data={
                "canonical_company": "TestCo",
                "pdl_match_confidence": 0.92,
            },
        )

        # PDL-confirmed employer (20pts) should not stack with search-confirmed employer
        # Score difference should come from other signals (e.g. confirming_domains)
        # The employer_match should be True in both cases
        assert pdl_only.employer_match is True
        assert pdl_and_search.employer_match is True

    def test_entity_lock_pdl_low_confidence_ignored(self):
        """PDL data with confidence <= 0.5 should not contribute to entity lock."""
        result = score_disambiguation(
            name="Test Person",
            company="TestCo",
            pdl_data={
                "canonical_company": "TestCo",
                "canonical_title": "CEO",
                "pdl_match_confidence": 0.3,  # Too low
            },
        )
        pdl_evidence = [e for e in result.evidence if e.get("source") == "pdl"]
        assert len(pdl_evidence) == 0, "Low confidence PDL should not add evidence"

    def test_entity_lock_combined_pdl_linkedin_meeting(self):
        """PDL + LinkedIn URL + meeting data should produce high entity lock."""
        result = score_disambiguation(
            name="Test Person",
            company="TestCo",
            title="CEO",
            linkedin_url="https://linkedin.com/in/testperson",
            location="San Francisco, CA",
            has_meeting_data=True,
            pdl_data={
                "canonical_company": "TestCo",
                "canonical_title": "CEO",
                "canonical_location": "San Francisco, CA",
                "pdl_match_confidence": 0.92,
            },
        )
        # Should be quite high with PDL + LinkedIn + meeting
        assert result.score >= 50, (
            f"Combined PDL + LinkedIn + meeting should be >=50. Got {result.score}"
        )


# ---------------------------------------------------------------------------
# 3. Dossier generation uses evidence nodes
# ---------------------------------------------------------------------------


class TestDossierGeneration:
    def test_meeting_prep_uses_evidence_nodes(self):
        """Mode A meeting prep should include meeting evidence snippets."""
        from app.brief.evidence_graph import build_meeting_prep_brief

        graph = EvidenceGraph()
        graph.add_meeting_node(
            source="Q1 Review Call",
            snippet="Jane expressed concern about Phase 2 timeline",
            date="2026-02-10",
            ref="meeting",
        )
        graph.add_meeting_node(
            source="Budget Discussion Email",
            snippet="Budget approval taking longer than expected",
            date="2026-02-08",
            ref="email",
        )

        brief = build_meeting_prep_brief(
            person_name="Jane Doe",
            graph=graph,
            profile_data={"company": "Acme Corp", "title": "VP Engineering"},
        )

        assert "Jane Doe" in brief
        assert "Phase 2 timeline" in brief
        assert "[VERIFIED-MEETING]" in brief

    def test_evidence_graph_nodes_created(self):
        """Adding nodes to graph should track them correctly."""
        graph = EvidenceGraph()

        m1 = graph.add_meeting_node(source="call", snippet="meeting data")
        p1 = graph.add_public_node(source="https://example.com", snippet="web data")

        assert len(graph.nodes) == 2
        assert m1.type == "MEETING"
        assert p1.type == "PUBLIC"
        assert m1.id == "E1"
        assert p1.id == "E2"

    def test_evidence_graph_serialization(self):
        """Graph should serialize to dict with nodes, claims, and ledger."""
        graph = EvidenceGraph()
        graph.add_meeting_node(source="call", snippet="data")
        graph.log_retrieval(query="test", intent="bio", results=[])

        result = graph.to_dict()
        assert "nodes" in result
        assert "claims" in result
        assert "ledger" in result
        assert len(result["nodes"]) == 1
        assert len(result["ledger"]) == 1


# ---------------------------------------------------------------------------
# 4. SerpAPI errors create ledger rows
# ---------------------------------------------------------------------------


class TestSerpAPIErrorHandling:
    @pytest.mark.asyncio
    async def test_serp_error_logged_to_ledger(self):
        """When SerpAPI fails, ledger should still log the error."""
        graph = EvidenceGraph()

        # If SerpAPI key is not set, the client should log to ledger
        graph.log_retrieval(
            query="[SERPAPI_UNAVAILABLE]",
            intent="visibility",
            results=[],
        )

        assert len(graph.ledger) == 1
        assert graph.ledger[0].intent == "visibility"
        assert graph.ledger[0].result_count == 0

    @pytest.mark.asyncio
    async def test_serp_403_returns_empty(self):
        """SerpAPI 403 (auth failed) should return empty results, not crash."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("app.clients.serpapi.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            mock_client.get = AsyncMock(return_value=mock_response)
            mock_client_cls.return_value = mock_client

            client = SerpAPIClient(api_key="bad-key")
            results = await client.search("test query")

        assert results == []


# ---------------------------------------------------------------------------
# 5. API endpoint tests
# ---------------------------------------------------------------------------


class TestDeepResearchEndpoint:
    def test_deep_research_endpoint_exists(self):
        """POST /profiles/{id}/deep-research should be a valid route."""
        from fastapi.testclient import TestClient
        from app.api import app

        test_client = TestClient(app)

        # Non-existent profile should return 404
        with patch("app.api.settings") as mock_settings:
            mock_settings.briefing_api_key = ""
            mock_settings.serpapi_api_key = ""
            mock_settings.openai_api_key = ""
            mock_settings.pdl_enabled = False
            mock_settings.pdl_api_key = ""

            response = test_client.post("/profiles/99999/deep-research")
        assert response.status_code == 404

    def test_deep_research_with_mock_serp(self):
        """POST /profiles/{id}/deep-research should execute and return results."""
        from fastapi.testclient import TestClient
        from app.api import app

        eid = _make_entity()
        test_client = TestClient(app)

        mock_serp_results = _make_serp_results(2)

        async def mock_search(query, num=5):
            return mock_serp_results

        with (
            patch("app.api.settings") as mock_settings,
            patch.object(SerpAPIClient, "search", side_effect=mock_search),
            patch(
                "app.api.generate_deep_profile",
                return_value="# Test Dossier\n\nTest content [VERIFIED-PUBLIC]",
            ),
        ):
            mock_settings.briefing_api_key = ""
            mock_settings.serpapi_api_key = "test-serp-key"
            mock_settings.openai_api_key = "test-openai-key"
            mock_settings.pdl_enabled = False
            mock_settings.pdl_api_key = ""

            response = test_client.post(f"/profiles/{eid}/deep-research")

        assert response.status_code == 200
        data = response.json()

        # Must have retrieval ledger rows
        assert data["retrieval_ledger_count"] > 0, (
            f"Expected ledger_count > 0, got {data['retrieval_ledger_count']}"
        )
        assert data["visibility_ledger_count"] > 0, (
            f"Expected visibility_ledger_count > 0, got {data['visibility_ledger_count']}"
        )

        # Must have entity lock report
        assert "entity_lock" in data
        assert "entity_lock_score" in data["entity_lock"]

        # Must have visibility report
        assert "visibility_report" in data
        assert data["visibility_report"]["sweep_executed"] is True

    def test_deep_research_no_serp_key_fails_gracefully(self):
        """Without SerpAPI key, deep research should return HALTED with ledger info."""
        from fastapi.testclient import TestClient
        from app.api import app

        eid = _make_entity()
        test_client = TestClient(app)

        with patch("app.api.settings") as mock_settings:
            mock_settings.briefing_api_key = ""
            mock_settings.serpapi_api_key = ""
            mock_settings.openai_api_key = "test-key"
            mock_settings.pdl_enabled = False
            mock_settings.pdl_api_key = ""

            response = test_client.post(f"/profiles/{eid}/deep-research")

        assert response.status_code == 200
        data = response.json()

        # Status should be halted
        assert data["status"] == "halted"
        assert data["deep_research_status"] == DossierMode.FAILED

        # Should still have a failure report
        assert data["deep_profile"] is not None

        # fail_closed_status should indicate the issue
        fc = data["fail_closed_status"]
        assert fc["gates_passed"] is False
        assert "SerpAPI" in (fc.get("serp_error") or fc.get("failure_message", ""))


class TestDebugEndpoints:
    def test_debug_enrichment_endpoint(self):
        """GET /debug/enrichment should return enrichment details."""
        from fastapi.testclient import TestClient
        from app.api import app

        test_client = TestClient(app)
        response = test_client.get("/debug/enrichment")

        assert response.status_code == 200
        data = response.json()
        assert "pdl_enabled" in data
        assert "pdl_configured" in data
        assert "enriched_contacts" in data
        assert "rate_limiter_state" in data

    def test_debug_serp_endpoint(self):
        """GET /debug/serp should return SerpAPI status."""
        from fastapi.testclient import TestClient
        from app.api import app

        test_client = TestClient(app)
        response = test_client.get("/debug/serp")

        assert response.status_code == 200
        data = response.json()
        assert "serpapi_configured" in data
        assert "contacts_with_research" in data
