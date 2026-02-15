"""Tests for artifact-first dossier pipeline (Workstream B).

Covers:
- Evidence graph building from PDF + meetings
- Artifact-first dossier generation (template)
- Coverage checking
- Dual-path: PDF artifacts enriching deep-profile
- Regression: SerpAPI failure → artifact-first still works
"""

from __future__ import annotations

from app.services.artifact_dossier import (
    build_artifact_dossier,
    build_artifact_evidence_graph,
    check_artifact_coverage,
    run_artifact_dossier_pipeline,
)


# ---------------------------------------------------------------------------
# Evidence graph from profile data
# ---------------------------------------------------------------------------


class TestBuildArtifactEvidenceGraph:
    def test_pdf_only_profile(self):
        """Graph built from PDF text only (no meetings)."""
        profile_data = {
            "linkedin_pdf_raw_text": "Jane Doe\nVP Engineering\nAbout\nExperienced leader.",
            "headline": "VP Engineering",
            "location": "San Francisco",
            "linkedin_pdf_sections": {
                "about": "Experienced technology leader.",
                "header": "Jane Doe\nVP Engineering",
            },
            "linkedin_pdf_experience": [
                {"title": "VP Engineering", "company": "BigCorp", "dates": "2020 - Present"},
            ],
            "linkedin_pdf_education": [{"school": "MIT", "details": "BSc CS"}],
            "linkedin_pdf_skills": ["Python", "Leadership"],
        }

        graph = build_artifact_evidence_graph(profile_data, person_name="Jane Doe")
        assert len(graph.nodes) >= 3  # headline + about + experience + edu + skills

        # All PDF nodes
        pdf_nodes = [n for n in graph.nodes.values() if n.type == "PDF"]
        assert len(pdf_nodes) >= 3

        # No meeting nodes
        meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]
        assert len(meeting_nodes) == 0

    def test_pdf_and_meetings(self):
        """Graph built from PDF + meeting interactions."""
        profile_data = {
            "linkedin_pdf_raw_text": "John Smith\nCTO\nAbout\nBuilding platforms.",
            "headline": "CTO",
            "linkedin_pdf_sections": {"about": "Building platforms."},
            "interactions": [
                {
                    "title": "Q1 Review",
                    "summary": "Discussed Q1 roadmap",
                    "date": "2026-01-15",
                    "type": "meeting",
                },
                {
                    "title": "Follow-up",
                    "summary": "Budget approval pending",
                    "date": "2026-01-20",
                    "type": "email",
                },
            ],
        }

        graph = build_artifact_evidence_graph(profile_data, person_name="John Smith")
        pdf_nodes = [n for n in graph.nodes.values() if n.type == "PDF"]
        meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]

        assert len(pdf_nodes) >= 1
        assert len(meeting_nodes) == 2

    def test_empty_profile(self):
        """Empty profile generates empty graph."""
        graph = build_artifact_evidence_graph({}, person_name="Nobody")
        assert len(graph.nodes) == 0

    def test_no_pdf_meetings_only(self):
        """Profile with meetings but no PDF."""
        profile_data = {
            "interactions": [
                {"title": "Meeting", "summary": "Discussed deal", "date": "2026-02-01"},
            ],
        }
        graph = build_artifact_evidence_graph(profile_data, person_name="Test")
        assert len(graph.nodes) == 1
        assert list(graph.nodes.values())[0].type == "MEETING"


# ---------------------------------------------------------------------------
# Coverage checking
# ---------------------------------------------------------------------------


class TestCheckArtifactCoverage:
    def test_sufficient_pdf_nodes_passes(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        for i in range(5):
            graph.add_pdf_node(source="test", snippet=f"Evidence {i}", ref=f"ref:{i}")

        passes, pct = check_artifact_coverage(graph)
        assert passes is True
        assert pct == 100.0

    def test_zero_nodes_fails(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        passes, pct = check_artifact_coverage(graph)
        assert passes is False

    def test_one_node_partial(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_pdf_node(source="test", snippet="Single evidence")
        passes, pct = check_artifact_coverage(graph)
        assert passes is True  # 1 node > 0
        assert pct > 0


# ---------------------------------------------------------------------------
# Template-based dossier generation
# ---------------------------------------------------------------------------


class TestBuildArtifactDossier:
    def test_generates_all_sections(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_pdf_node(
            source="linkedin_pdf:Test Person",
            snippet="VP of Engineering at TestCo",
            ref="headline",
        )
        graph.add_pdf_node(
            source="linkedin_pdf:Test Person",
            snippet="Experienced leader building cloud platforms",
            ref="about:1",
        )

        profile_data = {
            "company": "TestCo",
            "title": "VP of Engineering",
            "headline": "VP of Engineering at TestCo",
            "location": "New York",
            "linkedin_pdf_sections": {"about": "Experienced leader building cloud platforms."},
            "linkedin_pdf_experience": [
                {"title": "VP Engineering", "company": "TestCo", "dates": "2020 - Present"},
            ],
            "linkedin_pdf_education": [{"school": "Columbia", "details": "MBA"}],
            "linkedin_pdf_skills": ["Strategy", "Engineering", "Cloud"],
        }

        dossier = build_artifact_dossier("Test Person", graph, profile_data)

        # Must contain all 8 sections
        assert "## 1. Executive Summary" in dossier
        assert "## 2. Identity & Background" in dossier
        assert "## 3. Career Timeline" in dossier
        assert "## 4. Topic Position Map" in dossier
        assert "## 5. Rhetorical Patterns" in dossier
        assert "## 6. Gaps & Risks" in dossier
        assert "## 7. Interview Questions" in dossier
        assert "## 8. Primary-Source Index" in dossier

    def test_includes_person_name(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        dossier = build_artifact_dossier("Una Fox", graph, {"title": "CEO", "company": "Fox Inc"})
        assert "Una Fox" in dossier

    def test_tags_claims_with_verified_pdf(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        profile_data = {
            "title": "Director",
            "company": "TestCo",
            "headline": "Director of Sales at TestCo",
            "linkedin_pdf_sections": {"about": "Sales leader driving growth."},
        }

        dossier = build_artifact_dossier("Test Person", graph, profile_data)
        assert "[VERIFIED-PDF]" in dossier

    def test_empty_profile_shows_unknown(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        dossier = build_artifact_dossier("Unknown Person", graph, {})
        assert "[UNKNOWN]" in dossier

    def test_meeting_nodes_tagged_verified_meeting(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        graph.add_meeting_node(
            source="Q1 Review",
            snippet="Discussed revenue targets for 2026",
            date="2026-01-15",
        )

        dossier = build_artifact_dossier("Test", graph, {"interactions": [{}]})
        assert "[VERIFIED-MEETING]" in dossier

    def test_mode_label_artifact_first(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        dossier = build_artifact_dossier("Test", graph, {})
        assert "Artifact-First" in dossier

    def test_gaps_include_web_not_searched(self):
        from app.brief.evidence_graph import EvidenceGraph

        graph = EvidenceGraph()
        dossier = build_artifact_dossier("Test", graph, {})
        assert "No public visibility sweep" in dossier


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestRunArtifactDossierPipeline:
    def test_basic_pipeline(self):
        profile_data = {
            "linkedin_pdf_raw_text": "Jane Doe\nVP Eng\nAbout\nBuilding things.",
            "headline": "VP Eng",
            "linkedin_pdf_sections": {"about": "Building things."},
        }

        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name="Jane Doe",
            use_llm=False,  # Template mode
        )

        assert result["dossier_markdown"]
        assert result["mode"] == "artifact_first"
        assert result["artifact_count"] >= 1
        assert result["generated_at"]
        assert result["evidence_graph"]
        assert "nodes" in result["evidence_graph"]

    def test_empty_profile_still_generates(self):
        result = run_artifact_dossier_pipeline(
            profile_data={},
            person_name="Nobody",
            use_llm=False,
        )
        assert result["dossier_markdown"]  # Still produces something
        assert result["artifact_count"] == 0
        assert result["meeting_count"] == 0

    def test_with_meetings_only(self):
        profile_data = {
            "interactions": [
                {"title": "Deal Review", "summary": "Discussed pricing", "date": "2026-02-01"},
            ],
        }
        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name="Test",
            use_llm=False,
        )
        assert result["meeting_count"] == 1
        assert result["artifact_count"] == 0

    def test_llm_mode_with_no_api_key(self):
        """LLM mode falls back to template when LLM unavailable."""
        profile_data = {
            "linkedin_pdf_raw_text": "Test Person\nCTO\nAbout\nTech leader.",
            "linkedin_pdf_sections": {"about": "Tech leader."},
        }

        # use_llm=True but LLM will fail (no API key in tests)
        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name="Test",
            use_llm=True,  # Will fail gracefully
        )
        assert result["dossier_markdown"]  # Fallback to template


# ---------------------------------------------------------------------------
# Regression: SerpAPI failure → artifact-first still works
# ---------------------------------------------------------------------------


class TestArtifactFirstWithoutSerpAPI:
    """Validates that artifact-first dossier works independently of SerpAPI."""

    def test_dossier_generation_no_web(self):
        """Full dossier generated from PDF alone, no web search."""
        profile_data = {
            "linkedin_pdf_raw_text": "Ben Titmus\nCTO at DataFlow\nAbout\nBuilding data pipelines.",
            "headline": "CTO at DataFlow",
            "company": "DataFlow",
            "title": "CTO",
            "linkedin_pdf_sections": {"about": "Building data pipelines."},
            "linkedin_pdf_experience": [
                {"title": "CTO", "company": "DataFlow", "dates": "2021 - Present"},
                {"title": "VP Eng", "company": "PrevCo", "dates": "2018 - 2021"},
            ],
        }

        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name="Ben Titmus",
            use_llm=False,
        )

        assert result["dossier_markdown"]
        assert "Ben Titmus" in result["dossier_markdown"]
        assert "DataFlow" in result["dossier_markdown"]
        assert result["artifact_count"] >= 2
        assert "[VERIFIED-PDF]" in result["dossier_markdown"]

    def test_pdf_claims_not_tagged_as_public(self):
        """PDF evidence must be tagged VERIFIED-PDF, never VERIFIED-PUBLIC."""
        profile_data = {
            "linkedin_pdf_raw_text": "Test\nCEO\nAbout\nLeading innovation.",
            "linkedin_pdf_sections": {"about": "Leading innovation."},
            "company": "TestCo",
            "title": "CEO",
            "headline": "CEO at TestCo",
        }

        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name="Test",
            use_llm=False,
        )

        # Template dossier should NOT contain VERIFIED-PUBLIC (no web search)
        dossier = result["dossier_markdown"]
        assert "[VERIFIED-PDF]" in dossier
        assert "[VERIFIED-PUBLIC]" not in dossier
