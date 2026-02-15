"""Tests for citation enforcement, evidence discipline, and the no-hallucination guarantee.

These tests verify that:
1. Every claim-bearing section in a brief carries citations
2. When there's no evidence, the brief outputs "Unknown" rather than fabricating data
3. Citation objects are well-formed
4. Evidence tags are correctly parsed and applied
5. Strategic Operating Model sections are properly structured
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

from app.brief.generator import (
    _compute_snippet_hash,
    _parse_citation,
    _parse_citations,
    _parse_evidence_tag,
    _parse_tagged_claim,
    _parse_tagged_claims,
    generate_brief,
)
from app.models import (
    Citation,
    EvidenceTag,
    SourceType,
    TaggedClaim,
)
from app.retrieve.retriever import RetrievedEvidence


class TestSnippetHash:
    """Test the snippet hash function."""

    def test_compute_snippet_hash_deterministic(self):
        h1 = _compute_snippet_hash("Hello world")
        h2 = _compute_snippet_hash("Hello world")
        assert h1 == h2

    def test_compute_snippet_hash_different_inputs(self):
        h1 = _compute_snippet_hash("Hello")
        h2 = _compute_snippet_hash("World")
        assert h1 != h2

    def test_compute_snippet_hash_format(self):
        h = _compute_snippet_hash("test")
        assert len(h) == 16  # Truncated SHA-256
        assert all(c in "0123456789abcdef" for c in h)


class TestCitationParsing:
    """Test citation parsing from LLM output."""

    def test_parse_citation_basic(self):
        raw = {
            "source_type": "fireflies",
            "source_id": "ff-001",
            "timestamp": "2026-01-15T10:00:00",
            "excerpt": "We discussed the timeline",
            "snippet_hash": "abc123",
            "link": None,
        }
        citation = _parse_citation(raw)

        assert citation.source_type == SourceType.fireflies
        assert citation.source_id == "ff-001"
        assert citation.excerpt == "We discussed the timeline"

    def test_parse_citation_generates_hash_if_missing(self):
        raw = {
            "source_type": "gmail",
            "source_id": "msg-001",
            "timestamp": "2026-01-15T10:00:00",
            "excerpt": "Some excerpt text",
        }
        citation = _parse_citation(raw)
        assert citation.snippet_hash == _compute_snippet_hash("Some excerpt text")

    def test_parse_citations_empty(self):
        assert _parse_citations(None) == []
        assert _parse_citations([]) == []


class TestNoEvidenceBrief:
    """Test that the brief correctly handles NO evidence.

    This is the critical "no-hallucination" test: when we have zero
    source records, the brief must NOT fabricate data.
    """

    def test_no_evidence_produces_zero_confidence(self):
        evidence = RetrievedEvidence()
        assert not evidence.has_data

        brief = generate_brief(
            person="Ghost Person",
            company="Phantom Corp",
            topic="Imaginary Meeting",
            meeting_datetime=datetime(2026, 3, 1, 14, 0),
            evidence=evidence,
        )

        assert brief.header.confidence_score == 0.0

    def test_no_evidence_produces_unknown_markers(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        # Relationship context should indicate unknown
        assert "Unknown" in (brief.relationship_context.role or "Unknown")

        # Last interaction should be None
        assert brief.last_interaction is None

        # Open loops should be empty
        assert len(brief.open_loops) == 0

        # Watchouts should be empty
        assert len(brief.watchouts) == 0

    def test_no_evidence_no_fabricated_interactions(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        assert len(brief.interaction_history) == 0
        assert len(brief.appendix_evidence) == 0

    def test_no_evidence_has_empty_sources(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        assert brief.header.data_sources_used == []


class TestCitationEnforcement:
    """Test that citations are properly enforced in generated briefs."""

    def _make_evidence_with_data(self) -> RetrievedEvidence:
        """Create evidence with some mock data for testing."""
        evidence = RetrievedEvidence()
        evidence.interactions = [
            {
                "source_type": "fireflies",
                "source_id": "ff-test-001",
                "title": "Test Meeting",
                "date": "2026-01-15T10:00:00",
                "summary": "Discussed project timeline and budget concerns.",
                "participants": ["me@test.com", "them@test.com"],
                "action_items": ["Send proposal by Friday"],
                "body_preview": "Discussed project timeline. There is concern about budget.",
                "db_id": 1,
            }
        ]
        evidence.last_interaction = evidence.interactions[0]
        evidence.action_items = [
            {
                "description": "Send proposal by Friday",
                "source_type": "fireflies",
                "source_id": "ff-test-001",
                "date": "2026-01-15T10:00:00",
            }
        ]
        evidence.concern_snippets = [
            {
                "keyword": "concern",
                "snippet": "There is concern about budget",
                "source_type": "fireflies",
                "source_id": "ff-test-001",
                "date": "2026-01-15T10:00:00",
            }
        ]

        # Create a mock source record
        mock_record = MagicMock()
        mock_record.id = 1
        mock_record.source_type = "fireflies"
        mock_record.source_id = "ff-test-001"
        mock_record.title = "Test Meeting"
        mock_record.date = datetime(2026, 1, 15, 10, 0)
        mock_record.link = None
        mock_record.body = "Discussed project timeline. There is concern about budget."
        evidence.all_source_records = [mock_record]

        return evidence

    def test_fallback_brief_has_citations_on_last_interaction(self):
        """When LLM is unavailable, fallback brief should still cite sources."""
        from app.brief.generator import _build_fallback_brief
        from app.models import HeaderSection

        evidence = self._make_evidence_with_data()
        header = HeaderSection(person="Test", confidence_score=0.0)

        brief = _build_fallback_brief(header, evidence)

        assert brief.last_interaction is not None
        assert len(brief.last_interaction.citations) > 0
        assert brief.last_interaction.citations[0].source_id == "ff-test-001"

    def test_fallback_brief_has_citations_on_open_loops(self):
        from app.brief.generator import _build_fallback_brief
        from app.models import HeaderSection

        evidence = self._make_evidence_with_data()
        header = HeaderSection(person="Test", confidence_score=0.0)

        brief = _build_fallback_brief(header, evidence)

        assert len(brief.open_loops) > 0
        for loop in brief.open_loops:
            assert len(loop.citations) > 0
            assert loop.citations[0].source_type in (SourceType.fireflies, SourceType.gmail)

    def test_citation_has_required_fields(self):
        """Every Citation must have source_type, source_id, timestamp, excerpt, snippet_hash."""
        citation = Citation(
            source_type=SourceType.fireflies,
            source_id="ff-001",
            timestamp=datetime(2026, 1, 15),
            excerpt="Test excerpt",
            snippet_hash=_compute_snippet_hash("Test excerpt"),
        )

        assert citation.source_type == SourceType.fireflies
        assert citation.source_id == "ff-001"
        assert citation.timestamp is not None
        assert citation.excerpt == "Test excerpt"
        assert len(citation.snippet_hash) > 0

    def test_brief_output_serializable(self):
        """BriefOutput must be JSON-serializable."""
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        json_str = brief.model_dump_json()
        assert json_str is not None
        parsed = json.loads(json_str)
        assert "header" in parsed
        assert "relationship_context" in parsed
        assert "open_loops" in parsed

    def test_brief_output_has_strategic_sections(self):
        """BriefOutput must include all Strategic Operating Model sections."""
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        json_str = brief.model_dump_json()
        parsed = json.loads(json_str)
        assert "strategic_positioning" in parsed
        assert "power_map" in parsed
        assert "incentive_structure" in parsed
        assert "cognitive_patterns" in parsed
        assert "strategic_tensions" in parsed
        assert "behavioral_forecasts" in parsed
        assert "information_gaps" in parsed
        assert "conversation_strategy" in parsed
        assert "meeting_delta" in parsed
        assert "engine_improvements" in parsed

    def test_fallback_brief_has_information_gaps(self):
        """Fallback brief should flag gaps and improvement recommendations."""
        from app.brief.generator import _build_fallback_brief
        from app.models import HeaderSection

        evidence = self._make_evidence_with_data()
        header = HeaderSection(person="Test", confidence_score=0.0)

        brief = _build_fallback_brief(header, evidence)

        assert len(brief.information_gaps) > 0
        assert brief.information_gaps[0].gap != ""
        assert brief.information_gaps[0].strategic_impact != ""

    def test_fallback_brief_has_engine_improvements(self):
        """Fallback brief should include engine improvement recommendations."""
        from app.brief.generator import _build_fallback_brief
        from app.models import HeaderSection

        evidence = self._make_evidence_with_data()
        header = HeaderSection(person="Test", confidence_score=0.0)

        brief = _build_fallback_brief(header, evidence)

        assert len(brief.engine_improvements.missing_signals) > 0
        assert len(brief.engine_improvements.recommended_data_sources) > 0


class TestEvidenceTagParsing:
    """Test evidence discipline tag parsing."""

    def test_parse_verified_meeting(self):
        assert _parse_evidence_tag("VERIFIED_MEETING") == EvidenceTag.verified_meeting

    def test_parse_verified_public(self):
        assert _parse_evidence_tag("VERIFIED_PUBLIC") == EvidenceTag.verified_public

    def test_parse_inferred_high(self):
        assert _parse_evidence_tag("INFERRED_HIGH") == EvidenceTag.inferred_high

    def test_parse_inferred_low(self):
        assert _parse_evidence_tag("INFERRED_LOW") == EvidenceTag.inferred_low

    def test_parse_unknown(self):
        assert _parse_evidence_tag("UNKNOWN") == EvidenceTag.unknown

    def test_parse_none_defaults_to_unknown(self):
        assert _parse_evidence_tag(None) == EvidenceTag.unknown

    def test_parse_empty_defaults_to_unknown(self):
        assert _parse_evidence_tag("") == EvidenceTag.unknown

    def test_parse_handles_dashes(self):
        """LLM might use dashes or en-dashes instead of underscores."""
        assert _parse_evidence_tag("VERIFIED-MEETING") == EvidenceTag.verified_meeting
        assert _parse_evidence_tag("INFERRED-HIGH") == EvidenceTag.inferred_high

    def test_parse_handles_spaces(self):
        assert _parse_evidence_tag("VERIFIED MEETING") == EvidenceTag.verified_meeting

    def test_parse_case_insensitive(self):
        assert _parse_evidence_tag("verified_meeting") == EvidenceTag.verified_meeting
        assert _parse_evidence_tag("Inferred_High") == EvidenceTag.inferred_high

    def test_parse_garbage_defaults_to_unknown(self):
        assert _parse_evidence_tag("TOTALLY_INVALID") == EvidenceTag.unknown


class TestTaggedClaimParsing:
    """Test tagged claim parsing from LLM output."""

    def test_parse_tagged_claim_basic(self):
        raw = {
            "claim": "Subject controls Q1 revenue pipeline",
            "evidence_tag": "VERIFIED_MEETING",
            "citations": [],
        }
        claim = _parse_tagged_claim(raw)
        assert claim is not None
        assert claim.claim == "Subject controls Q1 revenue pipeline"
        assert claim.evidence_tag == EvidenceTag.verified_meeting

    def test_parse_tagged_claim_none(self):
        assert _parse_tagged_claim(None) is None

    def test_parse_tagged_claim_empty_dict(self):
        """Empty dict is falsy — treated same as None."""
        assert _parse_tagged_claim({}) is None

    def test_parse_tagged_claim_minimal(self):
        """Dict with at least one key parses with defaults."""
        claim = _parse_tagged_claim({"claim": "Some observation"})
        assert claim is not None
        assert claim.claim == "Some observation"
        assert claim.evidence_tag == EvidenceTag.unknown

    def test_parse_tagged_claim_with_citations(self):
        raw = {
            "claim": "Budget authority up to $500K",
            "evidence_tag": "VERIFIED_MEETING",
            "citations": [
                {
                    "source_type": "fireflies",
                    "source_id": "ff-001",
                    "timestamp": "2026-01-15T10:00:00",
                    "excerpt": "I can approve up to 500K",
                    "snippet_hash": "abc123",
                }
            ],
        }
        claim = _parse_tagged_claim(raw)
        assert claim is not None
        assert len(claim.citations) == 1
        assert claim.citations[0].source_id == "ff-001"

    def test_parse_tagged_claims_list(self):
        raw_list = [
            {"claim": "First claim", "evidence_tag": "VERIFIED_MEETING"},
            {"claim": "Second claim", "evidence_tag": "INFERRED_HIGH"},
        ]
        claims = _parse_tagged_claims(raw_list)
        assert len(claims) == 2
        assert claims[0].claim == "First claim"
        assert claims[1].evidence_tag == EvidenceTag.inferred_high

    def test_parse_tagged_claims_empty(self):
        assert _parse_tagged_claims(None) == []
        assert _parse_tagged_claims([]) == []


class TestNoEvidenceStrategicSections:
    """Test that no-evidence briefs populate strategic sections correctly."""

    def test_no_evidence_has_information_gaps(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        assert len(brief.information_gaps) > 0
        # Gap should be strategically meaningful, not generic
        assert any("interact" in g.gap.lower() or "transcript" in g.gap.lower()
                    for g in brief.information_gaps)

    def test_no_evidence_has_engine_improvements(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        ei = brief.engine_improvements
        assert len(ei.missing_signals) > 0
        assert len(ei.recommended_data_sources) > 0
        assert len(ei.capture_fields) > 0

    def test_no_evidence_empty_strategic_sections(self):
        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Nobody",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        assert len(brief.strategic_positioning) == 0
        assert len(brief.cognitive_patterns) == 0
        assert len(brief.strategic_tensions) == 0
        assert len(brief.behavioral_forecasts) == 0


class TestMarkdownRendering:
    """Test that the renderer produces the Strategic Intelligence format."""

    def test_renders_strategic_title(self):
        from app.brief.renderer import render_markdown

        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        md = render_markdown(brief)
        assert "Strategic Intelligence Brief" in md

    def test_renders_all_strategic_sections(self):
        from app.brief.renderer import render_markdown

        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company="TestCo",
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )

        md = render_markdown(brief)
        assert "Strategic Positioning Snapshot" in md
        assert "Power & Influence Map" in md
        assert "Incentive Structure Analysis" in md
        assert "Cognitive & Rhetorical Patterns" in md
        assert "Strategic Tensions" in md
        assert "Behavioral Forecast" in md
        assert "Information Gaps That Matter" in md
        assert "Executive Conversation Strategy" in md
        assert "Meeting Delta Analysis" in md
        assert "Engine Improvement Recommendations" in md

    def test_renders_evidence_tags(self):
        from app.brief.renderer import _tag

        assert "VERIFIED" in _tag(EvidenceTag.verified_meeting)
        assert "MEETING" in _tag(EvidenceTag.verified_meeting)
        assert "UNKNOWN" in _tag(EvidenceTag.unknown)

    def test_renders_tagged_claims(self):
        from app.brief.renderer import _render_tagged_claim

        claim = TaggedClaim(
            claim="Subject owns P&L for consulting division",
            evidence_tag=EvidenceTag.verified_meeting,
        )
        rendered = _render_tagged_claim(claim)
        assert "VERIFIED" in rendered
        assert "P&L for consulting division" in rendered
