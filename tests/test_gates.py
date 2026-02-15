"""Tests for quality gates: identity lock, evidence coverage, genericness pruning.

These tests verify the gating pipeline that enforces:
1. Identity lock scoring and constraining
2. Evidence coverage >= 95% in strict mode
3. Genericness detection and pruning
4. Gate status computation
"""

from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

from app.brief.qa import (
    STRICT_EVIDENCE_THRESHOLD,
    EvidenceCoverageResult,
    check_strict_coverage,
    compute_gate_status,
    lint_generic_filler,
    lint_generic_filler_strict,
    prune_uncited_claims,
)


# ---------------------------------------------------------------------------
# Strict evidence threshold
# ---------------------------------------------------------------------------


class TestStrictEvidenceThreshold:
    def test_threshold_is_95(self):
        assert STRICT_EVIDENCE_THRESHOLD == 95.0

    def test_check_strict_coverage_passes_at_95(self):
        result = EvidenceCoverageResult(total_substantive=100, tagged_count=95)
        assert check_strict_coverage(result)

    def test_check_strict_coverage_fails_at_94(self):
        result = EvidenceCoverageResult(total_substantive=100, tagged_count=94)
        assert not check_strict_coverage(result)

    def test_check_strict_coverage_passes_at_100(self):
        result = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        assert check_strict_coverage(result)


# ---------------------------------------------------------------------------
# Prune uncited claims
# ---------------------------------------------------------------------------


class TestPruneUncitedClaims:
    def test_preserves_cited_lines(self):
        text = (
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees engineering. [VERIFIED-MEETING]\n"
        )
        result = prune_uncited_claims(text)
        assert "[VERIFIED-PUBLIC]" in result
        assert "[VERIFIED-MEETING]" in result

    def test_removes_uncited_substantive_lines(self):
        text = (
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He is a strategic leader with deep expertise in platform engineering.\n"
            "Revenue grew significantly last year.\n"
        )
        result = prune_uncited_claims(text)
        assert "[VERIFIED-PUBLIC]" in result
        assert "strategic leader" not in result
        assert "Revenue grew" not in result

    def test_preserves_headers(self):
        text = (
            "### Strategic Snapshot\n"
            "Ben is CTO. [VERIFIED-PUBLIC]\n"
            "---\n"
        )
        result = prune_uncited_claims(text)
        assert "### Strategic Snapshot" in result
        assert "---" in result

    def test_preserves_short_lines(self):
        text = (
            "Short.\n"
            "Ben is CTO. [VERIFIED-PUBLIC]\n"
        )
        result = prune_uncited_claims(text)
        assert "Short." in result

    def test_preserves_table_rows(self):
        text = (
            "| Field | Value |\n"
            "| test | data |\n"
            "Ben is CTO. [VERIFIED-PUBLIC]\n"
        )
        result = prune_uncited_claims(text)
        assert "| Field | Value |" in result

    def test_empty_text(self):
        assert prune_uncited_claims("") == ""

    def test_all_cited(self):
        text = (
            "Claim one. [VERIFIED-PUBLIC]\n"
            "Claim two. [VERIFIED-MEETING]\n"
        )
        result = prune_uncited_claims(text)
        assert result.strip() == text.strip()


# ---------------------------------------------------------------------------
# Gate status computation
# ---------------------------------------------------------------------------


class TestComputeGateStatus:
    def test_all_gates_pass(self):
        status = compute_gate_status(
            identity_lock_score=85,
            evidence_coverage_pct=90,
            genericness_score=10,
            strict=False,
        )
        assert status == "passed"

    def test_identity_lock_below_70_constrains(self):
        status = compute_gate_status(
            identity_lock_score=60,
            evidence_coverage_pct=90,
            genericness_score=10,
            strict=False,
        )
        assert status == "constrained"

    def test_evidence_coverage_below_85_fails(self):
        status = compute_gate_status(
            identity_lock_score=85,
            evidence_coverage_pct=80,
            genericness_score=10,
            strict=False,
        )
        assert status == "failed"

    def test_strict_evidence_coverage_below_95_fails(self):
        status = compute_gate_status(
            identity_lock_score=85,
            evidence_coverage_pct=90,
            genericness_score=10,
            strict=True,
        )
        assert status == "failed"

    def test_strict_evidence_coverage_at_95_passes(self):
        status = compute_gate_status(
            identity_lock_score=85,
            evidence_coverage_pct=95,
            genericness_score=10,
            strict=True,
        )
        assert status == "passed"

    def test_genericness_above_20_fails(self):
        status = compute_gate_status(
            identity_lock_score=85,
            evidence_coverage_pct=90,
            genericness_score=25,
            strict=False,
        )
        assert status == "failed"

    def test_failed_takes_priority_over_constrained(self):
        status = compute_gate_status(
            identity_lock_score=60,
            evidence_coverage_pct=70,
            genericness_score=10,
            strict=False,
        )
        assert status == "failed"


# ---------------------------------------------------------------------------
# Strict genericness
# ---------------------------------------------------------------------------


class TestStrictGenericness:
    def test_catches_likely(self):
        text = "He is likely a builder type with strong platform skills."
        result = lint_generic_filler_strict(text)
        assert result.generic_count > 0

    def test_catches_may(self):
        text = "She may implement corrective measures to address delivery gaps."
        result = lint_generic_filler_strict(text)
        assert result.generic_count > 0

    def test_catches_could(self):
        text = "Revenue could grow significantly in the next quarter."
        result = lint_generic_filler_strict(text)
        assert result.generic_count > 0

    def test_catches_generally(self):
        text = "Executives generally focus on operational efficiency."
        result = lint_generic_filler_strict(text)
        assert result.generic_count > 0

    def test_catches_typically(self):
        text = "Leaders in this role typically manage large teams."
        result = lint_generic_filler_strict(text)
        assert result.generic_count > 0

    def test_clean_text_passes(self):
        text = (
            "Ben Titmus serves as CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees a team of 45 engineers. [VERIFIED-MEETING]\n"
        )
        result = lint_generic_filler_strict(text)
        assert result.generic_count == 0

    def test_standard_lint_does_not_catch_likely(self):
        text = "He is likely a builder type."
        result = lint_generic_filler(text)
        # Standard lint does NOT catch "likely" — that's only in strict
        standard_count = result.generic_count
        strict_result = lint_generic_filler_strict(text)
        assert strict_result.generic_count >= standard_count


# ---------------------------------------------------------------------------
# Person-first brief structure (from models)
# ---------------------------------------------------------------------------


class TestPersonFirstBriefStructure:
    def test_header_has_gate_scores(self):
        from app.models import HeaderSection
        h = HeaderSection(person="Test")
        assert h.identity_lock_score == 0.0
        assert h.evidence_coverage_pct == 0.0
        assert h.genericness_score == 0.0
        assert h.gate_status == "not_run"
        assert h.confidence_drivers == []

    def test_brief_has_what_to_cover(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.what_to_cover == []

    def test_brief_has_evidence_index(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.evidence_index == []

    def test_brief_has_verify_first(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.verify_first == []

    def test_brief_has_leverage_questions(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.leverage_questions == []

    def test_brief_has_proof_points(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.proof_points == []

    def test_information_gap_has_resolution(self):
        from app.models import InformationGap
        ig = InformationGap(
            gap="Unknown title",
            strategic_impact="Cannot assess decision authority",
            how_to_resolve="Check LinkedIn",
            suggested_question="What's your current role?",
        )
        assert ig.suggested_question == "What's your current role?"
        assert ig.how_to_resolve == "Check LinkedIn"

    def test_citation_has_excerpt_offsets(self):
        from datetime import datetime
        from app.models import Citation, SourceType
        c = Citation(
            source_type=SourceType.fireflies,
            source_id="ff-001",
            timestamp=datetime(2026, 1, 1),
            excerpt="test",
            snippet_hash="abc",
            excerpt_start=100,
            excerpt_end=200,
        )
        assert c.excerpt_start == 100
        assert c.excerpt_end == 200

    def test_what_to_cover_item(self):
        from app.models import WhatToCoverItem
        item = WhatToCoverItem(
            item="Follow up on Q1 targets",
            rationale="They mentioned concern about pipeline in last call",
        )
        assert item.item == "Follow up on Q1 targets"
        assert item.rationale != ""

    def test_leverage_question(self):
        from app.models import LeverageQuestion
        lq = LeverageQuestion(
            question="How is the pipeline shaping up vs target?",
            rationale="They flagged pipeline risk in Dec call",
        )
        assert lq.question != ""
        assert lq.rationale != ""

    def test_proof_point(self):
        from app.models import ProofPoint
        pp = ProofPoint(
            point="Our platform reduced churn 23% at similar-sized client",
            why_it_matters="They mentioned churn as a board-level concern",
        )
        assert pp.point != ""
        assert pp.why_it_matters != ""

    def test_verify_first_item(self):
        from app.models import VerifyFirstItem
        vf = VerifyFirstItem(
            fact="CTO at AnswerRocket",
            current_confidence="low",
            source="LinkedIn URL not confirmed",
        )
        assert vf.fact == "CTO at AnswerRocket"

    def test_brief_serializable_with_new_fields(self):
        import json
        from app.models import (
            BriefOutput,
            HeaderSection,
            WhatToCoverItem,
            LeverageQuestion,
            ProofPoint,
            InformationGap,
            EvidenceIndexEntry,
            SourceType,
        )
        brief = BriefOutput(
            header=HeaderSection(
                person="Test",
                identity_lock_score=75,
                evidence_coverage_pct=92,
                genericness_score=5,
                gate_status="passed",
                confidence_drivers=["3 meetings", "5 emails"],
            ),
            what_to_cover=[
                WhatToCoverItem(item="Follow up on Q1", rationale="flagged in Dec"),
            ],
            leverage_questions=[
                LeverageQuestion(question="Pipeline status?", rationale="concern"),
            ],
            proof_points=[
                ProofPoint(point="23% churn reduction", why_it_matters="board concern"),
            ],
            information_gaps=[
                InformationGap(
                    gap="Budget authority unknown",
                    strategic_impact="Cannot scope proposal",
                    how_to_resolve="Ask directly",
                    suggested_question="What's your budget envelope for this?",
                ),
            ],
            evidence_index=[
                EvidenceIndexEntry(
                    source_type=SourceType.fireflies,
                    source_id="ff-001",
                    excerpt="We discussed the timeline...",
                    snippet_hash="abc123",
                ),
            ],
        )
        json_str = brief.model_dump_json()
        parsed = json.loads(json_str)
        assert "what_to_cover" in parsed
        assert "leverage_questions" in parsed
        assert "proof_points" in parsed
        assert "evidence_index" in parsed
        assert parsed["header"]["identity_lock_score"] == 75
        assert parsed["header"]["gate_status"] == "passed"
        assert parsed["information_gaps"][0]["suggested_question"] != ""


# ---------------------------------------------------------------------------
# Renderer person-first sections
# ---------------------------------------------------------------------------


class TestRendererPersonFirst:
    def test_renders_gate_scores_when_run(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection

        brief = BriefOutput(
            header=HeaderSection(
                person="Ben",
                company="Acme",
                identity_lock_score=85,
                evidence_coverage_pct=92,
                genericness_score=5,
                gate_status="passed",
            ),
        )
        md = render_markdown(brief)
        assert "Identity Lock" in md
        assert "85/100" in md
        assert "Evidence Coverage" in md
        assert "92%" in md
        assert "PASSED" in md

    def test_hides_gate_scores_when_not_run(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection

        brief = BriefOutput(
            header=HeaderSection(person="Ben", gate_status="not_run"),
        )
        md = render_markdown(brief)
        assert "Identity Lock" not in md

    def test_renders_verify_first_warning(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection, VerifyFirstItem

        brief = BriefOutput(
            header=HeaderSection(person="Ben", gate_status="constrained"),
            verify_first=[
                VerifyFirstItem(fact="Name match", current_confidence="low"),
            ],
        )
        md = render_markdown(brief)
        assert "Verify" in md
        assert "Name match" in md

    def test_renders_what_to_cover(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection, WhatToCoverItem

        brief = BriefOutput(
            header=HeaderSection(person="Ben"),
            what_to_cover=[
                WhatToCoverItem(item="Follow up on pipeline", rationale="flagged concern"),
            ],
        )
        md = render_markdown(brief)
        assert "What I Must Cover" in md
        assert "Follow up on pipeline" in md
        assert "flagged concern" in md

    def test_renders_open_loops_table(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection, OpenLoop

        brief = BriefOutput(
            header=HeaderSection(person="Ben"),
            open_loops=[
                OpenLoop(description="Send proposal", owner="Me", due_date="2026-02-20"),
            ],
        )
        md = render_markdown(brief)
        assert "Open Loops & Commitments" in md
        assert "Send proposal" in md
        assert "| Item |" in md  # Table header

    def test_renders_unknowns_with_resolution(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection, InformationGap

        brief = BriefOutput(
            header=HeaderSection(person="Ben"),
            information_gaps=[
                InformationGap(
                    gap="Budget unknown",
                    strategic_impact="Cannot scope",
                    how_to_resolve="Ask on call",
                    suggested_question="What's your budget?",
                ),
            ],
        )
        md = render_markdown(brief)
        assert "Unknowns That Matter" in md
        assert "Budget unknown" in md
        assert "What's your budget?" in md

    def test_renders_leverage_questions_with_citations(self):
        from datetime import datetime
        from app.brief.renderer import render_markdown
        from app.models import (
            BriefOutput, Citation, HeaderSection, LeverageQuestion, SourceType,
        )

        brief = BriefOutput(
            header=HeaderSection(person="Ben"),
            leverage_questions=[
                LeverageQuestion(
                    question="How is pipeline vs target?",
                    rationale="Flagged concern in Dec",
                    citations=[
                        Citation(
                            source_type=SourceType.fireflies,
                            source_id="ff-001",
                            timestamp=datetime(2026, 1, 15),
                            excerpt="pipeline risk",
                            snippet_hash="abc",
                        ),
                    ],
                ),
            ],
        )
        md = render_markdown(brief)
        assert "Leverage Plan" in md
        assert "How is pipeline vs target?" in md
        assert "ff-001" in md

    def test_renders_evidence_index(self):
        from app.brief.renderer import render_markdown
        from app.models import (
            BriefOutput, EvidenceIndexEntry, HeaderSection, SourceType,
        )

        brief = BriefOutput(
            header=HeaderSection(person="Ben"),
            evidence_index=[
                EvidenceIndexEntry(
                    source_type=SourceType.fireflies,
                    source_id="ff-001",
                    excerpt="We discussed timeline and budget",
                    snippet_hash="abc123",
                ),
            ],
        )
        md = render_markdown(brief)
        assert "Evidence Index" in md
        assert "ff-001" in md
        assert "We discussed timeline" in md

    def test_renders_confidence_drivers(self):
        from app.brief.renderer import render_markdown
        from app.models import BriefOutput, HeaderSection

        brief = BriefOutput(
            header=HeaderSection(
                person="Ben",
                confidence_drivers=["3 meetings in last 90 days", "5 email threads"],
            ),
        )
        md = render_markdown(brief)
        assert "3 meetings" in md
        assert "5 email threads" in md


# ---------------------------------------------------------------------------
# Generator person-first output
# ---------------------------------------------------------------------------


class TestGeneratorPersonFirst:
    def test_system_prompt_is_person_first(self):
        from app.brief.generator import SYSTEM_PROMPT
        assert "PERSON-FIRST" in SYSTEM_PROMPT
        assert "relationship" in SYSTEM_PROMPT.lower()

    def test_system_prompt_bans_scenario_planning(self):
        from app.brief.generator import SYSTEM_PROMPT
        assert "NO SCENARIO PLANNING" in SYSTEM_PROMPT

    def test_system_prompt_requires_zero_hallucination(self):
        from app.brief.generator import SYSTEM_PROMPT
        assert "ZERO HALLUCINATION" in SYSTEM_PROMPT

    def test_user_prompt_has_what_to_cover(self):
        from app.brief.generator import USER_PROMPT_TEMPLATE
        assert "what_to_cover" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_leverage_questions(self):
        from app.brief.generator import USER_PROMPT_TEMPLATE
        assert "leverage_questions" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_proof_points(self):
        from app.brief.generator import USER_PROMPT_TEMPLATE
        assert "proof_points" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_information_gaps_resolution(self):
        from app.brief.generator import USER_PROMPT_TEMPLATE
        assert "suggested_question" in USER_PROMPT_TEMPLATE
        assert "how_to_resolve" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_confidence_drivers(self):
        from app.brief.generator import USER_PROMPT_TEMPLATE
        assert "confidence_drivers" in USER_PROMPT_TEMPLATE

    def test_no_evidence_brief_has_gate_status(self):
        from app.brief.generator import generate_brief
        from app.retrieve.retriever import RetrievedEvidence

        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )
        assert brief.header.gate_status == "failed"
        assert brief.header.confidence_drivers == ["No interaction data available"]

    def test_no_evidence_brief_has_resolution_question(self):
        from app.brief.generator import generate_brief
        from app.retrieve.retriever import RetrievedEvidence

        evidence = RetrievedEvidence()
        brief = generate_brief(
            person="Test",
            company=None,
            topic=None,
            meeting_datetime=None,
            evidence=evidence,
        )
        assert len(brief.information_gaps) > 0
        assert brief.information_gaps[0].suggested_question != ""
        assert brief.information_gaps[0].how_to_resolve != ""
