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
    audit_visibility_sweep,
    check_strict_coverage,
    compute_gate_status,
    generate_dossier_qa_report,
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


# ---------------------------------------------------------------------------
# Public Visibility Sweep Audit
# ---------------------------------------------------------------------------


class TestVisibilitySweepAudit:
    def test_full_sweep_passes(self):
        """All 10 categories searched should pass."""
        categories = [
            "ted", "tedx", "keynote", "conference", "summit",
            "podcast", "webinar", "youtube_talk", "panel", "interview_video",
        ]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert result.passes
        assert result.ted_tedx_searched
        assert result.podcast_webinar_searched
        assert result.conference_keynote_searched
        assert len(result.hard_failures) == 0

    def test_empty_sweep_fails(self):
        """No categories searched should fail all checks."""
        result = audit_visibility_sweep([], sweep_executed=False)
        assert not result.passes
        assert not result.ted_tedx_searched
        assert not result.podcast_webinar_searched
        assert not result.conference_keynote_searched
        assert len(result.hard_failures) == 4  # not executed + 3 group failures

    def test_missing_ted_tedx_fails(self):
        """Missing TED/TEDx sweep is a hard failure."""
        categories = ["keynote", "conference", "podcast", "webinar"]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert not result.passes
        assert not result.ted_tedx_searched
        assert "TED/TEDx" in result.hard_failures[0]

    def test_missing_podcast_webinar_fails(self):
        """Missing podcast/webinar sweep is a hard failure."""
        categories = ["ted", "tedx", "keynote", "conference"]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert not result.passes
        assert not result.podcast_webinar_searched
        assert any("podcast" in f.lower() for f in result.hard_failures)

    def test_missing_conference_keynote_fails(self):
        """Missing conference/keynote sweep is a hard failure."""
        categories = ["ted", "tedx", "podcast", "webinar"]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert not result.passes
        assert not result.conference_keynote_searched
        assert any("conference" in f.lower() for f in result.hard_failures)

    def test_partial_groups_pass(self):
        """Having at least one from each group should pass."""
        categories = ["ted", "podcast", "keynote"]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert result.passes
        assert result.ted_tedx_searched
        assert result.podcast_webinar_searched
        assert result.conference_keynote_searched

    def test_sweep_not_executed_always_fails(self):
        """Even with categories, sweep_executed=False adds a failure."""
        categories = ["ted", "podcast", "keynote"]
        result = audit_visibility_sweep(categories, sweep_executed=False)
        assert not result.passes
        assert "not executed at all" in result.hard_failures[0].lower()

    def test_summit_counts_for_conference_group(self):
        """Summit is in the conference/keynote group."""
        categories = ["ted", "podcast", "summit"]
        result = audit_visibility_sweep(categories, sweep_executed=True)
        assert result.conference_keynote_searched
        assert result.passes


class TestDossierQaWithVisibility:
    def test_qa_report_flags_missing_sweep(self):
        """QA report should flag when visibility sweep is not executed."""
        text = "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
        report = generate_dossier_qa_report(
            dossier_text=text,
            person_name="Ben",
            visibility_categories=[],
            visibility_sweep_executed=False,
        )
        visibility_flags = [
            f for f in report.hallucination_risk_flags if "VISIBILITY" in f
        ]
        assert len(visibility_flags) > 0

    def test_qa_report_no_flags_with_full_sweep(self):
        """QA report should not flag visibility when sweep is complete."""
        text = "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
        categories = [
            "ted", "tedx", "keynote", "conference", "summit",
            "podcast", "webinar", "youtube_talk", "panel", "interview_video",
        ]
        report = generate_dossier_qa_report(
            dossier_text=text,
            person_name="Ben",
            visibility_categories=categories,
            visibility_sweep_executed=True,
        )
        visibility_flags = [
            f for f in report.hallucination_risk_flags if "VISIBILITY" in f
        ]
        assert len(visibility_flags) == 0

    def test_qa_report_without_visibility_param(self):
        """QA report without visibility params should not flag visibility."""
        text = "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
        report = generate_dossier_qa_report(
            dossier_text=text,
            person_name="Ben",
        )
        visibility_flags = [
            f for f in report.hallucination_risk_flags if "VISIBILITY" in f
        ]
        assert len(visibility_flags) == 0


# ---------------------------------------------------------------------------
# New model tests: PublicVisibilityReport, DealProbabilityScore, InfluenceStrategy
# ---------------------------------------------------------------------------


class TestDossierModels:
    def test_public_visibility_report(self):
        from app.models import PublicVisibilityReport, VisibilityEntry
        report = PublicVisibilityReport(
            sweep_executed=True,
            categories_searched=["ted", "tedx", "podcast"],
            entries=[
                VisibilityEntry(
                    category="ted",
                    title="Talk at TED2025",
                    url="https://ted.com/talks/test",
                    date="2025-06-15",
                ),
            ],
            total_results=3,
            ted_tedx_found=True,
            podcast_webinar_found=True,
        )
        assert report.sweep_executed
        assert report.ted_tedx_found
        assert len(report.entries) == 1
        assert report.entries[0].category == "ted"

    def test_deal_probability_score(self):
        from app.models import DealProbabilityFactor, DealProbabilityScore
        score = DealProbabilityScore(
            total_score=67.0,
            factors=[
                DealProbabilityFactor(
                    factor="Incentive alignment",
                    weight_range="0-20",
                    score=15.0,
                    reasoning="Strong alignment with revenue targets",
                ),
                DealProbabilityFactor(
                    factor="Authority scope",
                    weight_range="0-15",
                    score=12.0,
                    reasoning="CTO with budget authority",
                ),
                DealProbabilityFactor(
                    factor="Political friction risk",
                    weight_range="-0-15",
                    score=-5.0,
                    reasoning="Competing initiative from VP Eng",
                ),
            ],
            positive_total=42.0,
            negative_total=-5.0,
            confidence_level="medium",
        )
        assert score.total_score == 67.0
        assert len(score.factors) == 3
        assert score.confidence_level == "medium"
        assert score.positive_total == 42.0
        assert score.negative_total == -5.0

    def test_influence_strategy(self):
        from app.models import InfluenceStrategy
        strategy = InfluenceStrategy(
            primary_leverage="Revenue growth alignment",
            secondary_leverage="Competitive threat framing",
            message_framing="Growth opportunity, not cost",
            psychological_tempo="Consultative slow build",
            pressure_points=["Q1 target deadline", "Board review in March"],
            avoidance_points=["Don't challenge team size", "Avoid pricing first"],
            early_warning_signs=["Asks for more stakeholders", "Delays follow-up"],
        )
        assert strategy.primary_leverage is not None
        assert len(strategy.pressure_points) == 2
        assert len(strategy.avoidance_points) == 2
        assert len(strategy.early_warning_signs) == 2

    def test_brief_has_new_dossier_fields(self):
        from app.models import BriefOutput, HeaderSection
        brief = BriefOutput(header=HeaderSection())
        assert brief.public_visibility.sweep_executed is False
        assert brief.deal_probability.total_score == 0.0
        assert brief.influence_strategy.primary_leverage is None

    def test_dossier_models_serializable(self):
        import json
        from app.models import (
            BriefOutput,
            DealProbabilityFactor,
            DealProbabilityScore,
            HeaderSection,
            InfluenceStrategy,
            PublicVisibilityReport,
            VisibilityEntry,
        )
        brief = BriefOutput(
            header=HeaderSection(person="Test"),
            public_visibility=PublicVisibilityReport(
                sweep_executed=True,
                categories_searched=["ted", "podcast"],
                entries=[
                    VisibilityEntry(category="ted", title="TED Talk"),
                ],
                total_results=5,
                ted_tedx_found=True,
            ),
            deal_probability=DealProbabilityScore(
                total_score=72.0,
                factors=[
                    DealProbabilityFactor(
                        factor="Incentive alignment",
                        weight_range="0-20",
                        score=18.0,
                    ),
                ],
                positive_total=72.0,
                confidence_level="high",
            ),
            influence_strategy=InfluenceStrategy(
                primary_leverage="Revenue growth",
                pressure_points=["Q1 deadline"],
            ),
        )
        json_str = brief.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["public_visibility"]["sweep_executed"] is True
        assert parsed["deal_probability"]["total_score"] == 72.0
        assert parsed["influence_strategy"]["primary_leverage"] == "Revenue growth"


# ---------------------------------------------------------------------------
# SerpAPI visibility sweep functions
# ---------------------------------------------------------------------------


class TestSerpAPIVisibility:
    def test_visibility_categories_list(self):
        from app.clients.serpapi import VISIBILITY_CATEGORIES
        assert len(VISIBILITY_CATEGORIES) == 10
        assert "ted" in VISIBILITY_CATEGORIES
        assert "tedx" in VISIBILITY_CATEGORIES
        assert "keynote" in VISIBILITY_CATEGORIES
        assert "podcast" in VISIBILITY_CATEGORIES
        assert "youtube_talk" in VISIBILITY_CATEGORIES

    def test_format_visibility_results_empty(self):
        from app.clients.serpapi import format_visibility_results_for_prompt
        result = format_visibility_results_for_prompt({
            "ted": [], "tedx": [], "keynote": [], "conference": [],
            "summit": [], "podcast": [], "webinar": [],
            "youtube_talk": [], "panel": [], "interview_video": [],
        })
        assert "PUBLIC VISIBILITY SWEEP" in result
        assert "Total visibility artifacts found:** 0" in result

    def test_format_visibility_results_with_data(self):
        from app.clients.serpapi import format_visibility_results_for_prompt
        results = {
            "ted": [{"title": "TED Talk by Ben", "link": "https://ted.com/test",
                      "snippet": "Great talk", "tier": 1, "date": "2025-01"}],
            "tedx": [],
            "keynote": [],
            "conference": [],
            "summit": [],
            "podcast": [{"title": "Podcast with Ben", "link": "https://podcast.com/test",
                         "snippet": "Discussion", "tier": 2, "date": ""}],
            "webinar": [],
            "youtube_talk": [],
            "panel": [],
            "interview_video": [],
        }
        result = format_visibility_results_for_prompt(results)
        assert "TED Talk by Ben" in result
        assert "Podcast with Ben" in result
        assert "Total visibility artifacts found:** 2" in result

    def test_search_plan_includes_visibility(self):
        from app.clients.serpapi import generate_search_plan
        plan = generate_search_plan(name="Ben Titmus", company="Acme Corp")
        visibility_entries = [p for p in plan if p["category"].startswith("visibility_")]
        assert len(visibility_entries) == 10
        categories = {e["category"] for e in visibility_entries}
        assert "visibility_ted" in categories
        assert "visibility_tedx" in categories
        assert "visibility_podcast" in categories
        assert "visibility_keynote" in categories

    def test_search_plan_visibility_has_rationale(self):
        from app.clients.serpapi import generate_search_plan
        plan = generate_search_plan(name="Test Person")
        visibility_entries = [p for p in plan if p["category"].startswith("visibility_")]
        for entry in visibility_entries:
            assert "query" in entry
            assert "rationale" in entry
            assert len(entry["rationale"]) > 10


# ---------------------------------------------------------------------------
# Profiler 11-section prompt tests
# ---------------------------------------------------------------------------


class TestProfiler11Sections:
    def test_prompt_has_11_sections(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "### 1." in USER_PROMPT_TEMPLATE
        assert "### 2." in USER_PROMPT_TEMPLATE
        assert "### 3." in USER_PROMPT_TEMPLATE
        assert "### 4." in USER_PROMPT_TEMPLATE
        assert "### 5." in USER_PROMPT_TEMPLATE
        assert "### 6." in USER_PROMPT_TEMPLATE
        assert "### 7." in USER_PROMPT_TEMPLATE
        assert "### 8." in USER_PROMPT_TEMPLATE
        assert "### 9." in USER_PROMPT_TEMPLATE
        assert "### 10." in USER_PROMPT_TEMPLATE
        assert "### 11." in USER_PROMPT_TEMPLATE

    def test_prompt_has_public_visibility_section(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "Public Visibility Report" in USER_PROMPT_TEMPLATE

    def test_prompt_has_deal_probability_section(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "Deal Probability Score" in USER_PROMPT_TEMPLATE
        assert "Incentive alignment" in USER_PROMPT_TEMPLATE
        assert "Authority scope" in USER_PROMPT_TEMPLATE
        assert "Political friction" in USER_PROMPT_TEMPLATE

    def test_prompt_has_influence_strategy_section(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "Influence Strategy Recommendation" in USER_PROMPT_TEMPLATE
        assert "Primary leverage" in USER_PROMPT_TEMPLATE
        assert "Psychological tempo" in USER_PROMPT_TEMPLATE
        assert "Early warning signs" in USER_PROMPT_TEMPLATE

    def test_prompt_has_visibility_research_placeholder(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "{visibility_research}" in USER_PROMPT_TEMPLATE

    def test_prompt_requires_all_11_sections(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "ALL 11 sections" in USER_PROMPT_TEMPLATE

    def test_prompt_qa_section_checks_visibility(self):
        from app.brief.profiler import USER_PROMPT_TEMPLATE
        assert "Public Visibility Sweep" in USER_PROMPT_TEMPLATE
        assert "10 categories" in USER_PROMPT_TEMPLATE

    def test_profiler_accepts_visibility_research(self):
        """generate_deep_profile should accept visibility_research param."""
        from unittest.mock import MagicMock, patch
        with patch("app.brief.profiler.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = "# Dossier"
            MockLLM.return_value = mock_instance

            from app.brief.profiler import generate_deep_profile
            generate_deep_profile(
                name="Test",
                visibility_research="## Visibility sweep results here",
            )
            user_prompt = mock_instance.chat.call_args[0][1]
            assert "Visibility sweep results here" in user_prompt

    def test_profiler_default_visibility_message(self):
        """Without visibility_research, default message should appear."""
        from unittest.mock import MagicMock, patch
        with patch("app.brief.profiler.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            mock_instance.chat.return_value = "# Dossier"
            MockLLM.return_value = mock_instance

            from app.brief.profiler import generate_deep_profile
            generate_deep_profile(name="Test")
            user_prompt = mock_instance.chat.call_args[0][1]
            assert "No visibility sweep was executed" in user_prompt
