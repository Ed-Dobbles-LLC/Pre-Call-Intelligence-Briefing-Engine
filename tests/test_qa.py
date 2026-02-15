"""Tests for QA gates: genericness linter, evidence coverage, contradiction detection, disambiguation."""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

from app.brief.qa import (
    Contradiction,
    DisambiguationResult,
    EvidenceCoverageResult,
    GenericFillerResult,
    InferredHAudit,
    PersonLevelResult,
    QAReport,
    SnapshotValidation,
    audit_inferred_h,
    check_evidence_coverage,
    check_person_level_ratio,
    check_snapshot_person_focus,
    detect_contradictions,
    generate_qa_report,
    lint_generic_filler,
    render_qa_report_markdown,
    score_disambiguation,
)


# ---------------------------------------------------------------------------
# Generic Filler Linter
# ---------------------------------------------------------------------------


class TestGenericFillerLinter:
    def test_clean_text_scores_zero(self):
        text = (
            "Ben Titmus serves as CTO at Acme Corp since 2023. [VERIFIED-PUBLIC]\n"
            "He oversees a team of 45 engineers in London. [VERIFIED-MEETING]\n"
            "Revenue grew 32% year-over-year under his leadership. [VERIFIED-PUBLIC]"
        )
        result = lint_generic_filler(text)
        assert result.genericness_score == 0
        assert result.generic_count == 0

    def test_flags_strategic_leader(self):
        text = "He is a strategic leader who drives value across the organization."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1
        assert any("strategic leader" in f["pattern"].lower() for f in result.flagged_sentences)

    def test_flags_passionate_about(self):
        text = "She is passionate about innovation and cutting-edge technology."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1

    def test_flags_proven_track_record(self):
        text = "He has a proven track record of driving results in the enterprise space."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1
        assert any("proven track record" in f["pattern"].lower() for f in result.flagged_sentences)

    def test_skips_generic_with_evidence_tag(self):
        text = 'He described himself as a "strategic leader" in the keynote [VERIFIED-PUBLIC].'
        result = lint_generic_filler(text)
        assert result.generic_count == 0

    def test_skips_short_sentences(self):
        text = "Leader.\nDrives."
        result = lint_generic_filler(text)
        assert result.total_sentences == 0

    def test_skips_markdown_headers(self):
        text = "### Strategic leader and visionary\nActual content here is fine."
        result = lint_generic_filler(text)
        # The header line should be skipped
        assert all(f["sentence"] != "### Strategic leader and visionary" for f in result.flagged_sentences)

    def test_flags_empowers_teams(self):
        text = "She empowers teams to deliver at scale with a holistic approach."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1

    def test_flags_bridges_the_gap(self):
        text = "He bridges the gap between business and technology stakeholders."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1

    def test_flags_at_the_intersection_of(self):
        text = "Operating at the intersection of AI and enterprise sales."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1

    def test_genericness_score_calculation(self):
        text = (
            "A strategic leader in the industry.\n"
            "Passionate about driving innovation.\n"
            "Proven track record of delivery.\n"
            "Has a holistic approach to business.\n"
            "Empowers teams to deliver results."
        )
        result = lint_generic_filler(text)
        assert result.genericness_score > 50

    def test_result_contains_line_numbers(self):
        text = "Line one is fine and normal content.\nHe is a thought leader in AI."
        result = lint_generic_filler(text)
        assert result.flagged_sentences
        assert result.flagged_sentences[0]["line"] == 2

    def test_empty_text(self):
        result = lint_generic_filler("")
        assert result.genericness_score == 0
        assert result.total_sentences == 0

    def test_flags_data_driven(self):
        text = "He takes a data-driven approach to every decision he makes."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1

    def test_flags_digital_transformation(self):
        text = "They are on a digital transformation journey across the business."
        result = lint_generic_filler(text)
        assert result.generic_count >= 1


# ---------------------------------------------------------------------------
# Evidence Coverage Gate
# ---------------------------------------------------------------------------


class TestEvidenceCoverage:
    def test_fully_tagged_text_passes(self):
        text = (
            "Ben Titmus is CTO at Acme Corp [VERIFIED-PUBLIC] as confirmed online.\n"
            "He mentioned budget constraints in Q4 review [VERIFIED-MEETING] quite clearly.\n"
            "Revenue likely exceeds $10M based on headcount [INFERRED-HIGH] from signals."
        )
        result = check_evidence_coverage(text)
        assert result.passes
        assert result.coverage_pct == 100.0

    def test_untagged_text_fails(self):
        text = (
            "He is responsible for product strategy.\n"
            "The company has 200 employees worldwide.\n"
            "They recently expanded to European markets.\n"
            "Ben leads the engineering organization.\n"
            "Growth has been strong this quarter."
        )
        result = check_evidence_coverage(text)
        assert not result.passes
        assert result.coverage_pct == 0.0

    def test_mixed_coverage(self):
        text = (
            "He is CTO at Acme [VERIFIED-PUBLIC] confirmed online.\n"
            "Revenue grew 20% last year [VERIFIED-PUBLIC] per filings.\n"
            "They seem focused on AI integration across the board.\n"
            "Headcount is approximately 150 [INFERRED-MEDIUM] from signals.\n"
            "The company culture appears collaborative overall."
        )
        result = check_evidence_coverage(text)
        assert result.tagged_count == 3
        assert len(result.untagged_sentences) == 2

    def test_skips_short_lines(self):
        text = "Short.\nAlso short.\nToo brief."
        result = check_evidence_coverage(text)
        assert result.total_substantive == 0

    def test_skips_headers_and_tables(self):
        text = (
            "### Strategic Snapshot\n"
            "| # | Fact | Tag |\n"
            "---\n"
            "This is a real substantive sentence that needs a tag."
        )
        result = check_evidence_coverage(text)
        # Only the last line should count
        assert result.total_substantive == 1

    def test_empty_text_passes(self):
        result = check_evidence_coverage("")
        assert result.passes
        assert result.coverage_pct == 100.0

    def test_coverage_threshold_is_85(self):
        # 5 out of 5 tagged = 100% = pass (well above 85%)
        lines = []
        for i in range(5):
            lines.append(f"Claim {i} about something specific and detailed [VERIFIED-PUBLIC] per sources.")
        result = check_evidence_coverage("\n".join(lines))
        assert result.passes

    def test_coverage_at_85_passes(self):
        # 17 out of 20 tagged = 85% = pass
        lines = []
        for i in range(17):
            lines.append(f"Claim {i} about something specific and detailed [VERIFIED-PUBLIC] per sources.")
        for i in range(3):
            lines.append(f"Untagged claim {i} about something else entirely here.")
        result = check_evidence_coverage("\n".join(lines))
        assert result.passes

    def test_coverage_just_below_80_fails(self):
        # 3 out of 5 = 60% = fail
        lines = []
        for i in range(3):
            lines.append(f"Claim {i} about something specific and detailed. [VERIFIED-PUBLIC]")
        lines.append("This claim has no evidence tag or citation attached.")
        lines.append("Another untagged claim about something important here.")
        result = check_evidence_coverage("\n".join(lines))
        assert not result.passes

    def test_backtick_tags_count(self):
        text = "He is CTO at Acme Corp `[VERIFIED-PUBLIC]` confirmed."
        result = check_evidence_coverage(text)
        assert result.tagged_count == 1


# ---------------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------------


class TestContradictionDetector:
    def test_no_contradictions_for_matching_claims(self):
        claims = [
            {"field": "title", "value": "CTO", "source": "linkedin"},
            {"field": "title", "value": "CTO", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 0

    def test_detects_title_contradiction(self):
        claims = [
            {"field": "title", "value": "CTO", "source": "linkedin"},
            {"field": "title", "value": "VP Engineering", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 1
        assert result[0].field == "title"
        assert result[0].severity == "high"

    def test_detects_company_contradiction(self):
        claims = [
            {"field": "company", "value": "Acme Corp", "source": "linkedin"},
            {"field": "company", "value": "Beta Inc", "source": "news"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 1
        assert result[0].severity == "high"

    def test_location_contradiction_is_medium_severity(self):
        claims = [
            {"field": "location", "value": "London", "source": "linkedin"},
            {"field": "location", "value": "New York", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 1
        assert result[0].severity == "medium"

    def test_ignores_substring_matches(self):
        """'London, UK' and 'London' should not be flagged."""
        claims = [
            {"field": "location", "value": "London", "source": "linkedin"},
            {"field": "location", "value": "London, UK", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 0

    def test_empty_claims(self):
        result = detect_contradictions([])
        assert len(result) == 0

    def test_single_claim_no_contradiction(self):
        claims = [{"field": "title", "value": "CTO", "source": "linkedin"}]
        result = detect_contradictions(claims)
        assert len(result) == 0

    def test_multiple_fields(self):
        claims = [
            {"field": "title", "value": "CTO", "source": "linkedin"},
            {"field": "title", "value": "CEO", "source": "meeting"},
            {"field": "company", "value": "Acme", "source": "linkedin"},
            {"field": "company", "value": "Acme", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 1
        assert result[0].field == "title"

    def test_skips_empty_values(self):
        claims = [
            {"field": "title", "value": "", "source": "linkedin"},
            {"field": "title", "value": "CTO", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 0

    def test_case_insensitive_comparison(self):
        claims = [
            {"field": "title", "value": "cto", "source": "linkedin"},
            {"field": "title", "value": "CTO", "source": "meeting"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 0

    def test_three_way_contradiction(self):
        claims = [
            {"field": "title", "value": "Chief Revenue Officer", "source": "linkedin"},
            {"field": "title", "value": "VP Sales", "source": "meeting"},
            {"field": "title", "value": "Head of Growth", "source": "news"},
        ]
        result = detect_contradictions(claims)
        assert len(result) == 3  # 3 pairs: CRO/VP, CRO/Head, VP/Head


# ---------------------------------------------------------------------------
# Disambiguation Scorer
# ---------------------------------------------------------------------------


class TestDisambiguationScorer:
    def test_zero_score_with_no_data(self):
        result = score_disambiguation(name="John Smith")
        assert result.score == 0
        assert not result.is_locked

    def test_linkedin_url_without_search_gives_10(self):
        """LinkedIn URL present (without retrieval verification) gives +10 weak evidence."""
        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
        )
        # URL present but not verified → +10 weak internal evidence
        assert result.linkedin_url_present
        assert not result.linkedin_confirmed
        assert not result.linkedin_verified_by_retrieval
        assert result.score == 10

    def test_name_in_linkedin_results(self):
        search_results = {
            "linkedin": [
                {"title": "Ben Titmus - CTO at Acme", "snippet": "London area", "link": "https://linkedin.com/in/bentitmus"}
            ]
        }
        result = score_disambiguation(
            name="Ben Titmus",
            search_results=search_results,
        )
        assert result.name_match
        assert result.score >= 15

    def test_company_match_in_linkedin(self):
        search_results = {
            "linkedin": [
                {"title": "Ben Titmus - CTO at Acme Corp", "snippet": "London", "link": "https://linkedin.com/in/bentitmus"}
            ]
        }
        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            search_results=search_results,
        )
        assert result.company_match
        assert result.score >= 25  # 15 name (no url) + 10 employer match

    def test_title_match_across_sources(self):
        search_results = {
            "general": [
                {"title": "Acme Corp appoints new CTO", "snippet": "Ben Titmus named CTO of Acme Corp", "link": "https://example.com"}
            ]
        }
        result = score_disambiguation(
            name="Ben Titmus",
            title="CTO",
            search_results=search_results,
        )
        assert result.title_match
        assert result.score >= 10  # Title match = 10 pts

    def test_apollo_employer_gives_points(self):
        """Apollo data with matching org name adds to employer score."""
        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            apollo_data={
                "name": "Ben Titmus",
                "title": "CTO",
                "organization": {"name": "Acme Corp"},
            },
        )
        assert result.employer_match
        assert result.score >= 10

    def test_location_match(self):
        search_results = {
            "general": [
                {"title": "Some article", "snippet": "Based in London, Ben Titmus leads", "link": "https://example.com"}
            ]
        }
        result = score_disambiguation(
            name="Ben Titmus",
            location="London",
            search_results=search_results,
        )
        assert result.location_match
        assert result.score >= 10  # Location = 10 pts

    def test_high_confidence_locks_identity(self):
        search_results = {
            "linkedin": [
                {"title": "Ben Titmus - CTO at Acme", "snippet": "London", "link": "https://linkedin.com/in/bentitmus"}
            ],
            "general": [
                {"title": "CTO profile", "snippet": "Ben Titmus is CTO at Acme in London", "link": "https://example.com"}
            ],
            "news": [
                {"title": "Acme CTO speaks at conf", "snippet": "Ben Titmus from Acme", "link": "https://news.example.com"}
            ],
        }
        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme",
            title="CTO",
            linkedin_url="https://linkedin.com/in/bentitmus",
            location="London",
            search_results=search_results,
            has_meeting_data=True,
        )
        assert result.is_locked
        assert result.score >= 70

    def test_meeting_data_gives_20_internal_points(self):
        """Meeting data gives +20 for internal confirmation."""
        result = score_disambiguation(
            name="Ben Titmus",
            has_meeting_data=True,
        )
        assert result.meeting_confirmed
        assert result.score == 20  # Internal verified evidence

    def test_multiple_sources_bonus(self):
        search_results = {
            "linkedin": [
                {"title": "Ben Titmus - CTO", "snippet": "Acme", "link": "https://linkedin.com/in/bentitmus"}
            ],
            "general": [
                {"title": "Ben Titmus CTO article", "snippet": "CTO of Acme", "link": "https://example.com"}
            ],
            "news": [
                {"title": "Acme CTO Ben Titmus", "snippet": "Ben Titmus leads", "link": "https://news.example.com"}
            ],
        }
        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme",
            title="CTO",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results=search_results,
        )
        assert result.multiple_sources_agree

    def test_score_capped_at_100(self):
        search_results = {
            "linkedin": [
                {"title": "Ben Titmus - CTO at Acme Corp", "snippet": "London area", "link": "https://linkedin.com/in/bentitmus"}
            ],
            "general": [
                {"title": "CTO at Acme Corp", "snippet": "Ben Titmus CTO London", "link": "https://example.com"}
            ],
        }
        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            title="CTO",
            linkedin_url="https://linkedin.com/in/bentitmus",
            location="London",
            search_results=search_results,
            apollo_data={"name": "Ben Titmus", "title": "CTO", "photo_url": "https://example.com/photo.jpg"},
        )
        assert result.score <= 100

    def test_evidence_trail(self):
        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
        )
        assert len(result.evidence) > 0
        # URL present but not yet verified → weight 10 (weak internal evidence)
        assert result.evidence[0]["weight"] == 10
        assert "not yet verified" in result.evidence[0]["signal"]


# ---------------------------------------------------------------------------
# QA Report
# ---------------------------------------------------------------------------


class TestQAReport:
    def test_passes_all_clean(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=0)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        report.contradictions = []
        assert report.passes_all

    def test_fails_on_high_genericness(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=3)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        assert not report.passes_all

    def test_fails_on_low_coverage(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=0)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=5)
        assert not report.passes_all

    def test_fails_on_contradictions(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=0)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        report.contradictions = [
            Contradiction(field="title", value_a="CTO", source_a="linkedin", value_b="VP", source_b="meeting")
        ]
        assert not report.passes_all


class TestGenerateQAReport:
    def test_generates_complete_report(self):
        text = (
            "He is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "Revenue grew 32% year-over-year. [VERIFIED-PUBLIC]\n"
            "His team has 45 engineers in London. [VERIFIED-MEETING]"
        )
        report = generate_qa_report(text)
        assert isinstance(report, QAReport)
        assert report.genericness.genericness_score == 0

    def test_includes_contradictions(self):
        text = "Some dossier text that is substantive enough."
        claims = [
            {"field": "title", "value": "CTO", "source": "linkedin"},
            {"field": "title", "value": "VP", "source": "meeting"},
        ]
        report = generate_qa_report(text, claims=claims)
        assert len(report.contradictions) == 1

    def test_includes_disambiguation(self):
        text = "Some dossier text that is substantive enough."
        disambiguation = DisambiguationResult(score=85, linkedin_confirmed=True)
        report = generate_qa_report(text, disambiguation=disambiguation)
        assert report.disambiguation.score == 85

    def test_hallucination_flags_on_high_genericness(self):
        # Many generic sentences
        text = (
            "He is a strategic leader who drives innovation.\n"
            "She is passionate about cutting-edge technology.\n"
            "A proven track record of delivering results.\n"
            "Empowers teams with a holistic approach to business.\n"
            "A visionary leader at the intersection of AI and data."
        )
        report = generate_qa_report(text)
        assert report.genericness.genericness_score > 30
        assert any("genericness" in f.lower() for f in report.hallucination_risk_flags)

    def test_hallucination_flags_on_low_coverage(self):
        text = (
            "He is responsible for product strategy overall.\n"
            "The company has 200 employees across offices.\n"
            "They recently expanded into new market segments.\n"
            "Growth has been strong in the past quarter.\n"
            "The team focuses on enterprise customers primarily."
        )
        report = generate_qa_report(text)
        assert not report.evidence_coverage.passes
        assert any("coverage" in f.lower() for f in report.hallucination_risk_flags)


class TestRenderQAReport:
    def test_renders_markdown(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=1)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=9)
        md = render_qa_report_markdown(report)
        assert "## QA Report" in md
        assert "Genericness Score" in md
        assert "Evidence Coverage" in md

    def test_renders_contradictions(self):
        report = QAReport()
        report.contradictions = [
            Contradiction(field="title", value_a="CTO", source_a="linkedin", value_b="VP", source_b="meeting")
        ]
        md = render_qa_report_markdown(report)
        assert "CTO" in md
        assert "VP" in md
        assert "linkedin" in md

    def test_renders_disambiguation(self):
        report = QAReport()
        report.disambiguation = DisambiguationResult(
            score=85,
            evidence=[{"signal": "LinkedIn URL confirmed", "weight": 20, "source": "user_input"}],
        )
        md = render_qa_report_markdown(report)
        assert "85/100" in md
        assert "LOCKED" in md

    def test_renders_hallucination_flags(self):
        report = QAReport()
        report.hallucination_risk_flags = ["High genericness score (45%)"]
        md = render_qa_report_markdown(report)
        assert "Hallucination Risk Flags" in md
        assert "45%" in md

    def test_renders_low_risk_when_clean(self):
        report = QAReport()
        md = render_qa_report_markdown(report)
        assert "Hallucination Risk:** Low" in md

    def test_renders_claims_to_verify(self):
        report = QAReport()
        report.top_claims_to_verify = ["Revenue claim of $10M", "Team size of 45"]
        md = render_qa_report_markdown(report)
        assert "Revenue claim" in md
        assert "Team size" in md

    def test_renders_person_level_ratio(self):
        report = QAReport()
        report.person_level = PersonLevelResult(
            total_lines=10, person_lines=4, company_lines=6, person_pct=40.0
        )
        md = render_qa_report_markdown(report)
        assert "Person-Level Ratio" in md
        assert "40%" in md
        assert "FAIL" in md

    def test_renders_snapshot_validation(self):
        report = QAReport()
        report.snapshot_validation = SnapshotValidation(
            total_bullets=5,
            person_bullets=2,
            non_person_bullets=["- Generic bullet one", "- Generic bullet two", "- Generic three"],
        )
        md = render_qa_report_markdown(report)
        assert "Snapshot Focus" in md
        assert "2/5" in md
        assert "FAIL" in md

    def test_renders_inferred_h_audit(self):
        report = QAReport()
        report.inferred_h_audit = InferredHAudit(
            total_inferred_h=3,
            with_upstream=2,
            without_upstream=[{"sentence": "Some claim [INFERRED-H]", "line": 5}],
        )
        md = render_qa_report_markdown(report)
        assert "INFERRED-H Audit" in md
        assert "2/3" in md


# ---------------------------------------------------------------------------
# Person-Level Ratio
# ---------------------------------------------------------------------------


class TestPersonLevelRatio:
    def test_person_focused_text_passes(self):
        text = (
            "Ben Titmus serves as CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees a team of 45 engineers in London. [VERIFIED-MEETING]\n"
            "Ben led the platform migration in Q3 2025. [VERIFIED-PUBLIC]\n"
            "His primary metric is engineering throughput. [INFERRED-M]\n"
            "Titmus reports directly to the CEO. [INFERRED-H]"
        )
        result = check_person_level_ratio(text, "Ben Titmus")
        assert result.passes
        assert result.person_pct >= 80

    def test_company_heavy_text_fails(self):
        text = (
            "The company has 200 employees across three offices.\n"
            "The organization focuses on enterprise SaaS delivery.\n"
            "Corporate strategy emphasizes market expansion.\n"
            "The firm recently raised a Series B round.\n"
            "The business targets mid-market verticals.\n"
            "Company revenue grew 40% year-over-year.\n"
            "Industry trends suggest consolidation ahead.\n"
            "Ben Titmus is the CTO. [VERIFIED-PUBLIC]\n"
            "He joined in 2023. [VERIFIED-MEETING]"
        )
        result = check_person_level_ratio(text, "Ben Titmus")
        assert not result.passes
        assert result.company_lines >= 5

    def test_empty_text_passes(self):
        result = check_person_level_ratio("", "Test Person")
        assert result.passes
        assert result.person_pct == 100.0

    def test_pronoun_detection(self):
        text = (
            "He was promoted to VP in 2024. [VERIFIED-PUBLIC]\n"
            "She manages the EMEA region. [VERIFIED-MEETING]\n"
            "Their focus is on operational efficiency. [INFERRED-M]"
        )
        result = check_person_level_ratio(text, "Someone Else")
        assert result.person_lines == 3
        assert result.passes

    def test_skips_headers_and_short_lines(self):
        text = (
            "### Strategic Snapshot\n"
            "Short.\n"
            "---\n"
            "He leads the engineering team across all products. [VERIFIED-MEETING]"
        )
        result = check_person_level_ratio(text, "Test Person")
        assert result.total_lines == 1


# ---------------------------------------------------------------------------
# Strategic Snapshot Validator
# ---------------------------------------------------------------------------


class TestSnapshotValidator:
    def test_all_person_bullets_pass(self):
        text = (
            "### 1. Strategic Identity Snapshot\n"
            "- Ben operates as a builder/scaler hybrid. [INFERRED-H]\n"
            "- He creates value through platform consolidation. [VERIFIED-MEETING]\n"
            "- Ben emphasizes technical credibility in public talks. [VERIFIED-PUBLIC]\n"
            "### 2. Verified Fact Table\n"
            "Other content here."
        )
        result = check_snapshot_person_focus(text, "Ben Titmus")
        assert result.passes
        assert result.total_bullets == 3
        assert result.person_bullets == 3

    def test_too_many_non_person_bullets_fail(self):
        text = (
            "### 1. Strategic Identity Snapshot\n"
            "- The company focuses on enterprise SaaS.\n"
            "- Revenue growth driven by market expansion.\n"
            "- Industry consolidation creates opportunity.\n"
            "- Ben leads the engineering team. [VERIFIED-MEETING]\n"
            "### 2. Verified Fact Table\n"
        )
        result = check_snapshot_person_focus(text, "Ben Titmus")
        assert not result.passes
        assert len(result.non_person_bullets) == 3

    def test_two_non_person_bullets_still_passes(self):
        text = (
            "### 1. Strategic Identity Snapshot\n"
            "- The company has 200 employees.\n"
            "- Revenue is approximately $20M.\n"
            "- Ben operates as an optimizer. [VERIFIED-MEETING]\n"
            "- He focuses on platform stability. [INFERRED-M]\n"
            "### 2. Verified Fact Table\n"
        )
        result = check_snapshot_person_focus(text, "Ben Titmus")
        assert result.passes
        assert len(result.non_person_bullets) == 2

    def test_no_snapshot_section(self):
        text = "### Some other section\n- Bullet point here."
        result = check_snapshot_person_focus(text, "Ben Titmus")
        assert result.passes  # No snapshot = nothing to fail
        assert result.total_bullets == 0

    def test_empty_text(self):
        result = check_snapshot_person_focus("", "Ben Titmus")
        assert result.passes
        assert result.total_bullets == 0


# ---------------------------------------------------------------------------
# INFERRED-H Auditor
# ---------------------------------------------------------------------------


class TestInferredHAudit:
    def test_all_inferred_h_have_upstream(self):
        text = (
            "Revenue likely exceeds $10M based on headcount signals [INFERRED-H]\n"
            "He is a builder because meeting transcripts show iterative approach [INFERRED-H]\n"
        )
        result = audit_inferred_h(text)
        assert result.passes
        assert result.total_inferred_h == 2
        assert result.with_upstream == 2

    def test_inferred_h_without_upstream_fails(self):
        text = (
            "Revenue likely exceeds $10M [INFERRED-H]\n"
            "He is probably a builder type [INFERRED-H]\n"
        )
        result = audit_inferred_h(text)
        assert not result.passes
        assert result.total_inferred_h == 2
        assert len(result.without_upstream) == 2

    def test_mixed_inferred_h(self):
        text = (
            "Revenue likely exceeds $10M based on headcount [INFERRED-H]\n"
            "He is probably a builder type [INFERRED-H]\n"
            "Decision rights include budget from meeting evidence [INFERRED-H]\n"
        )
        result = audit_inferred_h(text)
        assert not result.passes
        assert result.with_upstream == 2
        assert len(result.without_upstream) == 1

    def test_no_inferred_h_passes(self):
        text = (
            "He is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "Revenue grew 32%. [VERIFIED-MEETING]\n"
        )
        result = audit_inferred_h(text)
        assert result.passes
        assert result.total_inferred_h == 0

    def test_inferred_h_with_dash_variant(self):
        text = "Revenue data suggests growth based on filings [INFERRED–H]\n"
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 1
        assert result.passes

    def test_inferred_high_variant(self):
        text = "He is a builder [INFERRED-HIGH]\n"
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 1
        assert not result.passes  # No upstream signal words


# ---------------------------------------------------------------------------
# QA Report with new gates
# ---------------------------------------------------------------------------


class TestQAReportNewGates:
    def test_passes_all_requires_person_level(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=0)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        report.contradictions = []
        report.person_level = PersonLevelResult(
            total_lines=10, person_lines=3, company_lines=7, person_pct=30.0
        )
        assert not report.passes_all

    def test_passes_all_with_good_person_level(self):
        report = QAReport()
        report.genericness = GenericFillerResult(total_sentences=10, generic_count=0)
        report.evidence_coverage = EvidenceCoverageResult(total_substantive=10, tagged_count=10)
        report.contradictions = []
        report.person_level = PersonLevelResult(
            total_lines=10, person_lines=8, company_lines=2, person_pct=80.0
        )
        assert report.passes_all

    def test_generate_qa_report_includes_person_level(self):
        text = (
            "Ben Titmus is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees engineering. [VERIFIED-MEETING]\n"
            "Ben led the platform migration. [VERIFIED-PUBLIC]"
        )
        report = generate_qa_report(text, person_name="Ben Titmus")
        assert report.person_level.total_lines > 0
        assert report.person_level.passes

    def test_generate_qa_report_includes_snapshot_validation(self):
        text = (
            "### 1. Strategic Identity Snapshot\n"
            "- Ben operates as a builder. [INFERRED-H]\n"
            "- He creates value through consolidation. [VERIFIED-MEETING]\n"
            "### 2. Verified Fact Table\n"
            "Ben is CTO. [VERIFIED-PUBLIC]"
        )
        report = generate_qa_report(text, person_name="Ben Titmus")
        assert report.snapshot_validation.total_bullets == 2
        assert report.snapshot_validation.passes

    def test_generate_qa_report_includes_inferred_h_audit(self):
        text = (
            "Revenue likely exceeds $10M based on headcount [INFERRED-H]\n"
            "He is a builder type [INFERRED-H]\n"
        )
        report = generate_qa_report(text, person_name="Ben Titmus")
        assert report.inferred_h_audit.total_inferred_h == 2

    def test_hallucination_flags_company_heavy(self):
        text = (
            "The company has 200 employees across three offices.\n"
            "The organization focuses on enterprise SaaS delivery.\n"
            "Corporate strategy emphasizes market expansion.\n"
            "The firm recently raised a Series B round.\n"
            "The business targets mid-market verticals.\n"
            "Company revenue grew 40% year-over-year.\n"
            "Industry trends suggest consolidation ahead.\n"
            "Ben is CTO. [VERIFIED-PUBLIC]"
        )
        report = generate_qa_report(text, person_name="Ben Titmus")
        assert any("company" in f.lower() for f in report.hallucination_risk_flags)

    def test_hallucination_flags_inferred_h_missing_upstream(self):
        text = (
            "Revenue likely exceeds $10M [INFERRED-H]\n"
            "He is a builder [INFERRED-H]\n"
            "Ben is CTO. [VERIFIED-PUBLIC]"
        )
        report = generate_qa_report(text, person_name="Ben Titmus")
        assert any("INFERRED-H" in f for f in report.hallucination_risk_flags)
