"""Tests for the fail-closed Evidence Graph engine.

Covers:
1. EvidenceGraph — node, claim, and ledger management
2. Evidence coverage computation (claim-level and text-level)
3. Fail-closed gate checks (visibility sweep, evidence coverage, entity lock)
4. run_fail_closed_gates aggregate report
5. Visibility query battery generation
6. Highest-signal artifact extraction
7. Visibility coverage confidence scoring
8. QA-level enforce_fail_closed_gates
9. Pydantic models (EvidenceNode, Claim, RetrievalLedgerRow)
10. SerpAPI ledger integration helpers
"""

from __future__ import annotations

import os

os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

from app.brief.evidence_graph import (
    EVIDENCE_COVERAGE_THRESHOLD,
    ENTITY_LOCK_THRESHOLD,
    NARRATIVE_INFLATION_PHRASES,
    VISIBILITY_CATEGORY_GROUPS,
    VISIBILITY_QUERY_TEMPLATES,
    DossierMode,
    EvidenceGraph,
    FailClosedReport,
    GateResult,
    build_failure_report,
    build_meeting_prep_brief,
    build_visibility_queries,
    check_entity_lock_gate,
    check_evidence_coverage_gate,
    check_visibility_sweep_gate,
    compute_evidence_coverage,
    compute_evidence_coverage_from_text,
    compute_visibility_coverage_confidence,
    determine_dossier_mode,
    extract_highest_signal_artifacts,
    filter_prose_by_mode,
    prune_uncited_factual_lines,
    run_fail_closed_gates,
    validate_narrative_inflation,
    validate_pressure_evidence,
    validate_visibility_artifact_table,
)
from app.brief.qa import (
    audit_inferred_h,
    enforce_fail_closed_gates,
)
from app.models import Claim, EvidenceNode, RetrievalLedgerRow


# ---------------------------------------------------------------------------
# Pydantic model basics
# ---------------------------------------------------------------------------


class TestEvidenceNodeModel:
    def test_create_evidence_node(self):
        node = EvidenceNode(
            id="E1", type="MEETING", source="Q1 Review call",
            snippet="Discussed pipeline risks", date="2026-01-15",
        )
        assert node.id == "E1"
        assert node.type == "MEETING"
        assert node.snippet == "Discussed pipeline risks"

    def test_evidence_node_defaults(self):
        node = EvidenceNode(
            id="E2", type="PUBLIC", source="https://example.com",
            snippet="A brief excerpt",
        )
        assert node.ref == ""
        assert node.date == "UNKNOWN"

    def test_evidence_node_serializable(self):
        node = EvidenceNode(
            id="E3", type="PUBLIC", source="ted.com",
            snippet="Talk about AI", date="2025-06-01",
        )
        d = node.model_dump()
        assert d["id"] == "E3"
        assert d["type"] == "PUBLIC"


class TestClaimModel:
    def test_create_claim(self):
        claim = Claim(
            claim_id="C1", text="Ben is CTO at Acme",
            tag="VERIFIED-PUBLIC", evidence_ids=["E1", "E2"],
            confidence="H",
        )
        assert claim.claim_id == "C1"
        assert claim.tag == "VERIFIED-PUBLIC"
        assert len(claim.evidence_ids) == 2
        assert claim.confidence == "H"

    def test_claim_defaults(self):
        claim = Claim(claim_id="C2", text="Some assertion", tag="UNKNOWN")
        assert claim.evidence_ids == []
        assert claim.confidence == "L"

    def test_claim_serializable(self):
        claim = Claim(
            claim_id="C3", text="Test", tag="INFERRED-H",
            evidence_ids=["E1"],
        )
        d = claim.model_dump()
        assert d["claim_id"] == "C3"
        assert d["evidence_ids"] == ["E1"]


class TestRetrievalLedgerRowModel:
    def test_create_ledger_row(self):
        row = RetrievalLedgerRow(
            query_id="Q1",
            query='"Ben Titmus" TED',
            intent="visibility",
            top_results=[{"rank": 1, "title": "Test", "url": "https://x.com"}],
            selected_evidence_ids=["E1"],
            result_count=3,
        )
        assert row.query_id == "Q1"
        assert row.intent == "visibility"
        assert row.result_count == 3
        assert len(row.top_results) == 1

    def test_ledger_row_defaults(self):
        row = RetrievalLedgerRow(
            query_id="Q2", query="test", intent="bio",
        )
        assert row.top_results == []
        assert row.selected_evidence_ids == []
        assert row.result_count == 0


# ---------------------------------------------------------------------------
# EvidenceGraph core operations
# ---------------------------------------------------------------------------


class TestEvidenceGraphNodes:
    def test_add_node(self):
        g = EvidenceGraph()
        node = g.add_node(type="MEETING", source="call", snippet="test snippet")
        assert node.id == "E1"
        assert node.type == "MEETING"
        assert "E1" in g.nodes

    def test_add_meeting_node(self):
        g = EvidenceGraph()
        node = g.add_meeting_node(source="Q1 Review", snippet="Pipeline risk")
        assert node.type == "MEETING"
        assert node.id == "E1"

    def test_add_public_node(self):
        g = EvidenceGraph()
        node = g.add_public_node(source="https://ted.com/talk", snippet="AI keynote")
        assert node.type == "PUBLIC"
        assert node.id == "E1"

    def test_node_ids_increment(self):
        g = EvidenceGraph()
        n1 = g.add_node(type="MEETING", source="a", snippet="s1")
        n2 = g.add_node(type="PUBLIC", source="b", snippet="s2")
        n3 = g.add_node(type="MEETING", source="c", snippet="s3")
        assert n1.id == "E1"
        assert n2.id == "E2"
        assert n3.id == "E3"
        assert len(g.nodes) == 3

    def test_node_snippet_truncated(self):
        g = EvidenceGraph()
        long_snippet = "x" * 500
        node = g.add_node(type="PUBLIC", source="url", snippet=long_snippet)
        assert len(node.snippet) == 200

    def test_get_node(self):
        g = EvidenceGraph()
        n = g.add_meeting_node(source="call", snippet="test")
        assert g.get_node("E1") == n
        assert g.get_node("E99") is None

    def test_add_node_with_date_and_ref(self):
        g = EvidenceGraph()
        node = g.add_node(
            type="MEETING", source="call", snippet="test",
            ref="00:15:30", date="2026-01-15",
        )
        assert node.ref == "00:15:30"
        assert node.date == "2026-01-15"


class TestEvidenceGraphClaims:
    def test_add_claim(self):
        g = EvidenceGraph()
        claim = g.add_claim(
            text="Ben is CTO at Acme", tag="VERIFIED-PUBLIC",
            evidence_ids=["E1"], confidence="H",
        )
        assert claim.claim_id == "C1"
        assert "C1" in g.claims

    def test_claim_ids_increment(self):
        g = EvidenceGraph()
        c1 = g.add_claim(text="Claim 1", tag="VERIFIED-MEETING")
        c2 = g.add_claim(text="Claim 2", tag="INFERRED-H")
        assert c1.claim_id == "C1"
        assert c2.claim_id == "C2"

    def test_get_claim(self):
        g = EvidenceGraph()
        c = g.add_claim(text="Test claim", tag="UNKNOWN")
        assert g.get_claim("C1") == c
        assert g.get_claim("C99") is None

    def test_claim_defaults(self):
        g = EvidenceGraph()
        c = g.add_claim(text="No evidence", tag="UNKNOWN")
        assert c.evidence_ids == []
        assert c.confidence == "L"


class TestEvidenceGraphLedger:
    def test_log_retrieval_basic(self):
        g = EvidenceGraph()
        row = g.log_retrieval(
            query='"Ben" TED', intent="visibility",
            results=[{"title": "Talk", "link": "https://ted.com/x"}],
        )
        assert row.query_id == "Q1"
        assert row.intent == "visibility"
        assert row.result_count == 1
        assert len(g.ledger) == 1

    def test_log_retrieval_zero_results(self):
        g = EvidenceGraph()
        row = g.log_retrieval(
            query='"Ben" Vimeo', intent="visibility",
            results=[],
        )
        assert row.result_count == 0
        assert len(g.ledger) == 1

    def test_log_retrieval_none_results(self):
        g = EvidenceGraph()
        row = g.log_retrieval(
            query='"Ben" SlideShare', intent="visibility",
        )
        assert row.result_count == 0
        assert row.top_results == []

    def test_query_ids_increment(self):
        g = EvidenceGraph()
        r1 = g.log_retrieval(query="q1", intent="bio")
        r2 = g.log_retrieval(query="q2", intent="visibility")
        r3 = g.log_retrieval(query="q3", intent="press")
        assert r1.query_id == "Q1"
        assert r2.query_id == "Q2"
        assert r3.query_id == "Q3"

    def test_top_results_capped_at_5(self):
        g = EvidenceGraph()
        results = [{"title": f"Result {i}", "link": f"https://x.com/{i}"} for i in range(10)]
        row = g.log_retrieval(query="test", intent="bio", results=results)
        assert len(row.top_results) == 5
        assert row.result_count == 10

    def test_top_results_snippet_truncated(self):
        g = EvidenceGraph()
        results = [{"title": "T", "link": "https://x.com", "snippet": "y" * 500}]
        row = g.log_retrieval(query="test", intent="bio", results=results)
        assert len(row.top_results[0]["snippet"]) == 200

    def test_get_visibility_ledger_rows(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="bio")
        g.log_retrieval(query="q2", intent="visibility")
        g.log_retrieval(query="q3", intent="visibility")
        g.log_retrieval(query="q4", intent="press")
        vis_rows = g.get_visibility_ledger_rows()
        assert len(vis_rows) == 2
        assert all(r.intent == "visibility" for r in vis_rows)

    def test_selected_evidence_ids(self):
        g = EvidenceGraph()
        row = g.log_retrieval(
            query="test", intent="bio",
            results=[{"title": "T"}],
            selected_evidence_ids=["E1", "E2"],
        )
        assert row.selected_evidence_ids == ["E1", "E2"]


class TestEvidenceGraphSerialization:
    def test_to_dict_empty(self):
        g = EvidenceGraph()
        d = g.to_dict()
        assert d == {"nodes": [], "claims": [], "ledger": []}

    def test_to_dict_with_data(self):
        g = EvidenceGraph()
        g.add_meeting_node(source="call", snippet="test")
        g.add_claim(text="Claim 1", tag="VERIFIED-MEETING", evidence_ids=["E1"])
        g.log_retrieval(query="q1", intent="bio", results=[])
        d = g.to_dict()
        assert len(d["nodes"]) == 1
        assert len(d["claims"]) == 1
        assert len(d["ledger"]) == 1
        assert d["nodes"][0]["id"] == "E1"
        assert d["claims"][0]["claim_id"] == "C1"
        assert d["ledger"][0]["query_id"] == "Q1"


# ---------------------------------------------------------------------------
# Evidence Coverage Computation
# ---------------------------------------------------------------------------


class TestEvidenceCoverageClaims:
    def test_all_verified_100_pct(self):
        claims = [
            Claim(claim_id="C1", text="Fact 1", tag="VERIFIED-PUBLIC", evidence_ids=["E1"]),
            Claim(claim_id="C2", text="Fact 2", tag="VERIFIED-MEETING", evidence_ids=["E2"]),
        ]
        assert compute_evidence_coverage(claims) == 100.0

    def test_all_unknown_0_pct(self):
        claims = [
            Claim(claim_id="C1", text="Unknown 1", tag="UNKNOWN"),
            Claim(claim_id="C2", text="Unknown 2", tag="UNKNOWN"),
        ]
        assert compute_evidence_coverage(claims) == 0.0

    def test_mixed_coverage(self):
        claims = [
            Claim(claim_id="C1", text="Fact", tag="VERIFIED-PUBLIC", evidence_ids=["E1"]),
            Claim(claim_id="C2", text="Inferred", tag="INFERRED-H", evidence_ids=["E1"]),
            Claim(claim_id="C3", text="Unknown", tag="UNKNOWN"),
            Claim(claim_id="C4", text="Unknown 2", tag="UNKNOWN"),
        ]
        # 2 covered / 4 total = 50%
        assert compute_evidence_coverage(claims) == 50.0

    def test_empty_claims(self):
        assert compute_evidence_coverage([]) == 0.0

    def test_unknown_with_evidence_ids_counts(self):
        """An UNKNOWN claim with evidence_ids still counts as covered."""
        claims = [
            Claim(claim_id="C1", text="Partially known", tag="UNKNOWN", evidence_ids=["E1"]),
        ]
        assert compute_evidence_coverage(claims) == 100.0

    def test_inferred_variants(self):
        claims = [
            Claim(claim_id="C1", text="H", tag="INFERRED-H"),
            Claim(claim_id="C2", text="M", tag="INFERRED-M"),
            Claim(claim_id="C3", text="L", tag="INFERRED-L"),
        ]
        assert compute_evidence_coverage(claims) == 100.0

    def test_threshold_constant(self):
        assert EVIDENCE_COVERAGE_THRESHOLD == 85.0


class TestEvidenceCoverageText:
    def test_all_tagged_text(self):
        text = (
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees 45 engineers. [VERIFIED-MEETING]\n"
            "Pipeline risk was flagged. [INFERRED-H]\n"
        )
        assert compute_evidence_coverage_from_text(text) == 100.0

    def test_mixed_tagged_text(self):
        text = (
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He is a strategic leader in the industry.\n"
            "Revenue grew significantly last quarter.\n"
        )
        coverage = compute_evidence_coverage_from_text(text)
        # 1 tagged / 3 total ≈ 33.3%
        assert 30.0 <= coverage <= 40.0

    def test_headers_and_short_lines_excluded(self):
        text = (
            "### Section 1\n"
            "---\n"
            "Short.\n"
            "| Table | Row |\n"
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
        )
        # Only the last line is substantive, and it's tagged
        assert compute_evidence_coverage_from_text(text) == 100.0

    def test_empty_text(self):
        assert compute_evidence_coverage_from_text("") == 100.0

    def test_no_substantive_lines(self):
        text = "# Header\n---\n| x | y |\nOk.\n"
        assert compute_evidence_coverage_from_text(text) == 100.0

    def test_unknown_tag_counted(self):
        text = "We have no data on this topic. [UNKNOWN]\n"
        assert compute_evidence_coverage_from_text(text) == 100.0

    def test_en_dash_variant(self):
        """Tags with en-dash (VERIFIED\u2013PUBLIC) should match."""
        text = "Ben is CTO at Acme. [VERIFIED\u2013PUBLIC]\n"
        assert compute_evidence_coverage_from_text(text) == 100.0


# ---------------------------------------------------------------------------
# Fail-Closed Gate Checks
# ---------------------------------------------------------------------------


class TestVisibilitySweepGate:
    def test_passes_with_visibility_rows(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="visibility", results=[])
        g.log_retrieval(query="q2", intent="visibility", results=[])
        result = check_visibility_sweep_gate(g)
        assert result.passed
        assert result.gate_name == "VISIBILITY_SWEEP"
        assert "2 visibility" in result.details

    def test_fails_without_visibility_rows(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="bio", results=[])
        result = check_visibility_sweep_gate(g)
        assert not result.passed
        assert "FAIL" in result.remediation
        assert "VISIBILITY SWEEP NOT EXECUTED" in result.remediation

    def test_fails_with_empty_graph(self):
        g = EvidenceGraph()
        result = check_visibility_sweep_gate(g)
        assert not result.passed

    def test_non_visibility_intents_dont_count(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="bio")
        g.log_retrieval(query="q2", intent="press")
        g.log_retrieval(query="q3", intent="entity_lock")
        result = check_visibility_sweep_gate(g)
        assert not result.passed


class TestEvidenceCoverageGate:
    def test_passes_at_threshold(self):
        claims = [
            Claim(claim_id=f"C{i}", text=f"F{i}", tag="VERIFIED-PUBLIC", evidence_ids=["E1"])
            for i in range(85)
        ] + [
            Claim(claim_id=f"C{85+i}", text=f"U{i}", tag="UNKNOWN")
            for i in range(15)
        ]
        result = check_evidence_coverage_gate(claims)
        assert result.passed

    def test_fails_below_threshold(self):
        claims = [
            Claim(claim_id=f"C{i}", text=f"F{i}", tag="VERIFIED-PUBLIC", evidence_ids=["E1"])
            for i in range(50)
        ] + [
            Claim(claim_id=f"C{50+i}", text=f"U{i}", tag="UNKNOWN")
            for i in range(50)
        ]
        result = check_evidence_coverage_gate(claims)
        assert not result.passed
        assert "FAIL: EVIDENCE COVERAGE" in result.remediation

    def test_falls_back_to_text(self):
        text = (
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He oversees 45 engineers. [VERIFIED-MEETING]\n"
            "Pipeline risk was flagged. [INFERRED-H]\n"
        )
        result = check_evidence_coverage_gate([], dossier_text=text)
        assert result.passed

    def test_zero_coverage_when_no_input(self):
        result = check_evidence_coverage_gate([])
        assert not result.passed

    def test_gate_name(self):
        result = check_evidence_coverage_gate([])
        assert result.gate_name == "EVIDENCE_COVERAGE"


class TestEntityLockGate:
    def test_locked_at_70(self):
        result = check_entity_lock_gate(70)
        assert result.passed
        assert "LOCKED" in result.details

    def test_locked_at_100(self):
        result = check_entity_lock_gate(100)
        assert result.passed

    def test_partial_at_50(self):
        result = check_entity_lock_gate(50)
        assert not result.passed
        assert "PARTIAL" in result.details

    def test_locked_at_69(self):
        """69 >= 60 threshold — should be LOCKED (full dossier)."""
        result = check_entity_lock_gate(69)
        assert result.passed
        assert "LOCKED" in result.details

    def test_partial_at_59(self):
        """59 < 60 threshold — should be PARTIAL."""
        result = check_entity_lock_gate(59)
        assert not result.passed
        assert "PARTIAL" in result.details

    def test_not_locked_below_50(self):
        result = check_entity_lock_gate(30)
        assert not result.passed
        assert "NOT LOCKED" in result.details

    def test_gate_name(self):
        result = check_entity_lock_gate(80)
        assert result.gate_name == "ENTITY_LOCK"

    def test_threshold_constant(self):
        assert ENTITY_LOCK_THRESHOLD == 60

    def test_remediation_on_failure(self):
        result = check_entity_lock_gate(40)
        assert "Confirm LinkedIn" in result.remediation


# ---------------------------------------------------------------------------
# run_fail_closed_gates aggregate
# ---------------------------------------------------------------------------


class TestRunFailClosedGates:
    def test_all_pass(self):
        g = EvidenceGraph()
        # Add visibility rows
        for i in range(16):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        # Add high-coverage claims
        for i in range(10):
            g.add_claim(
                text=f"Verified fact {i}", tag="VERIFIED-PUBLIC",
                evidence_ids=["E1"],
            )
        report = run_fail_closed_gates(g, entity_lock_score=85)
        assert report.all_passed
        assert not report.is_constrained
        assert not report.should_halt
        assert len(report.gates) == 3

    def test_visibility_sweep_halts(self):
        g = EvidenceGraph()
        # No visibility rows
        for i in range(10):
            g.add_claim(text=f"F{i}", tag="VERIFIED-PUBLIC", evidence_ids=["E1"])
        report = run_fail_closed_gates(g, entity_lock_score=85)
        assert not report.all_passed
        assert report.should_halt
        assert "HALTED" in report.failure_output
        assert "VISIBILITY_SWEEP" in report.failure_output

    def test_low_coverage_halts(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="visibility")
        # Add mostly unknown claims
        for i in range(10):
            g.add_claim(text=f"U{i}", tag="UNKNOWN")
        report = run_fail_closed_gates(g, entity_lock_score=85)
        assert not report.all_passed
        assert report.should_halt

    def test_entity_lock_constrains(self):
        g = EvidenceGraph()
        for i in range(16):
            g.log_retrieval(query=f"q{i}", intent="visibility")
        for i in range(10):
            g.add_claim(text=f"F{i}", tag="VERIFIED-PUBLIC", evidence_ids=["E1"])
        report = run_fail_closed_gates(g, entity_lock_score=55)
        assert not report.all_passed
        assert report.is_constrained
        assert not report.should_halt  # Constrained, not halted

    def test_both_hard_gates_fail(self):
        g = EvidenceGraph()
        # No visibility, no claims
        report = run_fail_closed_gates(g, entity_lock_score=85)
        assert report.should_halt
        assert "VISIBILITY_SWEEP" in report.failure_output
        assert "EVIDENCE_COVERAGE" in report.failure_output

    def test_text_fallback_coverage(self):
        g = EvidenceGraph()
        for i in range(16):
            g.log_retrieval(query=f"q{i}", intent="visibility")
        text = "\n".join(
            f"Fact {i} is verified. [VERIFIED-PUBLIC]" for i in range(20)
        )
        report = run_fail_closed_gates(g, entity_lock_score=85, dossier_text=text)
        assert report.all_passed

    def test_failure_output_includes_entity_lock(self):
        g = EvidenceGraph()
        report = run_fail_closed_gates(g, entity_lock_score=40)
        assert "ENTITY_LOCK" in report.failure_output

    def test_gate_result_dataclass(self):
        gr = GateResult(gate_name="TEST", passed=True, details="ok")
        assert gr.gate_name == "TEST"
        assert gr.passed
        assert gr.remediation == ""

    def test_fail_closed_report_dataclass(self):
        r = FailClosedReport()
        assert r.gates == []
        assert not r.all_passed
        assert not r.is_constrained
        assert r.failure_output == ""


# ---------------------------------------------------------------------------
# Visibility Query Battery
# ---------------------------------------------------------------------------


class TestVisibilityQueryBattery:
    def test_template_count(self):
        assert len(VISIBILITY_QUERY_TEMPLATES) == 15

    def test_all_templates_are_visibility_intent(self):
        for _, intent in VISIBILITY_QUERY_TEMPLATES:
            assert intent == "visibility"

    def test_category_groups_cover_all_templates(self):
        all_indices = set()
        for indices in VISIBILITY_CATEGORY_GROUPS.values():
            all_indices.update(indices)
        # Should cover all 15 template indices
        assert all_indices == set(range(15))

    def test_four_category_groups(self):
        assert len(VISIBILITY_CATEGORY_GROUPS) == 4
        assert "ted_tedx" in VISIBILITY_CATEGORY_GROUPS
        assert "keynote_conference" in VISIBILITY_CATEGORY_GROUPS
        assert "podcast_webinar" in VISIBILITY_CATEGORY_GROUPS
        assert "youtube_video" in VISIBILITY_CATEGORY_GROUPS

    def test_build_visibility_queries_basic(self):
        queries = build_visibility_queries("Ben Titmus")
        assert len(queries) >= 15
        for query, intent in queries:
            assert intent == "visibility"
            assert "Ben Titmus" in query

    def test_build_visibility_queries_with_company(self):
        queries = build_visibility_queries("Ben Titmus", "Acme Corp")
        # Should have extra company-qualified query
        assert len(queries) >= 16
        company_queries = [q for q, _ in queries if "Acme Corp" in q]
        assert len(company_queries) >= 1

    def test_build_queries_contains_ted_keywords(self):
        queries = build_visibility_queries("Test Person")
        query_text = " ".join(q for q, _ in queries)
        assert "TED" in query_text
        assert "TEDx" in query_text

    def test_build_queries_contains_podcast_keywords(self):
        queries = build_visibility_queries("Test Person")
        query_text = " ".join(q for q, _ in queries)
        assert "podcast" in query_text.lower()

    def test_build_queries_contains_keynote_keywords(self):
        queries = build_visibility_queries("Test Person")
        query_text = " ".join(q for q, _ in queries)
        assert "keynote" in query_text.lower()

    def test_build_queries_contains_youtube_keywords(self):
        queries = build_visibility_queries("Test Person")
        query_text = " ".join(q for q, _ in queries)
        assert "YouTube" in query_text


# ---------------------------------------------------------------------------
# Highest-Signal Artifact Extraction
# ---------------------------------------------------------------------------


class TestHighestSignalArtifacts:
    def test_empty_graph(self):
        g = EvidenceGraph()
        artifacts = extract_highest_signal_artifacts(g)
        assert artifacts == []

    def test_extracts_from_visibility_rows(self):
        g = EvidenceGraph()
        g.log_retrieval(
            query='"Ben" TED', intent="visibility",
            results=[
                {"title": "Ben at TED 2025", "link": "https://ted.com/talks/ben"},
                {"title": "Another Talk", "link": "https://youtube.com/watch?v=abc"},
            ],
        )
        artifacts = extract_highest_signal_artifacts(g)
        assert len(artifacts) >= 1
        assert artifacts[0]["url"] == "https://ted.com/talks/ben"  # TED > YouTube

    def test_max_artifacts_limit(self):
        g = EvidenceGraph()
        for i in range(10):
            g.log_retrieval(
                query=f"q{i}", intent="visibility",
                results=[{"title": f"Talk {i}", "link": f"https://x.com/{i}"}],
            )
        artifacts = extract_highest_signal_artifacts(g, max_artifacts=3)
        assert len(artifacts) <= 3

    def test_deduplicates_by_url(self):
        g = EvidenceGraph()
        g.log_retrieval(
            query="q1", intent="visibility",
            results=[{"title": "Talk A", "link": "https://ted.com/x"}],
        )
        g.log_retrieval(
            query="q2", intent="visibility",
            results=[{"title": "Talk A copy", "link": "https://ted.com/x"}],
        )
        artifacts = extract_highest_signal_artifacts(g)
        urls = [a["url"] for a in artifacts]
        assert len(set(urls)) == len(urls)

    def test_priority_ordering(self):
        g = EvidenceGraph()
        g.log_retrieval(
            query='"Ben" podcast', intent="visibility",
            results=[{"title": "Ben on Podcast", "link": "https://podcast.com/ep1"}],
        )
        g.log_retrieval(
            query='"Ben" TED', intent="visibility",
            results=[{"title": "Ben TED Talk", "link": "https://ted.com/talks/ben"}],
        )
        artifacts = extract_highest_signal_artifacts(g)
        # TED should be ranked higher
        assert artifacts[0]["url"] == "https://ted.com/talks/ben"

    def test_non_visibility_rows_excluded(self):
        g = EvidenceGraph()
        g.log_retrieval(
            query="q1", intent="bio",
            results=[{"title": "Bio page", "link": "https://company.com/bio"}],
        )
        artifacts = extract_highest_signal_artifacts(g)
        assert artifacts == []

    def test_artifact_fields(self):
        g = EvidenceGraph()
        g.log_retrieval(
            query='"Ben" keynote', intent="visibility",
            results=[{
                "title": "Keynote at Summit",
                "link": "https://summit.com/video",
                "date": "2025-03-15",
            }],
        )
        artifacts = extract_highest_signal_artifacts(g)
        assert len(artifacts) == 1
        a = artifacts[0]
        assert "title" in a
        assert "venue" in a
        assert "date" in a
        assert "url" in a
        assert "why_it_matters" in a


# ---------------------------------------------------------------------------
# Visibility Coverage Confidence
# ---------------------------------------------------------------------------


class TestVisibilityCoverageConfidence:
    """Scoring: +10 per query family with results, +10 for TED/TEDx execution, cap 100."""

    def test_empty_graph_zero(self):
        g = EvidenceGraph()
        assert compute_visibility_coverage_confidence(g) == 0

    def test_no_visibility_rows_zero(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="bio", results=[{"title": "T"}])
        assert compute_visibility_coverage_confidence(g) == 0

    def test_all_categories_with_results(self):
        g = EvidenceGraph()
        # 15 visibility rows with results → 10 families × 10 + 10 TED bonus = 110 → cap 100
        for i in range(15):
            g.log_retrieval(
                query=f"q{i}", intent="visibility",
                results=[{"title": f"R{i}", "link": f"https://x.com/{i}"}],
            )
        assert compute_visibility_coverage_confidence(g) == 100

    def test_ted_tedx_only_with_results(self):
        g = EvidenceGraph()
        # Queries 0-3 (ted/tedx families) have results, rest empty
        # Families with results: {ted, tedx} = 2 × 10 = 20 + 10 TED bonus = 30
        for i in range(4):
            g.log_retrieval(
                query=f"q{i}", intent="visibility",
                results=[{"title": f"R{i}", "link": f"https://x.com/{i}"}],
            )
        for i in range(4, 15):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        assert compute_visibility_coverage_confidence(g) == 30

    def test_ted_and_keynotes_with_results(self):
        g = EvidenceGraph()
        # Queries 0-7 (ted/tedx + keynote/conference/summit/panel) have results, rest empty
        # Families: ted, tedx, keynote, conference, summit, panel = 6 × 10 = 60 + 10 TED = 70
        for i in range(8):
            g.log_retrieval(
                query=f"q{i}", intent="visibility",
                results=[{"title": f"R{i}", "link": f"https://x.com/{i}"}],
            )
        for i in range(8, 15):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        assert compute_visibility_coverage_confidence(g) == 70

    def test_ted_executed_but_no_results_gives_bonus(self):
        g = EvidenceGraph()
        # TED/TEDx queries executed with 0 results → +10 bonus only
        for i in range(4):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        assert compute_visibility_coverage_confidence(g) == 10

    def test_podcasts_only_no_ted_bonus(self):
        g = EvidenceGraph()
        # Skip first 8 queries (TED + keynote) — no TED bonus
        for i in range(8):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        # Only podcast/webinar queries (8-11) have results
        for i in range(8, 12):
            g.log_retrieval(
                query=f"q{i}", intent="visibility",
                results=[{"title": f"R{i}", "link": f"https://x.com/{i}"}],
            )
        for i in range(12, 15):
            g.log_retrieval(query=f"q{i}", intent="visibility", results=[])
        # Families: podcast, webinar, interview_video = 3 × 10 = 30 + 10 TED exec = 40
        assert compute_visibility_coverage_confidence(g) == 40


# ---------------------------------------------------------------------------
# enforce_fail_closed_gates (QA module)
# ---------------------------------------------------------------------------


class TestEnforceFailClosedGates:
    def test_all_pass(self):
        text = "\n".join(
            f"Fact {i} is verified. [VERIFIED-PUBLIC]" for i in range(20)
        )
        should_output, message = enforce_fail_closed_gates(
            dossier_text=text,
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert should_output
        assert message == ""

    def test_visibility_failure(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not should_output
        assert "VISIBILITY SWEEP NOT EXECUTED" in message

    def test_evidence_coverage_failure(self):
        # With 10+ web results, threshold is 85% — so 50% fails
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            web_results_count=15,
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message

    def test_both_failures(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            web_results_count=15,
        )
        assert not should_output
        assert "VISIBILITY SWEEP" in message
        assert "EVIDENCE COVERAGE" in message

    def test_entity_lock_included_in_failure(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=40,
            visibility_ledger_count=0,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            web_results_count=15,
        )
        assert not should_output
        assert "Entity Lock: 40/100" in message
        assert "NOT LOCKED" in message

    def test_passes_at_exact_thresholds_high_visibility(self):
        # High visibility (10+ web results) requires 85% coverage
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=70,
            visibility_ledger_count=12,
            evidence_coverage_pct=85.0,
            person_name="Test",
            web_results_count=15,
        )
        assert should_output

    def test_flat_85_threshold_rejects_62_pct(self):
        # v2: flat 85% threshold — 62% always fails regardless of web results
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=70,
            visibility_ledger_count=12,
            evidence_coverage_pct=62.0,
            person_name="Test",
            web_results_count=3,
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message

    def test_flat_85_threshold_rejects_72_pct(self):
        # v2: flat 85% threshold — 72% fails even with moderate web results
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=70,
            visibility_ledger_count=12,
            evidence_coverage_pct=72.0,
            person_name="Test",
            web_results_count=7,
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message

    def test_flat_85_threshold_rejects_84_pct(self):
        # v2: flat 85% — 84.9% fails (no partial pass)
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=84.9,
            person_name="Test",
            web_results_count=2,
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message

    def test_flat_85_threshold_passes_at_85(self):
        # v2: exactly 85% passes
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=70,
            visibility_ledger_count=12,
            evidence_coverage_pct=85.0,
            person_name="Test",
            web_results_count=3,
        )
        assert should_output


# ---------------------------------------------------------------------------
# Profiler fail-closed prompt rules
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Dossier Mode Determination (Pre-Synthesis Gates)
# ---------------------------------------------------------------------------


class TestDossierMode:
    def test_full_mode_when_locked(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.FULL
        assert "LOCKED" in reason

    def test_constrained_mode_when_partial(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=55,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.CONSTRAINED
        assert "PARTIAL DOSSIER" in reason

    def test_constrained_mode_when_not_locked(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=30,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.CONSTRAINED
        assert "NOT LOCKED" in reason

    def test_halted_when_no_visibility(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=False,
            has_public_results=True,
        )
        assert mode == DossierMode.HALTED
        assert "VISIBILITY SWEEP NOT EXECUTED" in reason

    def test_halted_when_no_public_results(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=True,
            has_public_results=False,
        )
        assert mode == DossierMode.HALTED
        assert "NO PUBLIC RETRIEVAL" in reason

    def test_halted_when_both_missing(self):
        mode, _ = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=False,
            has_public_results=False,
        )
        assert mode == DossierMode.HALTED

    def test_halted_includes_queries_to_run(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=False,
            has_public_results=True,
            person_name="Ben Titmus",
        )
        assert mode == DossierMode.HALTED
        assert '"Ben Titmus" TED' in reason
        assert '"Ben Titmus" podcast' in reason

    def test_mode_constants(self):
        assert DossierMode.FULL == "full"
        assert DossierMode.CONSTRAINED == "constrained"
        assert DossierMode.HALTED == "halted"
        assert DossierMode.MEETING_PREP == "meeting_prep"
        assert DossierMode.DEEP_RESEARCH == "deep_research"

    def test_deep_research_status_constants(self):
        assert DossierMode.NOT_STARTED == "NOT_STARTED"
        assert DossierMode.RUNNING == "RUNNING"
        assert DossierMode.FAILED == "FAILED"
        assert DossierMode.SUCCEEDED == "SUCCEEDED"

    def test_full_mode_at_exact_threshold(self):
        mode, _ = determine_dossier_mode(
            entity_lock_score=70,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.FULL

    def test_constrained_at_50(self):
        mode, _ = determine_dossier_mode(
            entity_lock_score=50,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.CONSTRAINED


# ---------------------------------------------------------------------------
# Prose Filtering by Mode
# ---------------------------------------------------------------------------


class TestFilterProseByMode:
    def test_full_mode_no_change(self):
        text = "Ben is CTO at Acme. [VERIFIED-PUBLIC]\nHe leads AI strategy. [INFERRED-H]"
        result = filter_prose_by_mode(text, DossierMode.FULL, entity_lock_score=85)
        assert result == text

    def test_constrained_partial_strips_inferred_h(self):
        text = (
            "Ben is CTO at Acme. [VERIFIED-PUBLIC]\n"
            "He leads AI strategy. [INFERRED-H]\n"
            "Unknown fact remains. [UNKNOWN]\n"
            "Low confidence inference. [INFERRED-L]\n"
        )
        result = filter_prose_by_mode(text, DossierMode.CONSTRAINED, entity_lock_score=55)
        assert "[VERIFIED-PUBLIC]" in result
        assert "[INFERRED-H]" not in result
        assert "[UNKNOWN]" in result
        assert "[INFERRED-L]" in result
        assert "PARTIAL DOSSIER" in result

    def test_constrained_partial_strips_inferred_m(self):
        text = "Medium inference claim here. [INFERRED-M]\nVerified. [VERIFIED-MEETING]"
        result = filter_prose_by_mode(text, DossierMode.CONSTRAINED, entity_lock_score=60)
        assert "[INFERRED-M]" not in result
        assert "[VERIFIED-MEETING]" in result

    def test_constrained_not_locked_strips_all_inferred(self):
        text = (
            "Ben is CTO. [VERIFIED-PUBLIC]\n"
            "He leads AI. [INFERRED-H]\n"
            "Maybe a board member. [INFERRED-M]\n"
            "Possibly in London. [INFERRED-L]\n"
        )
        result = filter_prose_by_mode(text, DossierMode.CONSTRAINED, entity_lock_score=30)
        assert "[VERIFIED-PUBLIC]" in result
        assert "[INFERRED-H]" not in result
        assert "[INFERRED-M]" not in result
        assert "[INFERRED-L]" not in result
        assert "NOT LOCKED" in result

    def test_constrained_preserves_headers(self):
        text = "### Section 1\n---\nBen is CTO. [VERIFIED-PUBLIC]\nInferred. [INFERRED-H]\n"
        result = filter_prose_by_mode(text, DossierMode.CONSTRAINED, entity_lock_score=55)
        assert "### Section 1" in result
        assert "---" in result

    def test_constrained_adds_banner(self):
        text = "Verified fact. [VERIFIED-PUBLIC]\n"
        result = filter_prose_by_mode(text, DossierMode.CONSTRAINED, entity_lock_score=55)
        assert "PARTIAL DOSSIER" in result
        assert "PARTIAL LOCK" in result

    def test_halted_mode_passthrough(self):
        text = "This should pass through unchanged."
        result = filter_prose_by_mode(text, DossierMode.HALTED, entity_lock_score=0)
        assert result == text


# ---------------------------------------------------------------------------
# Failure Report Building
# ---------------------------------------------------------------------------


class TestBuildFailureReport:
    def test_includes_header(self):
        g = EvidenceGraph()
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=0, graph=g, person_name="Ben",
        )
        assert "HALTED" in report
        assert "FAIL-CLOSED" in report

    def test_includes_current_state(self):
        g = EvidenceGraph()
        g.add_meeting_node(source="call", snippet="test")
        g.log_retrieval(query="q1", intent="bio", results=[])
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=0, graph=g, person_name="Ben",
        )
        assert "Entity Lock:           40/100" in report
        assert "Evidence Nodes:        1" in report
        assert "Retrieval Ledger Rows: 1" in report

    def test_includes_ledger_when_present(self):
        g = EvidenceGraph()
        g.log_retrieval(query='"Ben" TED', intent="visibility", results=[])
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=10, graph=g, person_name="Ben",
        )
        assert "RETRIEVAL LEDGER" in report
        assert '"Ben" TED' in report

    def test_includes_visibility_queries_when_missing(self):
        g = EvidenceGraph()
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=0, graph=g, person_name="Ben Titmus",
        )
        assert '"Ben Titmus" TED' in report
        assert '"Ben Titmus" podcast' in report

    def test_includes_what_to_do_next(self):
        g = EvidenceGraph()
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=0, graph=g, person_name="Ben",
        )
        assert "WHAT TO DO NEXT" in report

    def test_includes_what_will_change(self):
        g = EvidenceGraph()
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=0, graph=g, person_name="Ben",
        )
        assert "WHAT WILL CHANGE" in report

    def test_entity_lock_fix_guidance_when_low(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="visibility", results=[{"title": "T"}])
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=55,
            visibility_confidence=10, graph=g, person_name="Ben",
        )
        assert "LinkedIn verified via retrieval" in report
        assert "+30pts" in report
        assert "Meeting confirms identity" in report
        assert "+20pts" in report


# ---------------------------------------------------------------------------
# enforce_fail_closed_gates — public results requirement
# ---------------------------------------------------------------------------


class TestEnforceFailClosedGatesPublicResults:
    def test_fails_without_public_results(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
            has_public_results=False,
        )
        assert not should_output
        assert "NO PUBLIC RETRIEVAL" in message

    def test_passes_with_public_results(self):
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
            has_public_results=True,
        )
        assert should_output

    def test_no_public_results_combined_with_other_failures(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            has_public_results=False,
        )
        assert not should_output
        assert "NO PUBLIC RETRIEVAL" in message
        assert "VISIBILITY SWEEP" in message
        assert "EVIDENCE COVERAGE" in message

    def test_default_has_public_results_true(self):
        """Default parameter value should be True (backward compat)."""
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Test",
        )
        assert should_output


# ---------------------------------------------------------------------------
# Profiler fail-closed prompt rules
# ---------------------------------------------------------------------------


class TestProfilerFailClosedRules:
    def test_system_prompt_has_fail_closed(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "FAIL-CLOSED" in SYSTEM_PROMPT

    def test_system_prompt_bans_generic_phrases(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "BANNED phrases" in SYSTEM_PROMPT
        assert "strategic leader" in SYSTEM_PROMPT
        assert "proven track record" in SYSTEM_PROMPT
        assert "bridges the gap" in SYSTEM_PROMPT

    def test_system_prompt_no_self_contradictions(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "never contradict" in SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Mode A: Meeting-Prep Brief
# ---------------------------------------------------------------------------


class TestMeetingPrepBrief:
    """Tests for Mode A: fast, always-available, no web required."""

    def test_basic_output_structure(self):
        g = EvidenceGraph()
        g.add_meeting_node(source="Q1 Review", snippet="Pipeline risk discussed")
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "# Meeting-Prep Brief: Ben Titmus" in brief
        assert "## 1. What We Know" in brief
        assert "## 2. What To Do Next" in brief
        assert "## 3. Key Risks" in brief
        assert "## 4. Missing Intel" in brief

    def test_includes_meeting_evidence(self):
        g = EvidenceGraph()
        g.add_meeting_node(source="Q1 Review", snippet="Pipeline risk discussed")
        g.add_meeting_node(source="Follow-up", snippet="Budget approved for Q2")
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "Pipeline risk discussed" in brief
        assert "Budget approved for Q2" in brief
        assert "[VERIFIED-MEETING]" in brief

    def test_no_meetings_shows_unknown(self):
        g = EvidenceGraph()
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "[UNKNOWN]" in brief
        assert "No meeting or email history" in brief

    def test_never_requires_serpapi(self):
        """Mode A must NEVER block on SerpAPI or visibility sweep."""
        g = EvidenceGraph()
        # No visibility rows, no public data — should still produce output
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "Meeting-Prep Brief" in brief
        assert len(brief) > 100  # Substantive output

    def test_never_blocks_on_visibility_sweep(self):
        """Mode A produces output regardless of visibility sweep status."""
        g = EvidenceGraph()
        # Zero visibility rows
        assert len(g.get_visibility_ledger_rows()) == 0
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert brief  # Non-empty output
        assert "HALTED" not in brief
        assert "FAIL" not in brief

    def test_includes_profile_data(self):
        g = EvidenceGraph()
        profile = {"company": "Acme Corp", "title": "CTO"}
        brief = build_meeting_prep_brief("Ben Titmus", g, profile_data=profile)
        assert "Acme Corp" in brief
        assert "CTO" in brief

    def test_includes_action_items(self):
        g = EvidenceGraph()
        profile = {"action_items": ["Follow up on proposal", "Send deck"]}
        brief = build_meeting_prep_brief("Ben Titmus", g, profile_data=profile)
        assert "Follow up on proposal" in brief
        assert "Send deck" in brief

    def test_recommends_deep_research(self):
        g = EvidenceGraph()
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "Deep Research" in brief

    def test_tagging_only_meeting_and_inferred(self):
        """Mode A should only use [VERIFIED-MEETING], [INFERRED-L/M], [UNKNOWN]."""
        g = EvidenceGraph()
        g.add_meeting_node(source="call", snippet="Test snippet")
        brief = build_meeting_prep_brief("Ben Titmus", g)
        # Should NOT contain public tags
        assert "[VERIFIED-PUBLIC]" not in brief
        # Should contain meeting-appropriate tags
        assert "[VERIFIED-MEETING]" in brief or "[UNKNOWN]" in brief

    def test_no_public_claims(self):
        """Mode A must not make public claims unless stored as verified."""
        g = EvidenceGraph()
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "[VERIFIED-PUBLIC]" not in brief

    def test_prep_checklist_included(self):
        g = EvidenceGraph()
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "Prep Checklist" in brief
        assert "[ ]" in brief  # Checklist items

    def test_missing_linkedin_in_checklist(self):
        g = EvidenceGraph()
        profile = {}  # No linkedin_url
        brief = build_meeting_prep_brief("Ben Titmus", g, profile_data=profile)
        assert "LinkedIn" in brief

    def test_with_interaction_history(self):
        g = EvidenceGraph()
        profile = {
            "interactions": [
                {"title": "Q1 Review", "date": "2026-01-15", "summary": "Discussed pipeline"}
            ]
        }
        brief = build_meeting_prep_brief("Ben Titmus", g, profile_data=profile)
        assert "Discussed pipeline" in brief or "Q1 Review" in brief

    def test_mode_label(self):
        g = EvidenceGraph()
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "Meeting-Prep" in brief
        assert "internal evidence only" in brief

    def test_risks_with_single_interaction(self):
        g = EvidenceGraph()
        g.add_meeting_node(source="call", snippet="First meeting")
        brief = build_meeting_prep_brief("Ben Titmus", g)
        assert "one prior interaction" in brief.lower() or "limited context" in brief.lower()

    def test_risks_with_no_company(self):
        g = EvidenceGraph()
        profile = {}  # No company
        brief = build_meeting_prep_brief("Ben Titmus", g, profile_data=profile)
        assert "Company not confirmed" in brief


# ---------------------------------------------------------------------------
# Mode B: Deep Research Gate Requirements
# ---------------------------------------------------------------------------


class TestDeepResearchGateRequirements:
    """Tests that Mode B enforces fail-closed gates correctly."""

    def test_mode_b_halts_without_retrieval_ledger(self):
        """Mode B must halt if retrieval ledger is missing."""
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=False,
            has_public_results=False,
        )
        assert mode == DossierMode.HALTED

    def test_mode_b_halts_without_visibility_sweep(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=False,
            has_public_results=True,
        )
        assert mode == DossierMode.HALTED
        assert "VISIBILITY SWEEP NOT EXECUTED" in reason

    def test_mode_b_halts_without_public_results(self):
        mode, reason = determine_dossier_mode(
            entity_lock_score=85,
            visibility_executed=True,
            has_public_results=False,
        )
        assert mode == DossierMode.HALTED
        assert "NO PUBLIC RETRIEVAL" in reason

    def test_mode_b_full_when_all_gates_pass(self):
        mode, _ = determine_dossier_mode(
            entity_lock_score=80,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.FULL

    def test_mode_b_constrained_when_entity_lock_partial(self):
        mode, _ = determine_dossier_mode(
            entity_lock_score=55,
            visibility_executed=True,
            has_public_results=True,
        )
        assert mode == DossierMode.CONSTRAINED


class TestVisibilitySweepLedgerRequirement:
    """Visibility sweep cannot claim 'none found' unless >= 8 queries executed."""

    def test_8_queries_required(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=7,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not should_output
        assert "INSUFFICIENT VISIBILITY QUERIES" in message

    def test_8_queries_passes(self):
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=8,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert should_output

    def test_16_queries_passes(self):
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert should_output

    def test_0_queries_also_fails(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not should_output
        assert "VISIBILITY SWEEP NOT EXECUTED" in message


class TestEntityLockLinkedInEvidence:
    """LinkedIn confirmed must require an evidence node (public snippet/title)."""

    def test_linkedin_url_without_search_results_not_confirmed(self):
        """LinkedIn URL present but no public results → not confirmed, but +10 pts."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={},  # No search results at all
        )
        assert not result.linkedin_confirmed
        assert not result.linkedin_verified_by_retrieval
        assert result.linkedin_url_present
        assert result.score == 10  # Weak internal evidence

    def test_linkedin_url_with_matching_search_result_confirmed(self):
        """LinkedIn URL + matching search result → confirmed with 30 pts."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={
                "linkedin": [
                    {
                        "title": "Ben Titmus - CTO at Acme",
                        "snippet": "View Ben Titmus's profile",
                        "link": "https://linkedin.com/in/bentitmus",
                    }
                ]
            },
        )
        assert result.linkedin_confirmed
        assert result.linkedin_verified_by_retrieval
        assert result.score >= 30

    def test_linkedin_url_with_empty_title_snippet_not_confirmed(self):
        """LinkedIn result with empty title AND snippet → not confirmed, +10 for URL present."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={
                "linkedin": [
                    {
                        "title": "",
                        "snippet": "",
                        "link": "https://linkedin.com/in/bentitmus",
                    }
                ]
            },
        )
        assert not result.linkedin_confirmed
        assert not result.linkedin_verified_by_retrieval
        assert result.linkedin_url_present
        assert result.score == 10  # URL present gives weak +10

    def test_no_linkedin_url_but_search_finds_name(self):
        """No URL but search finds name → partial credit (15 pts, not 30)."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            search_results={
                "linkedin": [
                    {
                        "title": "Ben Titmus - CTO",
                        "snippet": "Profile on LinkedIn",
                        "link": "https://linkedin.com/in/bentitmus",
                    }
                ]
            },
        )
        assert result.name_match
        assert result.score >= 15
        assert not result.linkedin_confirmed  # No URL → not "confirmed"

    def test_evidence_includes_not_verified_message(self):
        """Evidence should include 'not yet verified' message when URL exists but no results."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={"linkedin": []},
        )
        assert not result.linkedin_confirmed
        # Check evidence for the not-yet-verified message
        msgs = [e["signal"] for e in result.evidence]
        assert any("not yet verified" in m for m in msgs)
        # Should still get +10 for URL present
        assert result.score == 10


class TestEntityLockNewWeights:
    """Entity lock scoring uses new weights (100 total)."""

    def test_public_linkedin_worth_30(self):
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={
                "linkedin": [
                    {"title": "Ben Titmus - CTO", "snippet": "Profile", "link": "https://linkedin.com/in/bentitmus"}
                ]
            },
        )
        # Should get exactly 30 pts for LinkedIn
        linkedin_weight = sum(
            e["weight"] for e in result.evidence if "LinkedIn" in e["signal"] and e["weight"] > 0
        )
        assert linkedin_weight == 30

    def test_employer_worth_20(self):
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            search_results={
                "general": [
                    {"title": "Ben Titmus at Acme Corp", "snippet": "Ben Titmus leads Acme Corp"}
                ],
                "news": [
                    {"title": "Acme Corp promotes Ben Titmus", "snippet": "Ben Titmus named CTO"}
                ],
            },
        )
        assert result.employer_match
        assert result.score >= 20

    def test_multiple_domains_worth_20(self):
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme",
            linkedin_url="https://linkedin.com/in/bentitmus",
            search_results={
                "linkedin": [
                    {"title": "Ben Titmus - CTO", "snippet": "View profile", "link": "https://linkedin.com/in/bentitmus"}
                ],
                "general": [
                    {"title": "Ben Titmus at Acme", "snippet": "Ben Titmus is CTO of Acme"}
                ],
                "news": [
                    {"title": "Acme news: Ben Titmus", "snippet": "Ben Titmus announced"}
                ],
            },
        )
        assert result.multiple_sources_agree
        # Should have 20pts for multi-domain (3+ domains)
        multi_weight = sum(
            e["weight"] for e in result.evidence if "domains agree" in e["signal"]
        )
        assert multi_weight >= 20

    def test_meeting_alone_gives_20(self):
        """Meeting data alone gives +20 internal confirmation points."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            has_meeting_data=True,
            search_results={},  # No public results
        )
        # Meeting gives +20 for internal confirmation
        assert result.score == 20
        assert result.meeting_confirmed

    def test_meeting_plus_linkedin_url_gives_30(self):
        """Meeting (+20) + LinkedIn URL present (+10) = 30 without retrieval."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            has_meeting_data=True,
            search_results={},  # No retrieval executed
        )
        assert result.meeting_confirmed
        assert result.linkedin_url_present
        # Meeting (20) + LinkedIn URL present (10) = 30
        assert result.score == 30

    def test_meeting_plus_verified_linkedin_gives_50(self):
        """Meeting (+20) + LinkedIn verified (+30) = 50."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            linkedin_url="https://linkedin.com/in/bentitmus",
            has_meeting_data=True,
            search_results={
                "linkedin": [
                    {"title": "Ben Titmus", "snippet": "Profile", "link": "https://linkedin.com/in/bentitmus"}
                ]
            },
        )
        assert result.meeting_confirmed
        assert result.linkedin_verified_by_retrieval
        # LinkedIn verified (30) + meeting (20) = 50
        assert result.score >= 50

    def test_location_worth_10(self):
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            location="London",
            search_results={
                "general": [
                    {"title": "Ben Titmus", "snippet": "Based in London, Ben Titmus leads"}
                ]
            },
        )
        assert result.location_match
        assert result.score >= 10

    def test_full_lock_achievable(self):
        """Should be possible to reach 100 with all signals."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            title="CTO",
            linkedin_url="https://linkedin.com/in/bentitmus",
            location="London",
            has_meeting_data=True,
            search_results={
                "linkedin": [
                    {"title": "Ben Titmus - CTO at Acme Corp", "snippet": "View Ben Titmus profile", "link": "https://linkedin.com/in/bentitmus"}
                ],
                "general": [
                    {"title": "Ben Titmus at Acme Corp", "snippet": "Ben Titmus, CTO of Acme Corp in London"}
                ],
                "news": [
                    {"title": "Acme Corp: Ben Titmus named CTO", "snippet": "Ben Titmus appointed CTO in London"}
                ],
            },
        )
        assert result.score >= 70
        assert result.linkedin_confirmed
        assert result.employer_match
        assert result.title_match
        assert result.location_match
        assert result.multiple_sources_agree


class TestFailureReportNewWeights:
    """Failure report should reference new scoring weights."""

    def test_includes_new_linkedin_weight(self):
        g = EvidenceGraph()
        # Need at least 1 public result so we get the weight breakdown
        g.log_retrieval(query="q1", intent="visibility", results=[{"title": "T"}])
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=10, graph=g, person_name="Ben",
        )
        assert "+30pts" in report
        assert "+10pts" in report  # LinkedIn URL present (weak)
        assert "+20pts" in report  # Meeting confirms identity

    def test_includes_multiple_domains_weight(self):
        g = EvidenceGraph()
        g.log_retrieval(query="q1", intent="visibility", results=[{"title": "T"}])
        report = build_failure_report(
            mode_reason="FAIL: TEST", entity_lock_score=40,
            visibility_confidence=10, graph=g, person_name="Ben",
        )
        assert "+20pts" in report
        assert "Multiple independent domains" in report


# ---------------------------------------------------------------------------
# REGRESSION PREVENTION — Entity Lock must not collapse to 0
# ---------------------------------------------------------------------------


class TestEntityLockRegressionPrevention:
    """Tests that would have caught the Entity Lock score collapse to 0."""

    def test_linkedin_url_plus_meeting_gives_nonzero(self):
        """REGRESSION: linkedin_url_present + meeting data must not be 0."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Andy Sweet",
            linkedin_url="https://linkedin.com/in/andysweet",
            has_meeting_data=True,
            search_results={},  # No retrieval executed
        )
        # LinkedIn URL present (+10) + meeting (+20) = 30
        assert result.score >= 30
        assert result.score > 0  # The key invariant

    def test_meeting_data_alone_gives_nonzero(self):
        """REGRESSION: meeting data must not give 0 — it's real internal evidence."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Michael Callero",
            has_meeting_data=True,
        )
        assert result.score > 0
        assert result.meeting_confirmed

    def test_linkedin_url_present_gives_nonzero(self):
        """REGRESSION: having a LinkedIn URL must give some points, not 0."""
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Una Fox",
            linkedin_url="https://linkedin.com/in/unafox",
        )
        assert result.score > 0
        assert result.linkedin_url_present

    def test_entity_lock_meaningful_without_retrieval(self):
        """REGRESSION: entity lock should show meaningful breakdown without retrieval.

        Scenario: contact with LinkedIn URL, meeting data, Apollo employer data.
        Score must be > 0 and signals dict must not be all-false.
        """
        from app.brief.qa import score_disambiguation

        result = score_disambiguation(
            name="Ben Titmus",
            company="Acme Corp",
            linkedin_url="https://linkedin.com/in/bentitmus",
            has_meeting_data=True,
            apollo_data={
                "name": "Ben Titmus",
                "title": "CTO",
                "organization": {"name": "Acme Corp"},
            },
        )
        # LinkedIn URL (+10) + meeting (+20) + Apollo employer (+10) = 40
        assert result.score >= 40
        assert result.linkedin_url_present
        assert result.meeting_confirmed
        assert result.employer_match


# ---------------------------------------------------------------------------
# Strategic Model v2: Factual Coverage + Strategic Sources
# ---------------------------------------------------------------------------


class TestFactualCoverage:
    """Test compute_factual_coverage_from_text excludes sections 9-11."""

    def test_factual_only_100_when_all_tagged(self):
        from app.brief.evidence_graph import compute_factual_coverage_from_text
        text = (
            "### 1. Executive Summary\n"
            "Jane is VP of Engineering at Acme Corp. [VERIFIED-PDF]\n"
            "She manages a team of 50 engineers. [VERIFIED-MEETING]\n"
            "### 9. Structural Incentive & Power Model\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]\n"
            "Untagged strategic reasoning that should be excluded from factual count.\n"
            "### 12. Primary Source Index\n"
            "LinkedIn profile used as primary source. [VERIFIED-PDF]\n"
        )
        pct = compute_factual_coverage_from_text(text)
        assert pct == 100.0

    def test_factual_ignores_section_9_10_11(self):
        from app.brief.evidence_graph import compute_factual_coverage_from_text
        text = (
            "### 1. Executive Summary\n"
            "Jane is VP of Engineering at Acme Corp. [VERIFIED-PDF]\n"
            "Some untagged claim about her background.\n"
            "### 9. Structural Incentive & Power Model\n"
            "Untagged strategic reasoning line one.\n"
            "Untagged strategic reasoning line two.\n"
            "### 10. Competitive Positioning Context\n"
            "Untagged competitive analysis.\n"
            "### 11. How to Win This Decision-Maker\n"
            "Untagged win strategy.\n"
            "### 12. Primary Source Index\n"
            "Source reference. [VERIFIED-PUBLIC]\n"
        )
        pct = compute_factual_coverage_from_text(text)
        # Sections 1 and 12 have 3 lines: 2 tagged, 1 untagged
        assert pct < 100.0
        assert pct > 0.0

    def test_factual_coverage_empty_text(self):
        from app.brief.evidence_graph import compute_factual_coverage_from_text
        assert compute_factual_coverage_from_text("") == 100.0


class TestStrategicSourcesPresent:
    """Test check_strategic_sources_present for Derived from headers."""

    def test_all_present(self):
        from app.brief.evidence_graph import check_strategic_sources_present
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]\n"
            "Analysis content here.\n"
            "### 10. Competitive Positioning Context\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PUBLIC + INFERRED-H]\n"
            "Analysis content here.\n"
            "### 11. How to Win This Decision-Maker\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-MEETING + VERIFIED-PDF]\n"
            "Analysis content here.\n"
        )
        ok, missing = check_strategic_sources_present(text)
        assert ok is True
        assert missing == []

    def test_missing_section_10(self):
        from app.brief.evidence_graph import check_strategic_sources_present
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]\n"
            "### 10. Competitive Positioning Context\n"
            "Some analysis without derived header.\n"
            "### 11. How to Win This Decision-Maker\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-MEETING + INFERRED-H]\n"
        )
        ok, missing = check_strategic_sources_present(text)
        assert ok is False
        assert len(missing) == 1
        assert "Section 10" in missing[0]

    def test_all_missing(self):
        from app.brief.evidence_graph import check_strategic_sources_present
        text = (
            "### 1. Executive Summary\n"
            "Content only in section 1.\n"
        )
        ok, missing = check_strategic_sources_present(text)
        assert ok is False
        assert len(missing) == 3


# ---------------------------------------------------------------------------
# v4 Hardening: Canonical Field Extraction + Validation
# ---------------------------------------------------------------------------


class TestCanonicalFieldExtraction:
    """Test extract_canonical_fields from dossier preamble."""

    def test_extracts_all_three_fields(self):
        from app.brief.evidence_graph import extract_canonical_fields
        text = (
            "**Canonical Company**: Acme Corp — [VERIFIED-PDF]\n"
            "**Canonical Title**: CTO — [VERIFIED-PUBLIC]\n"
            "**Canonical Location**: London, UK — [VERIFIED-MEETING]\n"
        )
        fields = extract_canonical_fields(text)
        assert len(fields) == 3
        assert fields["company"]["value"] == "Acme Corp"
        assert fields["company"]["tag"] == "VERIFIED-PDF"
        assert fields["title"]["value"] == "CTO"
        assert fields["title"]["tag"] == "VERIFIED-PUBLIC"
        assert fields["location"]["value"] == "London, UK"
        assert fields["location"]["tag"] == "VERIFIED-MEETING"

    def test_unverified_field(self):
        from app.brief.evidence_graph import extract_canonical_fields
        text = (
            "**Canonical Company**: Acme Corp — [VERIFIED-PDF]\n"
            "**Canonical Title**: UNVERIFIED — [UNKNOWN]\n"
            "**Canonical Location**: UNVERIFIED — [UNKNOWN]\n"
        )
        fields = extract_canonical_fields(text)
        assert fields["title"]["value"] == "UNVERIFIED"
        assert fields["title"]["tag"] == "UNKNOWN"

    def test_empty_text(self):
        from app.brief.evidence_graph import extract_canonical_fields
        assert extract_canonical_fields("") == {}

    def test_partial_fields(self):
        from app.brief.evidence_graph import extract_canonical_fields
        text = "**Canonical Company**: Acme Corp — [VERIFIED-PDF]\n"
        fields = extract_canonical_fields(text)
        assert len(fields) == 1
        assert "company" in fields


class TestCanonicalFieldValidation:
    """Test validate_canonical_fields rejects non-VERIFIED sources."""

    def test_all_verified_no_violations(self):
        from app.brief.evidence_graph import validate_canonical_fields
        canonical = {
            "company": {"value": "Acme Corp", "tag": "VERIFIED-PDF"},
            "title": {"value": "CTO", "tag": "VERIFIED-PUBLIC"},
            "location": {"value": "London", "tag": "VERIFIED-MEETING"},
        }
        violations = validate_canonical_fields(canonical)
        assert violations == []

    def test_inferred_tag_produces_violation(self):
        from app.brief.evidence_graph import validate_canonical_fields
        canonical = {
            "company": {"value": "Acme Corp", "tag": "INFERRED-H"},
        }
        violations = validate_canonical_fields(canonical)
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "CANONICAL_FIELD_NOT_VERIFIED"
        assert "INFERRED-H" in violations[0]["message"]

    def test_unverified_unknown_no_violation(self):
        from app.brief.evidence_graph import validate_canonical_fields
        canonical = {
            "title": {"value": "UNVERIFIED", "tag": "UNKNOWN"},
        }
        violations = validate_canonical_fields(canonical)
        assert violations == []


# ---------------------------------------------------------------------------
# v4 Hardening: Visibility Artifact Table Validation
# ---------------------------------------------------------------------------


class TestVisibilityArtifactTable:
    """Test validate_visibility_artifact_table for section 5."""

    def test_valid_table_with_urls(self):
        from app.brief.evidence_graph import validate_visibility_artifact_table
        text = (
            "### 5. Public Visibility\n"
            "| # | Type | Title | URL | Date | Signal |\n"
            "|---|------|-------|-----|------|--------|\n"
            "| 1 | TED | Talk 1 | https://ted.com/talk1 | 2025-01-01 | AI |\n"
            "| 2 | Keynote | Talk 2 | https://conf.com/talk2 | 2025-02-01 | Cloud |\n"
            "| 3 | Podcast | Ep 1 | https://pod.com/ep1 | 2025-03-01 | Data |\n"
            "| 4 | Panel | Panel 1 | https://panel.com/p1 | 2025-04-01 | ML |\n"
            "| 5 | Webinar | Web 1 | https://web.com/w1 | 2025-05-01 | DevOps |\n"
            "### 6. Quantified Claims\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count >= 5
        assert violations == []

    def test_zero_artifacts_declared(self):
        from app.brief.evidence_graph import validate_visibility_artifact_table
        text = (
            "### 5. Public Visibility\n"
            "**total_visibility_artifacts=0**\n"
            "No public speaking found.\n"
            "### 6. Quantified Claims\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0
        assert violations == []

    def test_missing_table_produces_violation(self):
        from app.brief.evidence_graph import validate_visibility_artifact_table
        text = (
            "### 5. Public Visibility\n"
            "Some text about visibility without a table.\n"
            "### 6. Quantified Claims\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "VISIBILITY_TABLE_MISSING"

    def test_missing_section_5(self):
        from app.brief.evidence_graph import validate_visibility_artifact_table
        text = "### 1. Executive Summary\nSome content.\n"
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "VISIBILITY_SECTION_MISSING"


# ---------------------------------------------------------------------------
# v4 Hardening: Reasoning Anchor Validation
# ---------------------------------------------------------------------------


class TestReasoningAnchorValidation:
    """Test validate_reasoning_anchors for sections 9-11."""

    def test_all_sections_have_anchors(self):
        from app.brief.evidence_graph import validate_reasoning_anchors
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "- Anchor 1: Revenue target of $50M — VERIFIED-PDF (Section 3)\n"
            "- Anchor 2: Board reporting line — VERIFIED-MEETING (Section 4)\n"
            "- Anchor 3: Growth mandate — VERIFIED-PUBLIC (Section 8)\n"
            "### 10. Competitive Positioning Context\n"
            "- Anchor 1: Acme competes with BigCo — VERIFIED-PUBLIC (Section 5)\n"
            "- Anchor 2: AI maturity stage 2 — INFERRED-H (Section 6)\n"
            "- Anchor 3: Consulting mix 60/40 — VERIFIED-PDF (Section 3)\n"
            "### 11. How to Win This Decision-Maker\n"
            "- Anchor 1: Measured on revenue growth — VERIFIED-MEETING (Section 4)\n"
            "- Anchor 2: Risk-averse decision style — VERIFIED-PUBLIC (Section 7)\n"
            "- Anchor 3: Budget authority $5M — INFERRED-H (Section 9)\n"
        )
        counts, violations = validate_reasoning_anchors(text)
        assert counts == {9: 3, 10: 3, 11: 3}
        assert violations == []

    def test_insufficient_anchors_produces_violation(self):
        from app.brief.evidence_graph import validate_reasoning_anchors
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "- Anchor 1: Revenue target — VERIFIED-PDF (Section 3)\n"
            "Some analysis without enough anchors.\n"
            "### 10. Competitive Positioning Context\n"
            "No anchors at all.\n"
            "### 11. How to Win This Decision-Maker\n"
            "- Anchor 1: Measured on growth — VERIFIED-MEETING (Section 4)\n"
            "- Anchor 2: Risk-averse — VERIFIED-PUBLIC (Section 7)\n"
            "- Anchor 3: Budget $5M — INFERRED-H (Section 9)\n"
        )
        counts, violations = validate_reasoning_anchors(text)
        assert counts[9] == 1
        assert counts[10] == 0
        assert counts[11] == 3
        assert len(violations) == 2  # section 9 and 10

    def test_constrained_declaration_no_violation(self):
        from app.brief.evidence_graph import validate_reasoning_anchors
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "**Insufficient evidence for full strategic model — downgrading to CONSTRAINED.**\n"
            "### 10. Competitive Positioning Context\n"
            "- Anchor 1: Competitor analysis — VERIFIED-PUBLIC (Section 5)\n"
            "- Anchor 2: AI maturity — INFERRED-H (Section 6)\n"
            "- Anchor 3: Market position — VERIFIED-PDF (Section 3)\n"
            "### 11. How to Win This Decision-Maker\n"
            "**Insufficient evidence for full win strategy — downgrading to CONSTRAINED.**\n"
        )
        counts, violations = validate_reasoning_anchors(text)
        assert counts[9] == -1  # declared constrained
        assert counts[11] == -1
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# v4 Hardening: Inference Language Validation
# ---------------------------------------------------------------------------


class TestInferenceLanguageValidation:
    """Test validate_inference_language for hedge words without derivation."""

    def test_hedge_with_derivation_ok(self):
        from app.brief.evidence_graph import validate_inference_language
        text = (
            "### 1. Executive Summary\n"
            "He likely prioritizes revenue growth (Derived from: VERIFIED-PDF role as CRO "
            "+ VERIFIED-MEETING discussion of pipeline targets).\n"
        )
        violations = validate_inference_language(text)
        assert violations == []

    def test_hedge_without_derivation_flagged(self):
        from app.brief.evidence_graph import validate_inference_language
        text = (
            "### 1. Executive Summary\n"
            "He likely prioritizes revenue growth based on his background.\n"
        )
        violations = validate_inference_language(text)
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "HEDGE_WITHOUT_DERIVATION"
        assert "likely" in violations[0]["message"]

    def test_hedge_with_evidence_tag_ok(self):
        from app.brief.evidence_graph import validate_inference_language
        text = (
            "### 3. Career Timeline\n"
            "He may have led the AI team during this period. [INFERRED-H]\n"
        )
        violations = validate_inference_language(text)
        assert violations == []

    def test_strategic_sections_exempt(self):
        from app.brief.evidence_graph import validate_inference_language
        text = (
            "### 9. Structural Incentive & Power Model\n"
            "He likely faces revenue pressure from the board.\n"
            "### 10. Competitive Positioning Context\n"
            "Acme may be losing market share to BigCo.\n"
        )
        violations = validate_inference_language(text)
        assert violations == []

    def test_multiple_violations(self):
        from app.brief.evidence_graph import validate_inference_language
        text = (
            "### 2. Identity & Disambiguation\n"
            "This suggests he may be the same person mentioned in press.\n"
            "Evidence indicates a strong leadership background.\n"
        )
        violations = validate_inference_language(text)
        assert len(violations) == 2


# ---------------------------------------------------------------------------
# v4 Hardening: FailClosedResult and enforce_fail_closed_gates_v4
# ---------------------------------------------------------------------------


class TestFailClosedResultV4:
    """Test the structured FailClosedResult from enforce_fail_closed_gates_v4."""

    def test_all_pass_returns_empty_failures(self):
        from app.brief.qa import enforce_fail_closed_gates_v4
        result = enforce_fail_closed_gates_v4(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert result.should_output
        assert result.message == ""
        assert result.failing_gate_names == []
        assert result.failures_by_section == {}

    def test_visibility_failure_structured(self):
        from app.brief.qa import enforce_fail_closed_gates_v4
        result = enforce_fail_closed_gates_v4(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not result.should_output
        assert "VISIBILITY_SWEEP" in result.failing_gate_names
        assert "visibility" in result.failures_by_section
        assert result.failures_by_section["visibility"][0]["rule_id"] == "VISIBILITY_NOT_EXECUTED"

    def test_coverage_failure_structured(self):
        from app.brief.qa import enforce_fail_closed_gates_v4
        result = enforce_fail_closed_gates_v4(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            web_results_count=15,
        )
        assert not result.should_output
        assert "EVIDENCE_COVERAGE" in result.failing_gate_names
        assert "evidence_coverage" in result.failures_by_section
        violations = result.failures_by_section["evidence_coverage"]
        assert violations[0]["rule_id"] == "COVERAGE_BELOW_THRESHOLD"

    def test_multiple_failures_structured(self):
        from app.brief.qa import enforce_fail_closed_gates_v4
        result = enforce_fail_closed_gates_v4(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=0,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
            web_results_count=15,
            has_public_results=False,
        )
        assert not result.should_output
        assert len(result.failing_gate_names) == 3
        assert "PUBLIC_RESULTS" in result.failing_gate_names
        assert "VISIBILITY_SWEEP" in result.failing_gate_names
        assert "EVIDENCE_COVERAGE" in result.failing_gate_names

    def test_insufficient_queries_structured(self):
        from app.brief.qa import enforce_fail_closed_gates_v4
        result = enforce_fail_closed_gates_v4(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=5,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not result.should_output
        assert "VISIBILITY_SWEEP" in result.failing_gate_names
        violations = result.failures_by_section["visibility"]
        assert violations[0]["rule_id"] == "INSUFFICIENT_VISIBILITY_QUERIES"


# ---------------------------------------------------------------------------
# v2: Coverage below 85% triggers failure (flat threshold)
# ---------------------------------------------------------------------------


class TestFlatCoverageThreshold:
    """Coverage must be >= 85% regardless of web result count."""

    def test_67_pct_fails(self):
        """67% coverage should fail — no partial pass allowed."""
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=80,
            visibility_ledger_count=15,
            evidence_coverage_pct=67.0,
            person_name="Test Subject",
            web_results_count=20,
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message
        assert "85" in message

    def test_84_pct_fails(self):
        """84.9% should fail — no partial pass at 67%."""
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=80,
            visibility_ledger_count=15,
            evidence_coverage_pct=84.9,
            person_name="Test",
        )
        assert not should_output

    def test_85_pct_passes(self):
        """Exactly 85% passes."""
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=80,
            visibility_ledger_count=15,
            evidence_coverage_pct=85.0,
            person_name="Test",
        )
        assert should_output

    def test_90_pct_passes(self):
        """90% passes."""
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=80,
            visibility_ledger_count=15,
            evidence_coverage_pct=90.0,
            person_name="Test",
        )
        assert should_output

    def test_qa_report_includes_counts(self):
        """QA report includes coverage %, uncited count, total count."""
        from app.brief.qa import check_evidence_coverage
        text = (
            "Ben Titmus is CTO at Acme Corp and leads their platform division [VERIFIED-PUBLIC]\n"
            "He has extensive engineering background and manages fifty engineers [VERIFIED-MEETING]\n"
            "He is a great leader and visionary thinker in the enterprise space\n"
            "Revenue grew significantly to over thirty percent last year across all divisions\n"
        )
        result = check_evidence_coverage(text)
        assert result.total_substantive > 0
        assert result.tagged_count > 0
        assert len(result.untagged_sentences) > 0


# ---------------------------------------------------------------------------
# v2: INFERRED-H without 2 anchors fails
# ---------------------------------------------------------------------------


class TestInferredHAnchoring:
    """INFERRED-H must cite >= 2 upstream anchors."""

    def test_two_anchors_passes(self):
        text = (
            "Ben likely prioritizes revenue growth based on meeting "
            "transcript and LinkedIn PDF evidence. [INFERRED-H]"
        )
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 1
        assert result.with_upstream == 1
        assert result.passes

    def test_zero_anchors_fails(self):
        text = "Ben is a strategic thinker. [INFERRED-H]"
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 1
        assert len(result.without_upstream) == 1
        assert not result.passes

    def test_one_anchor_insufficient(self):
        # Only one anchor pattern ("meeting") — need 2+
        text = "Ben owns the engineering roadmap per the meeting notes. [INFERRED-H]"
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 1
        assert len(result.insufficient_anchors) == 1
        assert result.insufficient_anchors[0]["anchor_count"] == 1
        assert not result.passes

    def test_multiple_claims_mixed(self):
        text = (
            "Ben owns budget per the meeting and LinkedIn evidence. [INFERRED-H]\n"
            "He is innovative. [INFERRED-H]\n"
            "He has delivery pressure per the transcript only. [INFERRED-H]\n"
        )
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 3
        assert result.with_upstream == 1  # first has meeting + LinkedIn
        assert len(result.without_upstream) == 1  # second has 0 anchors
        assert len(result.insufficient_anchors) == 1  # third has 1 anchor
        assert not result.passes

    def test_no_inferred_h_passes_trivially(self):
        text = "Ben is CTO at Acme. [VERIFIED-PUBLIC]"
        result = audit_inferred_h(text)
        assert result.total_inferred_h == 0
        assert result.passes


# ---------------------------------------------------------------------------
# v2: <5 artifacts collapses visibility to 0
# ---------------------------------------------------------------------------


class TestVisibilityInflation:
    """Fewer than 5 artifacts collapses effective count to 0."""

    def test_zero_artifacts_with_declaration(self):
        text = (
            "### 5. Public Visibility\n\n"
            "**total_visibility_artifacts=0**\n"
            "No public speaking found.\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0
        assert len(violations) == 0  # Properly declared

    def test_three_artifacts_collapses_to_zero(self):
        text = (
            "### 5. Public Visibility\n\n"
            "| # | Type | Title | URL | Date | Signal |\n"
            "|---|------|-------|-----|------|--------|\n"
            "| 1 | Podcast | AI Talk | https://example.com/1 | 2025 | Signal |\n"
            "| 2 | Keynote | Tech Summit | https://example.com/2 | 2025 | Signal |\n"
            "| 3 | Panel | Governance | https://example.com/3 | 2025 | Signal |\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0  # Collapsed to 0
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "VISIBILITY_INFLATION_PREVENTED"

    def test_four_artifacts_collapses_to_zero(self):
        text = (
            "### 5. Public Visibility\n\n"
            "| # | Type | Title | URL | Date | Signal |\n"
            "|---|------|-------|-----|------|--------|\n"
            "| 1 | Podcast | A | https://example.com/1 | 2025 | S |\n"
            "| 2 | Keynote | B | https://example.com/2 | 2025 | S |\n"
            "| 3 | Panel | C | https://example.com/3 | 2025 | S |\n"
            "| 4 | Webinar | D | https://example.com/4 | 2025 | S |\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0  # Still < 5
        assert violations[0]["rule_id"] == "VISIBILITY_INFLATION_PREVENTED"

    def test_five_artifacts_passes(self):
        text = (
            "### 5. Public Visibility\n\n"
            "| # | Type | Title | URL | Date | Signal |\n"
            "|---|------|-------|-----|------|--------|\n"
            "| 1 | Podcast | A | https://example.com/1 | 2025 | S |\n"
            "| 2 | Keynote | B | https://example.com/2 | 2025 | S |\n"
            "| 3 | Panel | C | https://example.com/3 | 2025 | S |\n"
            "| 4 | Webinar | D | https://example.com/4 | 2025 | S |\n"
            "| 5 | TEDx | E | https://example.com/5 | 2025 | S |\n"
        )
        count, violations = validate_visibility_artifact_table(text)
        assert count == 5
        assert len(violations) == 0

    def test_missing_section_5(self):
        text = "### 4. Public Statements\n\nContent here.\n"
        count, violations = validate_visibility_artifact_table(text)
        assert count == 0
        assert violations[0]["rule_id"] == "VISIBILITY_SECTION_MISSING"


# ---------------------------------------------------------------------------
# v2: Inflation phrases without anchors fail
# ---------------------------------------------------------------------------


class TestNarrativeInflation:
    """Banned inflation phrases require >= 2 verified anchors."""

    def test_inflation_phrase_without_anchors_fails(self):
        text = "### 1. Executive Summary\n\nBen is an emerging leader in AI.\n"
        violations = validate_narrative_inflation(text)
        assert len(violations) == 1
        assert violations[0]["rule_id"] == "NARRATIVE_INFLATION"
        assert "emerging leader" in violations[0]["phrase"]

    def test_inflation_phrase_with_one_anchor_fails(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is positioned as a thought leader. [VERIFIED-PUBLIC]\n"
        )
        violations = validate_narrative_inflation(text)
        assert len(violations) == 1  # Only 1 anchor, need 2

    def test_inflation_phrase_with_two_anchors_passes(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is positioned as a leader per [VERIFIED-PUBLIC] "
            "and [VERIFIED-MEETING] evidence.\n"
        )
        violations = validate_narrative_inflation(text)
        assert len(violations) == 0

    def test_all_inflation_phrases_detected(self):
        lines = ["### 1. Executive Summary\n"]
        for phrase in NARRATIVE_INFLATION_PHRASES:
            lines.append(f"This person is {phrase} in their field.\n")
        text = "\n".join(lines)
        violations = validate_narrative_inflation(text)
        assert len(violations) == len(NARRATIVE_INFLATION_PHRASES)

    def test_no_inflation_phrases_passes(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is CTO at Acme Corp. [VERIFIED-PUBLIC]\n"
            "He manages a team of 50 engineers. [VERIFIED-MEETING]\n"
        )
        violations = validate_narrative_inflation(text)
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# v2: Pressure dimension without evidence becomes UNKNOWN
# ---------------------------------------------------------------------------


class TestPressureEvidence:
    """Pressure must require explicit evidence, not just topic emphasis."""

    def test_signal_only_pressure_fails(self):
        text = (
            "### 8. Structural Pressure Model\n\n"
            "| Dimension | Level | Why |\n"
            "|---|---|---|\n"
            "| Revenue Pressure | High | talks about revenue growth frequently |\n"
        )
        violations = validate_pressure_evidence(text)
        assert len(violations) >= 1
        assert violations[0]["rule_id"] == "PRESSURE_FROM_SIGNAL_ONLY"

    def test_explicit_evidence_passes(self):
        text = (
            "### 8. Structural Pressure Model\n\n"
            "| Dimension | Level | Why |\n"
            "|---|---|---|\n"
            "| Revenue Pressure | High | revenue target of $50M, P&L ownership |\n"
        )
        violations = validate_pressure_evidence(text)
        # Has explicit evidence (revenue target, P&L) — passes
        assert len(violations) == 0

    def test_only_checks_section_8(self):
        text = (
            "### 7. Rhetorical Patterns\n\n"
            "| Dimension | Level | Why |\n"
            "|---|---|---|\n"
            "| Revenue Pressure | High | talks about revenue |\n"
            "\n### 9. Structural Incentive\n\n"
            "Content about pressures.\n"
        )
        violations = validate_pressure_evidence(text)
        assert len(violations) == 0  # Only section 8 is checked

    def test_pressure_claim_with_signal_only_warns(self):
        text = (
            "### 8. Structural Pressure Model\n\n"
            "Ben has delivery pressure because he frequently discusses "
            "deadlines and mentions project timelines.\n"
        )
        violations = validate_pressure_evidence(text)
        # "delivery pressure" + "mentions" (signal) without evidence anchor
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# v2: QA report includes all new fields
# ---------------------------------------------------------------------------


class TestQAReportV2Fields:
    """QA report must include v2 fields."""

    def test_qa_report_has_final_gate_status(self):
        from app.brief.qa import generate_qa_report
        report = generate_qa_report(
            dossier_text="Ben is CTO at Acme. [VERIFIED-PUBLIC]\n" * 20,
            person_name="Ben Titmus",
        )
        assert report.final_gate_status in ("PASS", "FAIL-CLOSED")

    def test_qa_report_passes_all_includes_inferred_h(self):
        from app.brief.qa import QAReport, InferredHAudit
        report = QAReport()
        report.inferred_h_audit = InferredHAudit(
            total_inferred_h=2, with_upstream=1,
            without_upstream=[{"sentence": "test", "line": 1, "anchor_count": 0}],
        )
        # INFERRED-H audit fails → passes_all should be False
        assert not report.passes_all
        assert report.final_gate_status == "FAIL-CLOSED"

    def test_qa_report_passes_all_includes_inflation(self):
        from app.brief.qa import QAReport
        report = QAReport()
        report.narrative_inflation_violations = [
            {"rule_id": "NARRATIVE_INFLATION", "phrase": "emerging leader"}
        ]
        assert not report.passes_all
        assert report.final_gate_status == "FAIL-CLOSED"

    def test_qa_report_markdown_includes_v2_sections(self):
        from app.brief.qa import generate_qa_report, render_qa_report_markdown
        report = generate_qa_report(
            dossier_text="Ben is CTO at Acme. [VERIFIED-PUBLIC]\n" * 20,
            person_name="Ben",
        )
        report.visibility_artifact_count = 7
        report.narrative_inflation_violations = [
            {"rule_id": "NARRATIVE_INFLATION", "line": "5", "phrase": "emerging leader",
             "message": "test violation"},
        ]
        md = render_qa_report_markdown(report)
        assert "Visibility Artifacts" in md
        assert "Narrative Inflation" in md
        assert "Final Gate Status" in md
        assert "Unsupported Sentence Count" in md


# ---------------------------------------------------------------------------
# v2: Auto-prune uncited factual lines
# ---------------------------------------------------------------------------


class TestPruneUncitedFactualLines:
    """Auto-prune removes uncited substantive lines in factual sections."""

    def test_keeps_tagged_lines(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is CTO at Acme Corp and leads platform. [VERIFIED-PUBLIC]\n"
            "He manages the engineering team of fifty. [VERIFIED-MEETING]\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 0
        assert "[VERIFIED-PUBLIC]" in pruned
        assert "[VERIFIED-MEETING]" in pruned

    def test_removes_uncited_factual_lines(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is CTO at Acme Corp and leads platform. [VERIFIED-PUBLIC]\n"
            "He is a strategic leader with deep expertise in modern platform engineering.\n"
            "Revenue grew significantly under his leadership last year in his division.\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 2
        assert "[VERIFIED-PUBLIC]" in pruned
        assert "strategic leader" not in pruned
        assert "Revenue grew" not in pruned

    def test_preserves_strategic_sections(self):
        text = (
            "### 9. Structural Incentive & Power Model\n\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]\n"
            "Ben operates from a position of organizational strength.\n"
            "His mandate is revenue growth as evidenced by his track record.\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 0  # Strategic sections untouched
        assert "organizational strength" in pruned
        assert "mandate is revenue" in pruned

    def test_preserves_headers_and_tables(self):
        text = (
            "### 3. Career Timeline\n\n"
            "| Company | Title | Dates | Tag |\n"
            "|---------|-------|-------|-----|\n"
            "| Acme | CTO | 2020-present | [VERIFIED-PDF] |\n"
            "### 4. Public Statements\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 0
        assert "### 3." in pruned
        assert "### 4." in pruned
        assert "| Acme |" in pruned

    def test_preserves_gap_phrases(self):
        text = (
            "### 5. Public Visibility\n\n"
            "No evidence available for public speaking appearances in any search sweep.\n"
            "No verified external speaking footprint located for this individual.\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 0
        assert "No evidence available" in pruned

    def test_improves_coverage_after_prune(self):
        """After pruning, factual coverage should increase."""
        from app.brief.evidence_graph import compute_factual_coverage_from_text
        text = (
            "### 1. Executive Summary\n\n"
            "Ben is CTO at Acme Corp platform division. [VERIFIED-PUBLIC]\n"
            "He manages the global engineering team of fifty. [VERIFIED-MEETING]\n"
            "He is a strategic thinker with broad expertise in enterprise tech.\n"
            "His leadership style focuses on outcomes and accountability.\n"
            "\n### 9. Structural Incentive\n\n"
            "[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]\n"
            "Analysis of incentive structure and power dynamics.\n"
        )
        coverage_before = compute_factual_coverage_from_text(text)
        pruned, removed = prune_uncited_factual_lines(text)
        coverage_after = compute_factual_coverage_from_text(pruned)
        assert removed == 2
        assert coverage_after > coverage_before

    def test_preserves_blockquotes(self):
        text = (
            "### 4. Public Statements\n\n"
            "> \"We focus on outcomes not process\" — Ben, 2025 interview\n"
            "This is an uncited commentary line about communication style.\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 1
        assert "We focus on outcomes" in pruned
        assert "uncited commentary" not in pruned

    def test_preserves_short_lines(self):
        text = (
            "### 1. Executive Summary\n\n"
            "Short.\n"
            "Ben is CTO at Acme Corp and leads platform. [VERIFIED-PUBLIC]\n"
        )
        pruned, removed = prune_uncited_factual_lines(text)
        assert removed == 0
        assert "Short." in pruned
