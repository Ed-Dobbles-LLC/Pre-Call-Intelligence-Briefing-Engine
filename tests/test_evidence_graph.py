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
    run_fail_closed_gates,
)
from app.brief.qa import enforce_fail_closed_gates
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

    def test_partial_at_69(self):
        result = check_entity_lock_gate(69)
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
        assert ENTITY_LOCK_THRESHOLD == 70

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
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=50.0,
            person_name="Ben Titmus",
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
        )
        assert not should_output
        assert "Entity Lock: 40/100" in message
        assert "NOT LOCKED" in message

    def test_passes_at_exact_thresholds(self):
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=70,
            visibility_ledger_count=12,
            evidence_coverage_pct=85.0,
            person_name="Test",
        )
        assert should_output

    def test_fails_just_below_coverage(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=16,
            evidence_coverage_pct=84.9,
            person_name="Test",
        )
        assert not should_output
        assert "EVIDENCE COVERAGE" in message


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

    def test_system_prompt_has_genericness_linter(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "GENERICNESS LINTER" in SYSTEM_PROMPT

    def test_system_prompt_bans_consultative_tempo(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "consultative tempo" in SYSTEM_PROMPT

    def test_system_prompt_bans_roi_focused(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "ROI-focused" in SYSTEM_PROMPT

    def test_system_prompt_bans_human_centered(self):
        from app.brief.profiler import SYSTEM_PROMPT
        assert "human-centered" in SYSTEM_PROMPT

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
    """Visibility sweep cannot claim 'none found' unless >= 12 queries executed."""

    def test_12_queries_required(self):
        should_output, message = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=11,
            evidence_coverage_pct=92.0,
            person_name="Ben Titmus",
        )
        assert not should_output
        assert "INSUFFICIENT VISIBILITY QUERIES" in message

    def test_12_queries_passes(self):
        should_output, _ = enforce_fail_closed_gates(
            dossier_text="Test",
            entity_lock_score=85,
            visibility_ledger_count=12,
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
