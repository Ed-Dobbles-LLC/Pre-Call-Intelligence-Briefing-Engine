"""Tests for the Decision Leverage Engine v1.

Required test cases:
1. test_decision_utility_scoring_penalizes_generic_claims
2. test_decision_utility_scoring_rewards_specific_anchored_claims
3. test_meeting_moves_have_evidence_refs
4. test_decision_grade_gate_fails_when_insufficient_high_utility_claims
5. test_executive_brief_max_claims_and_structure
"""

from __future__ import annotations

from app.brief.decision_leverage import (
    EXECUTIVE_BRIEF_THRESHOLD,
    build_executive_brief,
    compute_decision_grade,
    extract_claims_from_dossier,
    filter_claims_for_brief,
    generate_meeting_moves,
    score_claim,
)
from app.models import (
    DecisionGradeGateStatus,
    ExecutiveBrief,
    MeetingMove,
    MoveType,
    ScoredClaim,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dossier_with_claims(claims: list[tuple[str, str]], section: str = "1") -> str:
    """Build a minimal dossier text with tagged claim lines.

    Each tuple is (claim_text, evidence_tag).
    """
    lines = [f"### {section}. Test Section"]
    for text, tag in claims:
        lines.append(f"{text} [{tag}]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 1. Generic claims are penalized
# ---------------------------------------------------------------------------


class TestGenericClaimPenalty:
    """Generic claims with no anchors should score below 55."""

    def test_decision_utility_scoring_penalizes_generic_claims(self):
        """A claim full of generic filler and no anchors must score < 55."""
        scored = score_claim(
            claim_text=(
                "He is a strategic leader with a proven track record "
                "of driving transformative results."
            ),
            claim_tag="UNKNOWN",
            anchors=[],
            section_context="",
        )
        assert scored.decision_utility_score < 55, (
            f"Generic claim scored {scored.decision_utility_score}, expected < 55"
        )

    def test_generic_claim_with_single_weak_anchor_still_low(self):
        """A generic claim with a single INFERRED-L anchor stays low."""
        scored = score_claim(
            claim_text="She is passionate about cutting-edge innovation.",
            claim_tag="INFERRED-L",
            anchors=["INFERRED-L"],
            section_context="",
        )
        assert scored.decision_utility_score < 55

    def test_pure_filler_scores_near_zero(self):
        """Pure filler with zero specificity/behavior signals."""
        scored = score_claim(
            claim_text="A visionary thought leader passionate about holistic synergies.",
            claim_tag="UNKNOWN",
            anchors=[],
            section_context="",
        )
        assert scored.decision_utility_score <= 15


# ---------------------------------------------------------------------------
# 2. Specific anchored claims are rewarded
# ---------------------------------------------------------------------------


class TestSpecificAnchoredClaims:
    """Specific claims with strong anchors should score >= 70."""

    def test_decision_utility_scoring_rewards_specific_anchored_claims(self):
        """A specific claim with strong anchors and behavior keywords scores >= 70."""
        scored = score_claim(
            claim_text=(
                "Jane Smith approved a $2.5M budget for the Q3 2025 "
                "procurement initiative with a mandate to evaluate vendors "
                "and reports to the CEO. She is the champion and sponsor "
                "who decides on priority spend and risk appetite, "
                "responsible for OKR targets."
            ),
            claim_tag="VERIFIED-MEETING",
            anchors=["VERIFIED-MEETING", "VERIFIED-PDF"],
            section_context="Section 3: Career Timeline",
        )
        assert scored.decision_utility_score >= 70, (
            f"Specific anchored claim scored {scored.decision_utility_score}, expected >= 70"
        )

    def test_verified_public_with_numbers(self):
        """Verified public claim with quantified data should score well."""
        scored = score_claim(
            claim_text=(
                "Revenue grew 42% YoY to $180M under her leadership as SVP, "
                "confirmed by 2024 annual report. She decides on budget "
                "allocation, manages risk for the division, reports to "
                "the CEO with a mandate on headcount, and is responsible for OKR targets."
            ),
            claim_tag="VERIFIED-PUBLIC",
            anchors=["VERIFIED-PUBLIC", "VERIFIED-MEETING"],
            section_context="Section 6: Quantified Claims",
        )
        assert scored.decision_utility_score >= 60

    def test_multi_anchor_bonus(self):
        """Multiple distinct anchors should boost the evidence strength."""
        single = score_claim(
            claim_text="He owns the engineering roadmap and approves vendor spend.",
            claim_tag="VERIFIED-MEETING",
            anchors=["VERIFIED-MEETING"],
            section_context="",
        )
        multi = score_claim(
            claim_text="He owns the engineering roadmap and approves vendor spend.",
            claim_tag="VERIFIED-MEETING",
            anchors=["VERIFIED-MEETING", "VERIFIED-PDF", "VERIFIED-PUBLIC"],
            section_context="",
        )
        assert multi.decision_utility_score > single.decision_utility_score


# ---------------------------------------------------------------------------
# 3. Meeting moves have evidence refs
# ---------------------------------------------------------------------------


class TestMeetingMovesEvidenceRefs:
    """Every meeting move must have at least 1 evidence ref."""

    def test_meeting_moves_have_evidence_refs(self):
        """All moves generated from claims must carry evidence_refs."""
        claims = [
            score_claim(
                "Jane approved $3M budget for AI initiatives. [VERIFIED-MEETING]",
                claim_tag="VERIFIED-MEETING",
                anchors=["VERIFIED-MEETING"],
                section_context="Section 8: Structural Pressure Model",
            ),
            score_claim(
                "She is evaluated on revenue growth and NPS quarterly. [VERIFIED-PDF]",
                claim_tag="VERIFIED-PDF",
                anchors=["VERIFIED-PDF"],
                section_context="Section 6: Quantified Claims",
            ),
            score_claim(
                "Her team resists new vendor adoption due to training overhead. [INFERRED-H]",
                claim_tag="INFERRED-H",
                anchors=["INFERRED-H", "VERIFIED-MEETING"],
                section_context="Section 7: Rhetorical Patterns",
            ),
        ]
        moves = generate_meeting_moves(claims, person_name="Jane Smith")

        assert len(moves) >= 5, f"Expected >= 5 moves, got {len(moves)}"
        for move in moves:
            assert len(move.evidence_refs) >= 1, (
                f"Move '{move.move_type}' has 0 evidence_refs: {move.script[:60]}"
            )

    def test_empty_claims_still_produce_moves(self):
        """With zero claims, moves should still be generated (discovery mode)."""
        moves = generate_meeting_moves([], person_name="Unknown Person")
        assert len(moves) >= 5
        for move in moves:
            assert len(move.evidence_refs) >= 1

    def test_required_move_types_present(self):
        """Generated moves should cover all required types."""
        claims = [
            score_claim(
                "He controls a $5M annual budget for technology procurement. [VERIFIED-MEETING]",
                claim_tag="VERIFIED-MEETING",
                anchors=["VERIFIED-MEETING"],
                section_context="Section 8",
            ),
        ]
        moves = generate_meeting_moves(claims, person_name="Test Person")
        move_types = {m.move_type for m in moves}
        required = {MoveType.opener, MoveType.probe, MoveType.proof,
                    MoveType.wedge, MoveType.close, MoveType.avoid}
        assert required.issubset(move_types), (
            f"Missing move types: {required - move_types}"
        )


# ---------------------------------------------------------------------------
# 4. Decision grade gate fails with insufficient claims
# ---------------------------------------------------------------------------


class TestDecisionGradeGate:
    """Decision grade gate should FAIL when < 5 high-utility claims."""

    def test_decision_grade_gate_fails_when_insufficient_high_utility_claims(self):
        """Gate FAIL when the brief has fewer than 5 claims scoring >= 70."""
        # Build a brief with only 2 high-scoring claims
        brief = ExecutiveBrief(
            title="Test Brief",
            top_claims=[
                ScoredClaim(
                    text="Claim 1", decision_utility_score=80, tag="VERIFIED-MEETING",
                    anchors=["VERIFIED-MEETING"],
                ),
                ScoredClaim(
                    text="Claim 2", decision_utility_score=75, tag="VERIFIED-PDF",
                    anchors=["VERIFIED-PDF"],
                ),
            ],
            risks=["Risk 1"],
            agenda=["Agenda 1"],
            moves=[
                MeetingMove(
                    move_type=MoveType.opener, script="Open",
                    evidence_refs=["VERIFIED-MEETING"],
                ),
                MeetingMove(
                    move_type=MoveType.probe, script="Probe",
                    evidence_refs=["VERIFIED-PDF"],
                ),
                MeetingMove(
                    move_type=MoveType.proof, script="Proof",
                    evidence_refs=["VERIFIED-MEETING"],
                ),
                MeetingMove(
                    move_type=MoveType.wedge, script="Wedge",
                    evidence_refs=["INFERRED-H"],
                ),
                MeetingMove(
                    move_type=MoveType.close, script="Close",
                    evidence_refs=["VERIFIED-MEETING"],
                ),
                MeetingMove(
                    move_type=MoveType.avoid, script="Avoid",
                    evidence_refs=["INFERRED-M"],
                ),
            ],
        )
        dg = compute_decision_grade(
            brief=brief,
            identity_lock_score=80,
            evidence_coverage_pct=90.0,
        )
        assert dg.decision_grade_gate == DecisionGradeGateStatus.FAIL
        assert any("claims" in f.lower() for f in dg.decision_grade_failures)

    def test_decision_grade_gate_passes_with_sufficient_claims(self):
        """Gate PASS when 5+ claims scoring >= 70 and all moves have refs."""
        claims = [
            ScoredClaim(
                text=f"High-value claim {i}",
                decision_utility_score=75 + i,
                tag="VERIFIED-MEETING",
                anchors=["VERIFIED-MEETING"],
            )
            for i in range(6)
        ]
        moves = [
            MeetingMove(
                move_type=mt, script=f"Script for {mt.value}",
                evidence_refs=["VERIFIED-MEETING"],
            )
            for mt in MoveType
        ]
        brief = ExecutiveBrief(
            title="Test Brief",
            top_claims=claims,
            risks=["Risk 1"],
            agenda=["Agenda 1"],
            moves=moves,
        )
        dg = compute_decision_grade(
            brief=brief,
            identity_lock_score=80,
            evidence_coverage_pct=90.0,
        )
        assert dg.decision_grade_gate == DecisionGradeGateStatus.PASS
        assert dg.decision_grade_failures == []

    def test_decision_grade_score_range(self):
        """Decision grade score must be 0-100."""
        brief = ExecutiveBrief(title="Empty", top_claims=[], moves=[])
        dg = compute_decision_grade(brief=brief, identity_lock_score=0, evidence_coverage_pct=0)
        assert 0 <= dg.decision_grade_score <= 100

        dg2 = compute_decision_grade(brief=brief, identity_lock_score=100, evidence_coverage_pct=100)
        assert 0 <= dg2.decision_grade_score <= 100


# ---------------------------------------------------------------------------
# 5. Executive brief structure and claim filtering
# ---------------------------------------------------------------------------


class TestExecutiveBriefStructure:
    """Executive brief should contain only claims >= 70 and required move types."""

    def test_executive_brief_max_claims_and_structure(self):
        """Brief should filter to high-utility claims and have required moves."""
        # Build a dossier with a mix of high and low value claims
        high_value = [
            (
                "Jane approved a $4M budget for cloud migration reporting to CEO",
                "VERIFIED-MEETING",
            ),
            (
                "Revenue target of $50M for Q4 2025 with 30% growth mandate",
                "VERIFIED-PUBLIC",
            ),
            (
                "She blocked the previous vendor due to adoption friction concerns",
                "VERIFIED-MEETING",
            ),
            (
                "Board mandate requires 20% cost reduction by Q2 2026 deadline",
                "VERIFIED-PDF",
            ),
            (
                "She evaluates vendors on security compliance and headcount impact quarterly",
                "VERIFIED-MEETING",
            ),
            (
                "Internal champion for AI adoption with $2M allocation authority",
                "INFERRED-H",
            ),
        ]
        low_value = [
            ("She is a strategic leader.", "UNKNOWN"),
            ("The company focuses on growth.", "UNKNOWN"),
        ]

        dossier = _make_dossier_with_claims(high_value + low_value)
        brief = build_executive_brief(dossier, person_name="Jane Smith", company="Acme Corp")

        # All top_claims should score >= 70
        for claim in brief.top_claims:
            assert claim.decision_utility_score >= EXECUTIVE_BRIEF_THRESHOLD, (
                f"Claim in brief scored {claim.decision_utility_score}: {claim.text[:60]}"
            )

        # Title should contain person name
        assert "Jane Smith" in brief.title

        # Max 10 claims
        assert len(brief.top_claims) <= 10

        # Required move types
        move_types = {m.move_type for m in brief.moves}
        required = {MoveType.opener, MoveType.probe, MoveType.proof,
                    MoveType.wedge, MoveType.close, MoveType.avoid}
        assert required.issubset(move_types), (
            f"Missing move types: {required - move_types}"
        )

        # Risks should be populated
        assert len(brief.risks) >= 1

        # Agenda should be populated
        assert len(brief.agenda) >= 1

    def test_executive_brief_with_empty_dossier(self):
        """Brief from empty dossier should still be valid with defaults."""
        brief = build_executive_brief("", person_name="Nobody", company="NoCo")
        assert brief.title
        assert len(brief.agenda) >= 1
        assert len(brief.moves) >= 5  # Discovery mode fills moves

    def test_filter_claims_for_brief_threshold(self):
        """Only claims scoring >= 70 should pass the filter."""
        claims = [
            ScoredClaim(text="high", decision_utility_score=80),
            ScoredClaim(text="mid", decision_utility_score=65),
            ScoredClaim(text="low", decision_utility_score=40),
        ]
        filtered = filter_claims_for_brief(claims)
        assert len(filtered) == 1
        assert filtered[0].text == "high"


# ---------------------------------------------------------------------------
# 6. Claim extraction from dossier
# ---------------------------------------------------------------------------


class TestClaimExtraction:
    """Test extracting claims from dossier text."""

    def test_extract_tagged_claims(self):
        """Tagged lines should be extracted and scored."""
        dossier = _make_dossier_with_claims([
            ("She manages a $10M P&L and reports to the CEO.", "VERIFIED-MEETING"),
            ("The company was founded in 2010.", "VERIFIED-PUBLIC"),
        ])
        claims = extract_claims_from_dossier(dossier)
        assert len(claims) == 2
        assert all(isinstance(c, ScoredClaim) for c in claims)

    def test_untagged_lines_are_skipped(self):
        """Lines without evidence tags should not be extracted."""
        dossier = "### 1. Executive Summary\nShe is a great leader.\nNo evidence here."
        claims = extract_claims_from_dossier(dossier)
        assert len(claims) == 0

    def test_section_context_is_captured(self):
        """Extracted claims should carry their section context."""
        dossier = (
            "### 3. Career Timeline\n"
            "She served as VP of Engineering at Acme Corp from 2020-2023. [VERIFIED-PDF]\n"
        )
        claims = extract_claims_from_dossier(dossier)
        assert len(claims) == 1
        assert "Section 3" in claims[0].section
