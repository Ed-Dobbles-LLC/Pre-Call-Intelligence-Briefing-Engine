"""Decision Leverage Engine v1 — scoring, filtering, executive brief, meeting moves.

Deterministic scoring of dossier claims for decision utility. Produces:
1. Scored claims with utility tags
2. Filtered high-leverage claims (score >= 70) for executive brief
3. Prescriptive meeting moves with evidence citations
4. 1-page executive brief combining claims + moves + agenda
"""

from __future__ import annotations

import re
import logging

from app.models import (
    ClaimType,
    DecisionGradeGateStatus,
    DecisionGradeQA,
    ExecutiveBrief,
    MeetingMove,
    MoveType,
    RelevanceWindow,
    RiskLevel,
    ScoredClaim,
    UtilityTag,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXECUTIVE_BRIEF_THRESHOLD = 70
APPENDIX_THRESHOLD = 55

# Generic phrases that reduce utility (penalty applied)
GENERIC_PENALTY_PHRASES: list[re.Pattern] = [
    re.compile(r"\b(ROI|scalab(le|ility)|pragmatic|best practices)\b", re.IGNORECASE),
    re.compile(r"\bdata[- ]driven\b", re.IGNORECASE),
    re.compile(r"\bresults[- ]driven\b", re.IGNORECASE),
    re.compile(r"\bstrategic leader\b", re.IGNORECASE),
    re.compile(r"\bthought leader\b", re.IGNORECASE),
    re.compile(r"\bpassionate about\b", re.IGNORECASE),
    re.compile(r"\btransformative\b", re.IGNORECASE),
    re.compile(r"\bgame[- ]chang(ing|er)\b", re.IGNORECASE),
    re.compile(r"\bcutting[- ]edge\b", re.IGNORECASE),
    re.compile(r"\bproven track record\b", re.IGNORECASE),
    re.compile(r"\bempowers teams\b", re.IGNORECASE),
    re.compile(r"\bbridges the gap\b", re.IGNORECASE),
    re.compile(r"\bat the intersection of\b", re.IGNORECASE),
    re.compile(r"\binnovative approach\b", re.IGNORECASE),
    re.compile(r"\bholistic\b", re.IGNORECASE),
    re.compile(r"\bsynerg(y|ies)\b", re.IGNORECASE),
    re.compile(r"\bvisionary\b", re.IGNORECASE),
]

# Anchor type weights for evidence strength
ANCHOR_TYPE_WEIGHTS: dict[str, int] = {
    "VERIFIED-PDF": 8,
    "VERIFIED-PUBLIC": 7,
    "VERIFIED-MEETING": 9,
    "INFERRED-H": 5,
    "INFERRED-M": 3,
    "INFERRED-L": 1,
    "UNKNOWN": 0,
}

# Utility tag keyword triggers (deterministic mapping)
_UTILITY_TAG_TRIGGERS: dict[UtilityTag, list[re.Pattern]] = {
    UtilityTag.sponsor_risk: [
        re.compile(r"\b(sponsor|champion|advocate|internal ally)\b", re.IGNORECASE),
    ],
    UtilityTag.veto_risk: [
        re.compile(r"\b(veto|block|gatekeep|resist|pushback|skeptic)\b", re.IGNORECASE),
    ],
    UtilityTag.sales_cycle: [
        re.compile(
            r"\b(timeline|urgency|deadline|procurement|RFP|evaluation|pilot|POC)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.adoption_friction: [
        re.compile(
            r"\b(adoption|change management|resistance|rollout|training|onboard)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.credibility: [
        re.compile(
            r"\b(credibility|trust|reputation|track record|reference|testimonial)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.budget_authority: [
        re.compile(
            r"\b(budget|spend|sign off|procurement|P&L|cost center|allocation)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.politics: [
        re.compile(
            r"\b(politic|faction|power|stakeholder|alignment|coalition|territory)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.differentiation: [
        re.compile(
            r"\b(differentiat|unique|competitive advantage|moat|edge|distinct)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.negotiation_lever: [
        re.compile(
            r"\b(negotiat|lever|concession|pricing|terms|discount|trade-off)\b",
            re.IGNORECASE,
        ),
    ],
    UtilityTag.unknowns_to_resolve: [
        re.compile(r"\b(unknown|unclear|gap|unverified|TBD|open question)\b", re.IGNORECASE),
    ],
}

# Behavior impact keywords (higher score)
_BEHAVIOR_KEYWORDS: list[re.Pattern] = [
    re.compile(r"\b(decision|decides|approve|reject|evaluate)\b", re.IGNORECASE),
    re.compile(r"\b(accountab|responsible for|measured on|KPI|OKR|quota)\b", re.IGNORECASE),
    re.compile(r"\b(risk|threat|pressure|mandate|deadline)\b", re.IGNORECASE),
    re.compile(r"\b(next step|action|priority|initiative|blocker)\b", re.IGNORECASE),
    re.compile(r"\b(budget|revenue|headcount|P&L|margin)\b", re.IGNORECASE),
    re.compile(r"\b(reports to|reporting line|hierarchy|authority)\b", re.IGNORECASE),
    re.compile(r"\b(veto|champion|sponsor|gatekeeper)\b", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Claim Scoring
# ---------------------------------------------------------------------------


def _compute_evidence_strength(anchors: list[str], claim_type: ClaimType) -> int:
    """Score evidence strength (0-25) based on anchor count and types."""
    if not anchors:
        return 0

    total = 0
    for anchor in anchors:
        anchor_upper = anchor.upper().strip("[]")
        best_weight = 0
        for tag, weight in ANCHOR_TYPE_WEIGHTS.items():
            if tag in anchor_upper:
                best_weight = max(best_weight, weight)
        total += best_weight

    # Base: up to 15 from anchor weights, bonus for multiple anchors
    raw = min(total, 15)
    multi_bonus = min(len(anchors) - 1, 2) * 5 if len(anchors) > 1 else 0
    return min(raw + multi_bonus, 25)


def _compute_specificity(text: str) -> int:
    """Score specificity (0-25) via heuristics: digits, names, roles."""
    score = 0

    # Digits (revenue, headcount, percentages, dates)
    digits = re.findall(r"\d[\d,.%$€£]+", text)
    score += min(len(digits) * 5, 12)

    # Named entities (capitalized multi-word, likely proper nouns)
    named = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b", text)
    score += min(len(named) * 3, 6)

    # Role/title references
    role_re = re.compile(
        r"\b(CEO|CTO|CFO|COO|VP|SVP|EVP|Director|Manager|Head of|Partner|Founder)\b",
        re.IGNORECASE,
    )
    if role_re.search(text):
        score += 4

    # Company references (Inc, Corp, LLC, or well-known patterns)
    company_re = re.compile(r"\b\w+(?:Inc|Corp|LLC|Ltd|Co)\b|\b[A-Z]{2,}\b")
    if company_re.search(text):
        score += 3

    return min(score, 25)


def _compute_behavior_impact(text: str) -> int:
    """Score behavior impact (0-35) — actionable implications."""
    score = 0
    for pattern in _BEHAVIOR_KEYWORDS:
        if pattern.search(text):
            score += 5
    return min(score, 35)


def _compute_generic_penalty(text: str) -> int:
    """Penalty (0-15) for generic phrases without specifics."""
    hits = 0
    for pattern in GENERIC_PENALTY_PHRASES:
        if pattern.search(text):
            hits += 1

    # If the claim has generic phrases but also has specifics (digits),
    # reduce the penalty
    has_specifics = bool(re.search(r"\d[\d,.%$]+", text))
    if has_specifics:
        hits = max(0, hits - 1)

    return min(hits * 5, 15)


def _detect_utility_tags(text: str) -> list[UtilityTag]:
    """Detect utility tags from claim text using keyword triggers."""
    tags: list[UtilityTag] = []
    for tag, patterns in _UTILITY_TAG_TRIGGERS.items():
        for pattern in patterns:
            if pattern.search(text):
                tags.append(tag)
                break
    return tags


def _detect_claim_type(tag: str) -> ClaimType:
    """Map an evidence tag string to ClaimType."""
    upper = tag.upper()
    if "VERIFIED" in upper:
        return ClaimType.fact
    if "STRATEGIC MODEL" in upper or "STRATEGIC" in upper:
        return ClaimType.strategic_model
    if "INFERRED" in upper:
        return ClaimType.inference
    return ClaimType.fact


def _detect_relevance_window(text: str) -> RelevanceWindow:
    """Detect relevance window from text heuristics."""
    now_re = re.compile(
        r"\b(this week|immediate|urgent|right now|current quarter|Q[1-4] 202[5-7])\b",
        re.IGNORECASE,
    )
    near_re = re.compile(
        r"\b(next quarter|30 days|pipeline|upcoming|this year|near-term)\b",
        re.IGNORECASE,
    )
    if now_re.search(text):
        return RelevanceWindow.now
    if near_re.search(text):
        return RelevanceWindow.next_30d
    return RelevanceWindow.evergreen


def _generate_behavior_implication(text: str, tags: list[UtilityTag]) -> str:
    """Generate a one-sentence behavior implication."""
    if UtilityTag.budget_authority in tags:
        return "Has direct budget influence — position pricing early."
    if UtilityTag.veto_risk in tags:
        return "Potential blocker — address objections before formal review."
    if UtilityTag.sponsor_risk in tags:
        return "Internal champion — equip with ammunition for internal sell."
    if UtilityTag.adoption_friction in tags:
        return "Adoption concerns present — lead with change management story."
    if UtilityTag.politics in tags:
        return "Political dynamics at play — map stakeholder alignment."
    if UtilityTag.sales_cycle in tags:
        return "Timeline signal detected — calibrate urgency in next meeting."
    if UtilityTag.differentiation in tags:
        return "Competitive positioning matters — sharpen differentiation."
    if UtilityTag.negotiation_lever in tags:
        return "Negotiation angle present — prepare concession framework."
    if UtilityTag.credibility in tags:
        return "Trust signal — reinforce with proof points."
    if UtilityTag.unknowns_to_resolve in tags:
        return "Critical unknown — resolve before advancing."
    return "Review for meeting preparation."


def score_claim(
    claim_text: str,
    claim_tag: str = "UNKNOWN",
    anchors: list[str] | None = None,
    section_context: str = "",
) -> ScoredClaim:
    """Score a single claim for decision utility.

    Returns a ScoredClaim with deterministic scoring:
    - Evidence Strength (0-25): anchor count + types
    - Specificity (0-25): digits, named entities, roles
    - Behavior Impact (0-35): actionable keywords
    - Generic Penalty (0-15): deducted for filler phrases
    """
    anchors = anchors or []
    claim_type = _detect_claim_type(claim_tag)
    full_text = claim_text + " " + section_context

    evidence = _compute_evidence_strength(anchors, claim_type)
    specificity = _compute_specificity(claim_text)
    behavior = _compute_behavior_impact(full_text)
    penalty = _compute_generic_penalty(claim_text)

    raw_score = evidence + specificity + behavior - penalty
    score = max(0, min(100, raw_score))

    tags = _detect_utility_tags(full_text)
    implication = _generate_behavior_implication(full_text, tags)
    window = _detect_relevance_window(claim_text)

    return ScoredClaim(
        text=claim_text,
        claim_type=claim_type,
        confidence=("H" if "H" in claim_tag.upper() else
                     "M" if "M" in claim_tag.upper() else "L"),
        tag=claim_tag,
        anchors=anchors,
        utility_tags=tags,
        decision_utility_score=score,
        behavior_implication=implication,
        relevance_window=window,
        section=section_context,
    )


# ---------------------------------------------------------------------------
# Claim Extraction from Dossier Text
# ---------------------------------------------------------------------------

# Matches lines with evidence tags
_TAGGED_LINE_RE = re.compile(
    r"\[(?:VERIFIED[–\-](?:MEETING|PUBLIC|PDF)|INFERRED[–\-][HML]"
    r"|UNKNOWN|STRATEGIC MODEL[^\]]*)\]",
    re.IGNORECASE,
)

_SECTION_HEADER_RE = re.compile(r"^###\s+(\d+)\.\s+(.+)")
_EVIDENCE_TAG_RE = re.compile(
    r"\[(VERIFIED[–\-](?:MEETING|PUBLIC|PDF)|INFERRED[–\-][HML]"
    r"|UNKNOWN|STRATEGIC MODEL[^\]]*)\]",
    re.IGNORECASE,
)


def extract_claims_from_dossier(dossier_text: str) -> list[ScoredClaim]:
    """Extract and score all tagged claims from a dossier.

    Parses tagged lines from factual sections (1-8, 12) and scores each
    for decision utility.
    """
    claims: list[ScoredClaim] = []
    current_section = ""
    lines = dossier_text.split("\n")

    for line in lines:
        stripped = line.strip()

        # Track section headers
        header_match = _SECTION_HEADER_RE.match(stripped)
        if header_match:
            sec_num = header_match.group(1)
            sec_name = header_match.group(2).strip()
            current_section = f"Section {sec_num}: {sec_name}"
            continue

        # Skip structural lines
        if not stripped or len(stripped) < 25:
            continue
        if stripped.startswith(("#", "|", "---", ">")):
            continue

        # Only process lines with evidence tags
        tag_match = _EVIDENCE_TAG_RE.search(stripped)
        if not tag_match:
            continue

        tag = tag_match.group(1)
        # Clean claim text (remove the tag itself for readability)
        claim_text = _EVIDENCE_TAG_RE.sub("", stripped).strip()
        claim_text = re.sub(r"\s+", " ", claim_text).strip(" -–*•")

        if len(claim_text) < 15:
            continue

        # Build anchors list from tag
        anchors = [tag]

        scored = score_claim(
            claim_text=claim_text,
            claim_tag=tag,
            anchors=anchors,
            section_context=current_section,
        )
        claims.append(scored)

    return claims


# ---------------------------------------------------------------------------
# Claim Filtering
# ---------------------------------------------------------------------------


def filter_claims_for_brief(claims: list[ScoredClaim]) -> list[ScoredClaim]:
    """Keep only claims with decision_utility_score >= 70 for executive brief.

    Claims scoring 55-69 are appendix-only (not returned here).
    Claims below 55 are dropped entirely.
    """
    return [c for c in claims if c.decision_utility_score >= EXECUTIVE_BRIEF_THRESHOLD]


def filter_claims_for_appendix(claims: list[ScoredClaim]) -> list[ScoredClaim]:
    """Claims scoring 55-69 — included in appendix notes only."""
    return [
        c for c in claims
        if APPENDIX_THRESHOLD <= c.decision_utility_score < EXECUTIVE_BRIEF_THRESHOLD
    ]


# ---------------------------------------------------------------------------
# Meeting Moves Generator
# ---------------------------------------------------------------------------


def _build_move_from_claim(
    claim: ScoredClaim,
    move_type: MoveType,
    script: str,
    why: str,
    risk: RiskLevel = RiskLevel.low,
) -> MeetingMove:
    """Helper to build a MeetingMove from a claim's evidence refs."""
    refs = claim.anchors if claim.anchors else [claim.tag]
    return MeetingMove(
        move_type=move_type,
        script=script,
        why_it_works=why,
        evidence_refs=refs,
        risk_level=risk,
    )


def generate_meeting_moves(
    top_claims: list[ScoredClaim],
    person_name: str = "",
) -> list[MeetingMove]:
    """Generate prescriptive meeting moves from top-scoring claims.

    Must output at least:
    - 1 opener, 3 probes, 2 proof points, 1 wedge, 1 close, 2 avoids
    """
    moves: list[MeetingMove] = []

    # Sort claims by score descending
    sorted_claims = sorted(top_claims, key=lambda c: c.decision_utility_score, reverse=True)

    # --- Opener (1) ---
    if sorted_claims:
        best = sorted_claims[0]
        moves.append(_build_move_from_claim(
            best,
            MoveType.opener,
            script=(
                f"I noticed your focus on {_extract_topic(best.text)} — "
                f"that aligns with what we've been hearing from leaders in "
                f"similar positions. What's driving that priority right now?"
            ),
            why=(
                f"Opens with their stated priority ({best.tag}), "
                "demonstrating preparation and inviting them to frame the conversation."
            ),
        ))
    else:
        # Discovery opener if no claims
        moves.append(MeetingMove(
            move_type=MoveType.opener,
            script=(
                "I'd like to understand what's top of mind for you right now "
                "and where you see the biggest opportunities."
            ),
            why_it_works="Discovery opener — insufficient evidence for a targeted opening.",
            evidence_refs=["INSUFFICIENT_EVIDENCE"],
            risk_level=RiskLevel.low,
        ))

    # --- Probes (3) ---
    probe_claims = [c for c in sorted_claims if c.utility_tags]
    probe_templates = [
        ("budget_authority", "Can you walk me through your evaluation and approval process for initiatives like this?", "Surfaces budget authority and procurement path."),
        ("adoption_friction", "What's been the biggest barrier to adoption for new tools in your organization?", "Surfaces internal resistance and change management needs."),
        ("sales_cycle", "What does your timeline look like for making a decision here?", "Calibrates urgency and procurement cadence."),
        ("veto_risk", "Who else needs to be comfortable with this before moving forward?", "Maps stakeholder landscape and potential blockers."),
        ("politics", "How does this initiative fit within your broader organizational priorities?", "Reveals political dynamics and alignment."),
        ("sponsor_risk", "Who internally is championing this initiative?", "Identifies internal sponsors and their conviction."),
        ("credibility", "What would success look like for you specifically?", "Anchors to their personal metrics."),
        ("differentiation", "What alternatives have you considered?", "Surfaces competitive landscape from their perspective."),
    ]

    probes_added = 0
    used_tags: set[str] = set()
    for claim in probe_claims:
        if probes_added >= 3:
            break
        for tag_name, script, why in probe_templates:
            if tag_name in used_tags:
                continue
            if any(t.value == tag_name for t in claim.utility_tags):
                moves.append(_build_move_from_claim(
                    claim, MoveType.probe, script=script, why=why,
                ))
                used_tags.add(tag_name)
                probes_added += 1
                break

    # Fill remaining probes with generic discovery
    generic_probes = [
        ("What's the most important outcome you need from this engagement?",
         "Discovery probe — identifies primary success metric."),
        ("What's worked and what hasn't in previous attempts at this?",
         "Discovery probe — surfaces lessons learned and preferences."),
        ("Where do you see the biggest risk in your current approach?",
         "Discovery probe — opens risk discussion."),
    ]
    for script, why in generic_probes:
        if probes_added >= 3:
            break
        moves.append(MeetingMove(
            move_type=MoveType.probe,
            script=script,
            why_it_works=why,
            evidence_refs=sorted_claims[0].anchors if sorted_claims else ["DISCOVERY"],
            risk_level=RiskLevel.low,
        ))
        probes_added += 1

    # --- Proof Points (2) ---
    credibility_claims = [
        c for c in sorted_claims
        if any(t in (UtilityTag.credibility, UtilityTag.differentiation)
               for t in c.utility_tags)
    ]
    proof_source = credibility_claims or sorted_claims
    for i, claim in enumerate(proof_source[:2]):
        topic = _extract_topic(claim.text)
        moves.append(_build_move_from_claim(
            claim,
            MoveType.proof,
            script=(
                f"We recently helped a similar organization address "
                f"{topic}. The result was measurable impact "
                f"in the area they cared about most."
            ),
            why=(
                f"Mapped to their stated concern ({claim.tag}). "
                "Proof resonates when it mirrors their exact pressure."
            ),
        ))

    # Fill if needed
    if len([m for m in moves if m.move_type == MoveType.proof]) < 2:
        moves.append(MeetingMove(
            move_type=MoveType.proof,
            script="We can share a reference from a comparable organization facing similar dynamics.",
            why_it_works="Generic proof offer — specifics should be prepared pre-meeting.",
            evidence_refs=sorted_claims[0].anchors if sorted_claims else ["GENERIC"],
            risk_level=RiskLevel.low,
        ))

    # --- Wedge (1) ---
    tension_claims = [
        c for c in sorted_claims
        if any(t in (UtilityTag.veto_risk, UtilityTag.adoption_friction,
                     UtilityTag.politics) for t in c.utility_tags)
    ]
    if tension_claims:
        tc = tension_claims[0]
        moves.append(_build_move_from_claim(
            tc,
            MoveType.wedge,
            script=(
                f"I want to stress-test something: {_extract_topic(tc.text)} — "
                f"what happens if this doesn't get addressed in the next quarter?"
            ),
            why=(
                "Surfaces urgency by making the cost of inaction concrete. "
                f"Based on {tc.tag} evidence."
            ),
            risk=RiskLevel.med,
        ))
    else:
        moves.append(MeetingMove(
            move_type=MoveType.wedge,
            script="What happens if you stay on the current path for another year?",
            why_it_works="Generic wedge — surfaces cost of inaction.",
            evidence_refs=sorted_claims[0].anchors if sorted_claims else ["DISCOVERY"],
            risk_level=RiskLevel.med,
        ))

    # --- Close (1) ---
    if sorted_claims:
        best = sorted_claims[0]
        moves.append(_build_move_from_claim(
            best,
            MoveType.close,
            script=(
                f"Based on what you've shared about {_extract_topic(best.text)}, "
                f"I'd recommend we schedule a focused session to map this out "
                f"with the relevant stakeholders. Does next week work?"
            ),
            why=(
                "Ties back to their stated priority and proposes a concrete next step."
            ),
        ))
    else:
        moves.append(MeetingMove(
            move_type=MoveType.close,
            script="Can we schedule a follow-up to go deeper on what we discussed?",
            why_it_works="Discovery close — advances conversation.",
            evidence_refs=["DISCOVERY"],
            risk_level=RiskLevel.low,
        ))

    # --- Avoids (2) ---
    avoid_topics = []
    for claim in sorted_claims:
        if any(t in (UtilityTag.veto_risk, UtilityTag.politics) for t in claim.utility_tags):
            avoid_topics.append(claim)
    if avoid_topics:
        for claim in avoid_topics[:2]:
            moves.append(_build_move_from_claim(
                claim,
                MoveType.avoid,
                script=(
                    f"Do NOT lead with generic ROI claims or challenge their "
                    f"current approach to {_extract_topic(claim.text)} directly."
                ),
                why=(
                    f"Based on {claim.tag}: this topic is politically sensitive. "
                    "Let them frame the problem first."
                ),
                risk=RiskLevel.high,
            ))
    # Fill avoids
    while len([m for m in moves if m.move_type == MoveType.avoid]) < 2:
        moves.append(MeetingMove(
            move_type=MoveType.avoid,
            script=(
                "Avoid making assumptions about their decision process or timeline "
                "without confirming first."
            ),
            why_it_works="Generic avoid — insufficient evidence to identify specific landmines.",
            evidence_refs=sorted_claims[0].anchors if sorted_claims else ["GENERIC"],
            risk_level=RiskLevel.low,
        ))

    return moves


def _extract_topic(text: str) -> str:
    """Extract a short topic phrase from claim text (first ~8 words)."""
    words = text.split()
    if len(words) <= 8:
        return text.lower().rstrip(".")
    return " ".join(words[:8]).lower().rstrip(".") + "..."


# ---------------------------------------------------------------------------
# Executive Brief Builder
# ---------------------------------------------------------------------------


def build_executive_brief(
    dossier_text: str,
    person_name: str = "",
    company: str = "",
) -> ExecutiveBrief:
    """Build a 1-page executive brief from a dossier.

    Steps:
    1. Extract and score all tagged claims
    2. Filter to high-utility claims (>= 70)
    3. Generate meeting moves
    4. Build agenda from top utility tags
    5. Extract risks
    """
    # 1. Extract and score
    all_claims = extract_claims_from_dossier(dossier_text)

    # 2. Filter
    top_claims = filter_claims_for_brief(all_claims)
    # Sort by score descending, limit to top 10 for 1-page constraint
    top_claims.sort(key=lambda c: c.decision_utility_score, reverse=True)
    top_claims = top_claims[:10]

    # 3. Generate moves
    moves = generate_meeting_moves(top_claims, person_name=person_name)

    # 4. Build agenda from top utility tags
    tag_counts: dict[str, int] = {}
    for claim in top_claims:
        for tag in claim.utility_tags:
            tag_counts[tag.value] = tag_counts.get(tag.value, 0) + 1

    tag_to_agenda: dict[str, str] = {
        "budget_authority": "Clarify budget authority and procurement path",
        "adoption_friction": "Address adoption barriers and change management",
        "sales_cycle": "Align on evaluation timeline and decision process",
        "veto_risk": "Map stakeholder landscape and potential blockers",
        "sponsor_risk": "Identify and equip internal champion",
        "politics": "Navigate organizational dynamics",
        "credibility": "Establish credibility with relevant proof points",
        "differentiation": "Differentiate from alternatives they are considering",
        "negotiation_lever": "Frame value proposition for commercial discussion",
        "unknowns_to_resolve": "Resolve critical information gaps",
    }
    sorted_tags = sorted(tag_counts.items(), key=lambda x: -x[1])
    agenda = [tag_to_agenda.get(tag, f"Discuss {tag}") for tag, _ in sorted_tags[:6]]
    if not agenda:
        agenda = [
            "Understand current priorities and challenges",
            "Present relevant capabilities",
            "Identify next steps",
        ]

    # 5. Extract risks
    risks: list[str] = []
    for claim in all_claims:
        if any(t in (UtilityTag.veto_risk, UtilityTag.sponsor_risk,
                     UtilityTag.adoption_friction) for t in claim.utility_tags):
            risk_text = claim.text[:150]
            ref = f" [{claim.tag}]" if claim.tag != "UNKNOWN" else " [UNKNOWN]"
            risks.append(risk_text + ref)
    if not risks:
        risks = ["No specific risks identified from available evidence [UNKNOWN]"]
    risks = risks[:5]

    title = f"Executive Brief: {person_name}" if person_name else "Executive Brief"
    if company:
        title += f" ({company})"

    return ExecutiveBrief(
        title=title,
        top_claims=top_claims,
        risks=risks,
        agenda=agenda,
        moves=moves,
    )


# ---------------------------------------------------------------------------
# Decision Grade Gate
# ---------------------------------------------------------------------------

MIN_HIGH_UTILITY_CLAIMS = 5
MIN_TOTAL_MOVES = 5
REQUIRED_MOVE_TYPES = {MoveType.opener, MoveType.probe, MoveType.proof,
                       MoveType.wedge, MoveType.close, MoveType.avoid}


def compute_decision_grade(
    brief: ExecutiveBrief,
    identity_lock_score: int = 0,
    evidence_coverage_pct: float = 0.0,
) -> DecisionGradeQA:
    """Compute decision grade score and gate status.

    Score formula:
      0.25 * identity_lock_score
    + 0.35 * avg(top_claims.utility_score)
    + 0.20 * evidence_coverage_pct
    + 0.20 * move_quality_score

    Gate FAIL if:
    - fewer than 5 claims with score >= 70
    - fewer than 5 total moves
    - any move has zero evidence_refs
    """
    failures: list[str] = []

    # Count high-utility claims
    high_claims = [c for c in brief.top_claims if c.decision_utility_score >= 70]
    if len(high_claims) < MIN_HIGH_UTILITY_CLAIMS:
        failures.append(
            f"Only {len(high_claims)} claims with score >= 70 "
            f"(minimum {MIN_HIGH_UTILITY_CLAIMS} required)"
        )

    # Check total moves
    if len(brief.moves) < MIN_TOTAL_MOVES:
        failures.append(
            f"Only {len(brief.moves)} meeting moves "
            f"(minimum {MIN_TOTAL_MOVES} required)"
        )

    # Check evidence refs on all moves
    empty_ref_moves = [m for m in brief.moves if not m.evidence_refs]
    if empty_ref_moves:
        failures.append(
            f"{len(empty_ref_moves)} meeting moves have zero evidence_refs"
        )

    # Compute score components
    avg_utility = 0.0
    if high_claims:
        avg_utility = sum(c.decision_utility_score for c in high_claims) / len(high_claims)

    # Move quality: count of required move types present
    move_types_present = {m.move_type for m in brief.moves}
    move_coverage = len(move_types_present & REQUIRED_MOVE_TYPES)
    move_quality = min(move_coverage / len(REQUIRED_MOVE_TYPES) * 100, 100)

    decision_grade = int(
        0.25 * identity_lock_score
        + 0.35 * avg_utility
        + 0.20 * evidence_coverage_pct
        + 0.20 * move_quality
    )
    decision_grade = max(0, min(100, decision_grade))

    gate_status = (
        DecisionGradeGateStatus.PASS if not failures
        else DecisionGradeGateStatus.FAIL
    )

    summary = {
        "total_claims_scored": len(brief.top_claims),
        "high_utility_claims": len(high_claims),
        "avg_utility_score": round(avg_utility, 1),
        "total_moves": len(brief.moves),
        "move_types_present": sorted(m.value for m in move_types_present),
        "move_quality_score": round(move_quality, 1),
        "identity_lock_component": round(0.25 * identity_lock_score, 1),
        "utility_component": round(0.35 * avg_utility, 1),
        "coverage_component": round(0.20 * evidence_coverage_pct, 1),
        "move_component": round(0.20 * move_quality, 1),
    }

    return DecisionGradeQA(
        decision_utility_summary=summary,
        decision_grade_score=decision_grade,
        decision_grade_gate=gate_status,
        decision_grade_failures=failures,
    )
