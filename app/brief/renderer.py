"""Render a BriefOutput as human-readable markdown.

Outputs a Strategic Operating Model format with evidence discipline
tags on every claim. No generic filler, no corporate fluff.
"""

from __future__ import annotations

from app.models import (
    BriefOutput,
    Citation,
    EvidenceTag,
    TaggedClaim,
)

# Evidence tag display labels
_TAG_LABELS = {
    EvidenceTag.verified_meeting: "VERIFIED \u2013 MEETING",
    EvidenceTag.verified_public: "VERIFIED \u2013 PUBLIC",
    EvidenceTag.inferred_high: "INFERRED \u2013 HIGH CONFIDENCE",
    EvidenceTag.inferred_low: "INFERRED \u2013 LOW CONFIDENCE",
    EvidenceTag.unknown: "UNKNOWN",
}


def _cite(citations: list[Citation]) -> str:
    """Format inline citation references."""
    if not citations:
        return ""
    refs = []
    for c in citations:
        date_str = (
            c.timestamp.strftime("%Y-%m-%d")
            if hasattr(c.timestamp, "strftime")
            else str(c.timestamp)[:10]
        )
        refs.append(f"[{c.source_type.value}:{c.source_id}:{date_str}]")
    return " " + " ".join(refs)


def _tag(evidence_tag: EvidenceTag) -> str:
    """Format an evidence tag for inline display."""
    return f"`[{_TAG_LABELS.get(evidence_tag, 'UNKNOWN')}]`"


def _render_tagged_claim(claim: TaggedClaim, prefix: str = "-") -> str:
    """Render a single tagged claim as a markdown bullet."""
    return f"{prefix} {_tag(claim.evidence_tag)} {claim.claim}{_cite(claim.citations)}"


def _render_tagged_claims(
    claims: list[TaggedClaim], prefix: str = "-"
) -> list[str]:
    """Render a list of tagged claims as markdown bullets."""
    return [_render_tagged_claim(c, prefix) for c in claims]


def render_markdown(brief: BriefOutput) -> str:
    """Convert a BriefOutput to a Strategic Intelligence markdown dossier."""
    lines: list[str] = []
    h = brief.header

    # ── Header ──
    lines.append("# Strategic Intelligence Brief")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    if h.person:
        lines.append(f"| **Person** | {h.person} |")
    if h.company:
        lines.append(f"| **Company** | {h.company} |")
    if h.topic:
        lines.append(f"| **Topic** | {h.topic} |")
    if h.meeting_datetime:
        lines.append(
            f"| **Meeting** | {h.meeting_datetime.strftime('%Y-%m-%d %H:%M')} |"
        )
    lines.append(
        f"| **Generated** | {h.brief_generated_at.strftime('%Y-%m-%d %H:%M UTC')} |"
    )
    lines.append(f"| **Confidence** | {h.confidence_score:.0%} |")
    lines.append(
        f"| **Sources** | {', '.join(h.data_sources_used) or 'none'} |"
    )
    lines.append("")

    # ── 1. Strategic Positioning Snapshot ──
    lines.append("## 1. Strategic Positioning Snapshot")
    lines.append("")
    if brief.strategic_positioning:
        lines.extend(_render_tagged_claims(brief.strategic_positioning))
    else:
        lines.append("*No strategic positioning data available*")
    lines.append("")

    # ── 2. Power & Influence Map ──
    lines.append("## 2. Power & Influence Map")
    lines.append("")
    pm = brief.power_map
    _pm_fields = [
        ("Formal authority", pm.formal_authority),
        ("Informal influence", pm.informal_influence),
        ("Revenue control", pm.revenue_control),
        ("Decision gate ownership", pm.decision_gate_ownership),
        ("Needs to impress", pm.needs_to_impress),
        ("Veto risk", pm.veto_risk),
    ]
    has_pm = False
    for label, claim in _pm_fields:
        if claim:
            has_pm = True
            lines.append(
                f"- **{label}**: {_tag(claim.evidence_tag)} "
                f"{claim.claim}{_cite(claim.citations)}"
            )
        else:
            lines.append(f"- **{label}**: `[UNKNOWN]`")
    if not has_pm:
        lines.append("*No power map data available*")
    lines.append("")

    # ── 3. Incentive Structure Analysis ──
    lines.append("## 3. Incentive Structure Analysis")
    lines.append("")
    inc = brief.incentive_structure
    _inc_sections = [
        ("Short-term (0\u20133 months)", inc.short_term),
        ("Medium-term (3\u201312 months)", inc.medium_term),
        ("Career incentives", inc.career),
        ("Risk exposure", inc.risk_exposure),
        ("Where they personally win", inc.personal_wins),
        ("Where they personally lose", inc.personal_losses),
    ]
    has_inc = False
    for label, claims in _inc_sections:
        if claims:
            has_inc = True
            lines.append(f"**{label}:**")
            lines.extend(_render_tagged_claims(claims))
            lines.append("")
    if not has_inc:
        lines.append("*No incentive data available*")
        lines.append("")

    # ── 4. Cognitive & Rhetorical Patterns ──
    lines.append("## 4. Cognitive & Rhetorical Patterns")
    lines.append("")
    if brief.cognitive_patterns:
        for cp in brief.cognitive_patterns:
            lines.append(
                f"- **{cp.pattern_type}**: {_tag(cp.evidence_tag)} "
                f"{cp.observation}{_cite(cp.citations)}"
            )
            if cp.evidence_quote:
                lines.append(f'  > "{cp.evidence_quote}"')
    else:
        lines.append("*No cognitive patterns identified from available evidence*")
    lines.append("")

    # ── 5. Strategic Tensions ──
    lines.append("## 5. Strategic Tensions")
    lines.append("")
    if brief.strategic_tensions:
        for st in brief.strategic_tensions:
            lines.append(
                f"- **{st.tension}**: {_tag(st.evidence_tag)} "
                f"{st.evidence}{_cite(st.citations)}"
            )
    else:
        lines.append("*No strategic tensions identified*")
    lines.append("")

    # ── 6. Behavioral Forecast ──
    lines.append("## 6. Behavioral Forecast")
    lines.append("")
    if brief.behavioral_forecasts:
        for bf in brief.behavioral_forecasts:
            lines.append(f"**{bf.scenario}**")
            lines.append(f"- Predicted: {bf.predicted_reaction}")
            lines.append(f"- Reasoning: {bf.reasoning}{_cite(bf.citations)}")
            lines.append("")
    else:
        lines.append("*No behavioral forecasts \u2014 insufficient evidence*")
    lines.append("")

    # ── 7. Information Gaps That Matter ──
    lines.append("## 7. Information Gaps That Matter")
    lines.append("")
    if brief.information_gaps:
        for ig in brief.information_gaps:
            lines.append(f"- **{ig.gap}** \u2014 {ig.strategic_impact}")
    else:
        lines.append("*No material information gaps identified*")
    lines.append("")

    # ── 8. Executive Conversation Strategy ──
    lines.append("## 8. Executive Conversation Strategy")
    lines.append("")
    cs = brief.conversation_strategy
    if cs.leverage_angles:
        lines.append("**Leverage Angles:**")
        for i, la in enumerate(cs.leverage_angles, 1):
            lines.append(
                f"{i}. {_tag(la.evidence_tag)} {la.claim}{_cite(la.citations)}"
            )
        lines.append("")
    if cs.stress_tests:
        lines.append("**Stress Tests:**")
        for i, st in enumerate(cs.stress_tests, 1):
            lines.append(
                f"{i}. {_tag(st.evidence_tag)} {st.claim}{_cite(st.citations)}"
            )
        lines.append("")
    if cs.credibility_builders:
        lines.append("**Credibility Builders:**")
        for i, cb in enumerate(cs.credibility_builders, 1):
            lines.append(
                f"{i}. {_tag(cb.evidence_tag)} {cb.claim}{_cite(cb.citations)}"
            )
        lines.append("")
    if cs.contrarian_wedge:
        cw = cs.contrarian_wedge
        lines.append(
            f"**Contrarian Wedge:** {_tag(cw.evidence_tag)} "
            f"{cw.claim}{_cite(cw.citations)}"
        )
        lines.append("")
    if cs.collaboration_vector:
        cv = cs.collaboration_vector
        lines.append(
            f"**High-Upside Collaboration:** {_tag(cv.evidence_tag)} "
            f"{cv.claim}{_cite(cv.citations)}"
        )
        lines.append("")
    if not any([
        cs.leverage_angles, cs.stress_tests, cs.credibility_builders,
        cs.contrarian_wedge, cs.collaboration_vector,
    ]):
        lines.append("*No conversation strategy \u2014 insufficient evidence*")
        lines.append("")

    # ── 9. Meeting Delta Analysis ──
    lines.append("## 9. Meeting Delta Analysis")
    lines.append("")
    md = brief.meeting_delta
    if md.alignments:
        lines.append("**Alignments (public persona = meeting signals):**")
        lines.extend(_render_tagged_claims(md.alignments))
        lines.append("")
    if md.divergences:
        lines.append("**Divergences (public persona \u2260 meeting signals):**")
        lines.extend(_render_tagged_claims(md.divergences))
        lines.append("")
    if not md.alignments and not md.divergences:
        lines.append(
            "*No delta analysis \u2014 requires both public and meeting data*"
        )
        lines.append("")

    # ── Operational Context (legacy sections) ──
    lines.append("---")
    lines.append("")
    lines.append("## Operational Context")
    lines.append("")

    # Relationship Context
    rc = brief.relationship_context
    if rc.role or rc.influence_level or rc.relationship_health:
        lines.append("### Relationship Context")
        lines.append("")
        if rc.role:
            lines.append(f"- **Role**: {rc.role}{_cite(rc.citations)}")
        if rc.influence_level:
            tag = " *(inferred)*" if rc.influence_level_inferred else ""
            lines.append(
                f"- **Influence level**: {rc.influence_level}{tag}"
                f"{_cite(rc.citations)}"
            )
        if rc.relationship_health:
            tag = " *(inferred)*" if rc.relationship_health_inferred else ""
            lines.append(
                f"- **Relationship health**: {rc.relationship_health}{tag}"
                f"{_cite(rc.citations)}"
            )
        lines.append("")

    # Last Interaction
    lines.append("### Last Interaction")
    lines.append("")
    if brief.last_interaction:
        li = brief.last_interaction
        date_str = ""
        if li.date:
            date_str = (
                f" ({li.date.strftime('%Y-%m-%d') if hasattr(li.date, 'strftime') else str(li.date)[:10]})"
            )
        lines.append(f"{li.summary}{date_str}{_cite(li.citations)}")
        if li.commitments:
            lines.append("")
            lines.append("**Commitments:**")
            for c in li.commitments:
                lines.append(f"- {c}")
    else:
        lines.append("*Unknown \u2013 no interaction data available*")
    lines.append("")

    # Interaction History
    if brief.interaction_history:
        lines.append("### Recent Interaction History")
        lines.append("")
        for ix in brief.interaction_history[:10]:
            date_str = ""
            if ix.date:
                date_str = (
                    f"**{ix.date.strftime('%Y-%m-%d') if hasattr(ix.date, 'strftime') else str(ix.date)[:10]}** \u2013 "
                )
            lines.append(f"- {date_str}{ix.summary}{_cite(ix.citations)}")
        lines.append("")

    # Open Loops
    lines.append("### Open Loops")
    lines.append("")
    if brief.open_loops:
        for ol in brief.open_loops:
            owner = f" (Owner: {ol.owner})" if ol.owner else ""
            due = f" [Due: {ol.due_date}]" if ol.due_date else ""
            lines.append(
                f"- {ol.description}{owner}{due}{_cite(ol.citations)}"
            )
    else:
        lines.append("*No open loops identified*")
    lines.append("")

    # Watchouts
    lines.append("### Watchouts & Risks")
    lines.append("")
    if brief.watchouts:
        for w in brief.watchouts:
            severity_icon = {"high": "\U0001F534", "medium": "\U0001F7E1", "low": "\U0001F7E2"}.get(
                w.severity, "\u26AA"
            )
            lines.append(
                f"- {severity_icon} **{w.severity.upper()}**: "
                f"{w.description}{_cite(w.citations)}"
            )
    else:
        lines.append("*No watchouts identified*")
    lines.append("")

    # Meeting Objectives
    if brief.meeting_objectives:
        lines.append("### Meeting Objectives")
        lines.append("")
        for mo in brief.meeting_objectives:
            lines.append(f"- **Objective**: {mo.objective}")
            lines.append(
                f"  - *Measurable outcome*: "
                f"{mo.measurable_outcome}{_cite(mo.citations)}"
            )
        lines.append("")

    # Agenda
    if brief.agenda.variants:
        lines.append("### Suggested Agenda")
        lines.append("")
        for variant in brief.agenda.variants:
            lines.append(f"#### {variant.duration_minutes}-Minute Version")
            lines.append("")
            lines.append("| Time | Block | Notes |")
            lines.append("|------|-------|-------|")
            elapsed = 0
            for block in variant.blocks:
                lines.append(
                    f"| {elapsed}\u2013{elapsed + block.minutes} min "
                    f"| {block.label} | {block.notes or ''} |"
                )
                elapsed += block.minutes
            lines.append("")

    # ── Engine Improvement Recommendations ──
    lines.append("---")
    lines.append("")
    lines.append("## Engine Improvement Recommendations")
    lines.append("")
    ei = brief.engine_improvements
    if ei.missing_signals:
        lines.append("**Missing Signals:**")
        for s in ei.missing_signals:
            lines.append(f"- {s}")
        lines.append("")
    if ei.recommended_data_sources:
        lines.append("**Recommended Data Sources:**")
        for ds in ei.recommended_data_sources:
            lines.append(f"- {ds}")
        lines.append("")
    if ei.capture_fields:
        lines.append("**Capture Fields for Future Calls:**")
        for cf in ei.capture_fields:
            lines.append(f"- {cf}")
        lines.append("")
    if not ei.missing_signals and not ei.recommended_data_sources and not ei.capture_fields:
        lines.append("*No improvement recommendations at this time*")
        lines.append("")

    # ── Appendix ──
    lines.append("## Appendix: Evidence Sources")
    lines.append("")
    if brief.appendix_evidence:
        lines.append("| # | Type | ID | Date | Title |")
        lines.append("|---|------|----|------|-------|")
        _dash = "\u2014"
        for i, ev in enumerate(brief.appendix_evidence, 1):
            date_str = ev.date.strftime("%Y-%m-%d") if ev.date else _dash
            title_str = ev.title or _dash
            lines.append(
                f"| {i} | {ev.source_type.value} | `{ev.source_id}` "
                f"| {date_str} | {title_str} |"
            )
    else:
        lines.append("*No evidence sources available*")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Pre-Call Intelligence Briefing Engine*")

    return "\n".join(lines)
