"""Render a BriefOutput as a person-first, evidence-locked pre-call brief.

Outputs a concise 1-2 page markdown brief organised as sections A-I.
Every claim carries evidence tags and inline citations.
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
    """Convert a BriefOutput to a person-first pre-call intelligence brief."""
    lines: list[str] = []
    h = brief.header

    # ── A) Header ──
    lines.append("# Pre-Call Intelligence Brief")
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
    # Gate scores
    if h.gate_status != "not_run":
        lines.append(
            f"| **Identity Lock** | {h.identity_lock_score:.0f}/100 |"
        )
        lines.append(
            f"| **Evidence Coverage** | {h.evidence_coverage_pct:.0f}% |"
        )
        lines.append(
            f"| **Genericness** | {h.genericness_score:.0f}% |"
        )
        lines.append(f"| **Gate Status** | {h.gate_status.upper()} |")
    if h.confidence_drivers:
        lines.append(
            f"| **Confidence Drivers** | {'; '.join(h.confidence_drivers)} |"
        )
    lines.append("")

    # Identity verification warning
    if brief.verify_first:
        lines.append("> **\u26a0\ufe0f Identity Lock < 70 — Verify these facts before "
                      "relying on public claims:**")
        for vf in brief.verify_first:
            lines.append(f"> - {vf.fact} (confidence: {vf.current_confidence})")
        lines.append("")

    # ── B) Relationship & Interaction Snapshot ──
    lines.append("## Relationship & Interaction Snapshot")
    lines.append("")

    # Relationship Context
    rc = brief.relationship_context
    if rc.role or rc.influence_level or rc.relationship_health:
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
    lines.append("### Last Contact")
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
            lines.append("**Their commitments:**")
            for c in li.commitments:
                lines.append(f"- {c}")
    else:
        lines.append("*Unknown \u2013 no interaction data available*")
    lines.append("")

    # Interaction History
    if brief.interaction_history:
        lines.append("### Recent Interactions")
        lines.append("")
        for ix in brief.interaction_history[:10]:
            date_str = ""
            if ix.date:
                date_str = (
                    f"**{ix.date.strftime('%Y-%m-%d') if hasattr(ix.date, 'strftime') else str(ix.date)[:10]}** \u2013 "
                )
            lines.append(f"- {date_str}{ix.summary}{_cite(ix.citations)}")
        lines.append("")

    # ── C) Open Loops & Commitments ──
    lines.append("## Open Loops & Commitments")
    lines.append("")
    if brief.open_loops:
        lines.append("| Item | Owner | Due | Status | Evidence |")
        lines.append("|------|-------|-----|--------|----------|")
        for ol in brief.open_loops:
            owner = ol.owner or "\u2014"
            due = ol.due_date or "\u2014"
            cite = _cite(ol.citations).strip() if ol.citations else "\u2014"
            lines.append(
                f"| {ol.description} | {owner} | {due} | {ol.status} | {cite} |"
            )
    else:
        lines.append("*No open loops identified*")
    lines.append("")

    # ── D) Watchouts & Risks ──
    lines.append("## Watchouts & Risks")
    lines.append("")
    if brief.watchouts:
        for w in brief.watchouts:
            severity_icon = {
                "high": "\U0001F534", "medium": "\U0001F7E1", "low": "\U0001F7E2"
            }.get(w.severity, "\u26AA")
            lines.append(
                f"- {severity_icon} **{w.severity.upper()}**: "
                f"{w.description}{_cite(w.citations)}"
            )
    else:
        lines.append("*No watchouts identified*")
    lines.append("")

    # ── E) What I Must Cover ──
    lines.append("## What I Must Cover")
    lines.append("")
    if brief.what_to_cover:
        for wtc in brief.what_to_cover:
            lines.append(f"- {wtc.item}{_cite(wtc.citations)}")
            if wtc.rationale:
                lines.append(f"  *Rationale: {wtc.rationale}*")
    elif brief.meeting_objectives:
        for mo in brief.meeting_objectives:
            lines.append(f"- **{mo.objective}**")
            lines.append(
                f"  - *Measurable outcome*: "
                f"{mo.measurable_outcome}{_cite(mo.citations)}"
            )
    else:
        lines.append("*Unknown \u2013 insufficient evidence to determine agenda items*")
    lines.append("")

    # ── F) Leverage Plan ──
    lines.append("## Leverage Plan")
    lines.append("")

    # Leverage questions (prefer detailed, fall back to legacy)
    if brief.leverage_questions:
        lines.append("**Questions to ask:**")
        for i, lq in enumerate(brief.leverage_questions[:3], 1):
            lines.append(f"{i}. {lq.question}{_cite(lq.citations)}")
            if lq.rationale:
                lines.append(f"   *{lq.rationale}*")
        lines.append("")
    elif brief.leverage_plan.questions:
        lines.append("**Questions to ask:**")
        for i, q in enumerate(brief.leverage_plan.questions[:3], 1):
            lines.append(f"{i}. {q}")
        lines.append("")

    # Proof points (prefer detailed, fall back to legacy)
    if brief.proof_points:
        lines.append("**Proof points to deploy:**")
        for i, pp in enumerate(brief.proof_points[:2], 1):
            lines.append(f"{i}. {pp.point}{_cite(pp.citations)}")
            if pp.why_it_matters:
                lines.append(f"   *{pp.why_it_matters}*")
        lines.append("")
    elif brief.leverage_plan.proof_points:
        lines.append("**Proof points to deploy:**")
        for i, pp in enumerate(brief.leverage_plan.proof_points[:2], 1):
            lines.append(f"{i}. {pp}")
        lines.append("")

    # Tension to surface
    if brief.tension_to_surface_detail:
        td = brief.tension_to_surface_detail
        lines.append(
            f"**Tension to surface:** {_tag(td.evidence_tag)} "
            f"{td.claim}{_cite(td.citations)}"
        )
        lines.append("")
    elif brief.leverage_plan.tension_to_surface:
        lines.append(
            f"**Tension to surface:** {brief.leverage_plan.tension_to_surface}"
        )
        lines.append("")

    # Direct ask
    if brief.direct_ask:
        da = brief.direct_ask
        lines.append(
            f"**Direct ask:** {_tag(da.evidence_tag)} "
            f"{da.claim}{_cite(da.citations)}"
        )
        lines.append("")
    elif brief.leverage_plan.ask:
        lines.append(f"**Direct ask:** {brief.leverage_plan.ask}")
        lines.append("")

    has_leverage = (
        brief.leverage_questions or brief.proof_points
        or brief.tension_to_surface_detail or brief.direct_ask
        or brief.leverage_plan.questions or brief.leverage_plan.proof_points
    )
    if not has_leverage:
        lines.append("*Unknown \u2013 insufficient evidence for leverage plan*")
        lines.append("")

    # ── G) Suggested Agenda ──
    if brief.agenda.variants:
        lines.append("## Suggested Agenda")
        lines.append("")
        for variant in brief.agenda.variants:
            lines.append(f"### {variant.duration_minutes}-Minute Version")
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

    # ── H) Unknowns That Matter ──
    lines.append("## Unknowns That Matter")
    lines.append("")
    if brief.information_gaps:
        lines.append("| Unknown | Why It Matters | How to Resolve | Suggested Question |")
        lines.append("|---------|----------------|----------------|--------------------|")
        for ig in brief.information_gaps:
            question = ig.suggested_question or "\u2014"
            how = ig.how_to_resolve or "\u2014"
            lines.append(
                f"| {ig.gap} | {ig.strategic_impact} | {how} | {question} |"
            )
    else:
        lines.append("*No material unknowns identified*")
    lines.append("")

    # ── I) Evidence Index ──
    lines.append("## Evidence Index")
    lines.append("")
    if brief.evidence_index:
        lines.append("| # | Type | ID | Date | Excerpt | Link |")
        lines.append("|---|------|----|------|---------|------|")
        _dash = "\u2014"
        for i, ev in enumerate(brief.evidence_index, 1):
            date_str = ev.timestamp.strftime("%Y-%m-%d") if ev.timestamp else _dash
            excerpt = ev.excerpt[:80] if ev.excerpt else _dash
            link = ev.link or _dash
            lines.append(
                f"| {i} | {ev.source_type.value} | `{ev.source_id}` "
                f"| {date_str} | {excerpt} | {link} |"
            )
    elif brief.appendix_evidence:
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

    # ── Engine Improvements (internal) ──
    ei = brief.engine_improvements
    if ei.missing_signals or ei.recommended_data_sources or ei.capture_fields:
        lines.append("---")
        lines.append("")
        lines.append("## Engine Improvement Recommendations")
        lines.append("")
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

    lines.append("---")
    lines.append("*Generated by Pre-Call Intelligence Briefing Engine*")

    return "\n".join(lines)
