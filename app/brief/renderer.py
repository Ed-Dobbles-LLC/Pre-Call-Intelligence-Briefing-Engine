"""Render a BriefOutput as human-readable markdown."""

from __future__ import annotations

from app.models import BriefOutput, Citation


def _cite(citations: list[Citation]) -> str:
    """Format inline citation references."""
    if not citations:
        return ""
    refs = []
    for c in citations:
        date_str = c.timestamp.strftime("%Y-%m-%d") if hasattr(c.timestamp, "strftime") else str(c.timestamp)[:10]
        refs.append(f"[{c.source_type.value}:{c.source_id}:{date_str}]")
    return " " + " ".join(refs)


def render_markdown(brief: BriefOutput) -> str:
    """Convert a BriefOutput to a Markdown string."""
    lines: list[str] = []
    h = brief.header

    # ── Header ──
    lines.append(f"# Pre-Call Intelligence Brief")
    lines.append("")
    lines.append(f"| Field | Value |")
    lines.append(f"|-------|-------|")
    if h.person:
        lines.append(f"| **Person** | {h.person} |")
    if h.company:
        lines.append(f"| **Company** | {h.company} |")
    if h.topic:
        lines.append(f"| **Topic** | {h.topic} |")
    if h.meeting_datetime:
        lines.append(f"| **Meeting** | {h.meeting_datetime.strftime('%Y-%m-%d %H:%M')} |")
    lines.append(f"| **Generated** | {h.brief_generated_at.strftime('%Y-%m-%d %H:%M UTC')} |")
    lines.append(f"| **Confidence** | {h.confidence_score:.0%} |")
    lines.append(f"| **Sources** | {', '.join(h.data_sources_used) or 'none'} |")
    lines.append("")

    # ── Relationship Context ──
    rc = brief.relationship_context
    lines.append("## Relationship Context")
    lines.append("")
    if rc.role:
        inferred_tag = " *(inferred)*" if rc.influence_level_inferred else ""
        lines.append(f"- **Role**: {rc.role}{_cite(rc.citations)}")
    if rc.influence_level:
        tag = " *(inferred)*" if rc.influence_level_inferred else ""
        lines.append(f"- **Influence level**: {rc.influence_level}{tag}{_cite(rc.citations)}")
    if rc.relationship_health:
        tag = " *(inferred)*" if rc.relationship_health_inferred else ""
        lines.append(f"- **Relationship health**: {rc.relationship_health}{tag}{_cite(rc.citations)}")
    lines.append("")

    # ── Last Interaction ──
    lines.append("## Last Interaction")
    lines.append("")
    if brief.last_interaction:
        li = brief.last_interaction
        date_str = ""
        if li.date:
            date_str = f" ({li.date.strftime('%Y-%m-%d') if hasattr(li.date, 'strftime') else str(li.date)[:10]})"
        lines.append(f"{li.summary}{date_str}{_cite(li.citations)}")
        if li.commitments:
            lines.append("")
            lines.append("**Commitments:**")
            for c in li.commitments:
                lines.append(f"- {c}")
    else:
        lines.append("*Unknown – no interaction data available*")
    lines.append("")

    # ── Interaction History ──
    if brief.interaction_history:
        lines.append("## Recent Interaction History")
        lines.append("")
        for ix in brief.interaction_history[:10]:
            date_str = ""
            if ix.date:
                date_str = f"**{ix.date.strftime('%Y-%m-%d') if hasattr(ix.date, 'strftime') else str(ix.date)[:10]}** – "
            lines.append(f"- {date_str}{ix.summary}{_cite(ix.citations)}")
        lines.append("")

    # ── Open Loops ──
    lines.append("## Open Loops")
    lines.append("")
    if brief.open_loops:
        for ol in brief.open_loops:
            owner = f" (Owner: {ol.owner})" if ol.owner else ""
            due = f" [Due: {ol.due_date}]" if ol.due_date else ""
            lines.append(f"- {ol.description}{owner}{due}{_cite(ol.citations)}")
    else:
        lines.append("*No open loops identified*")
    lines.append("")

    # ── Watchouts ──
    lines.append("## Watchouts & Risks")
    lines.append("")
    if brief.watchouts:
        for w in brief.watchouts:
            severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(w.severity, "⚪")
            lines.append(f"- {severity_icon} **{w.severity.upper()}**: {w.description}{_cite(w.citations)}")
    else:
        lines.append("*No watchouts identified*")
    lines.append("")

    # ── Meeting Objectives ──
    lines.append("## Meeting Objectives")
    lines.append("")
    if brief.meeting_objectives:
        for mo in brief.meeting_objectives:
            lines.append(f"- **Objective**: {mo.objective}")
            lines.append(f"  - *Measurable outcome*: {mo.measurable_outcome}{_cite(mo.citations)}")
    else:
        lines.append("*Unknown – insufficient data to recommend objectives*")
    lines.append("")

    # ── Leverage Plan ──
    lines.append("## Leverage Plan")
    lines.append("")
    lp = brief.leverage_plan
    if lp.questions:
        lines.append("**Questions to ask:**")
        for i, q in enumerate(lp.questions, 1):
            lines.append(f"{i}. {q}")
        lines.append("")
    if lp.proof_points:
        lines.append("**Proof points:**")
        for pp in lp.proof_points:
            lines.append(f"- {pp}")
        lines.append("")
    if lp.tension_to_surface:
        lines.append(f"**Tension to surface:** {lp.tension_to_surface}")
        lines.append("")
    if lp.ask:
        lines.append(f"**The Ask:** {lp.ask}")
        lines.append("")
    if lp.citations:
        lines.append(f"*Sources: {_cite(lp.citations).strip()}*")
        lines.append("")

    # ── Agenda ──
    lines.append("## Suggested Agenda")
    lines.append("")
    if brief.agenda.variants:
        for variant in brief.agenda.variants:
            lines.append(f"### {variant.duration_minutes}-Minute Version")
            lines.append("")
            lines.append("| Time | Block | Notes |")
            lines.append("|------|-------|-------|")
            elapsed = 0
            for block in variant.blocks:
                lines.append(
                    f"| {elapsed}–{elapsed + block.minutes} min | {block.label} | {block.notes or ''} |"
                )
                elapsed += block.minutes
            lines.append("")
    else:
        lines.append("*No agenda generated – insufficient data*")
    lines.append("")

    # ── Appendix ──
    lines.append("## Appendix: Evidence Sources")
    lines.append("")
    if brief.appendix_evidence:
        lines.append("| # | Type | ID | Date | Title |")
        lines.append("|---|------|----|------|-------|")
        for i, ev in enumerate(brief.appendix_evidence, 1):
            date_str = ev.date.strftime("%Y-%m-%d") if ev.date else "—"
            lines.append(
                f"| {i} | {ev.source_type.value} | `{ev.source_id}` | {date_str} | {ev.title or '—'} |"
            )
    else:
        lines.append("*No evidence sources available*")
    lines.append("")

    lines.append("---")
    lines.append("*Generated by Pre-Call Intelligence Briefing Engine*")

    return "\n".join(lines)
