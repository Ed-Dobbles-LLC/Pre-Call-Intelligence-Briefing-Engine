"""Brief generation: convert retrieved evidence into a cited JSON + markdown brief.

This is the core intelligence layer.  It:
1. Assembles a comprehensive prompt with all retrieved evidence
2. Calls the LLM with strict instructions to cite every claim
3. Post-processes the response to enforce citation integrity
4. Produces both BriefOutput (JSON) and markdown
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime

from app.clients.openai_client import LLMClient
from app.models import (
    Agenda,
    AgendaBlock,
    AgendaVariant,
    BriefOutput,
    Citation,
    EvidenceItem,
    HeaderSection,
    InteractionRecord,
    LeveragePlan,
    MeetingObjective,
    OpenLoop,
    RelationshipContext,
    SourceType,
    Watchout,
)
from app.retrieve.retriever import RetrievedEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Pre-Call Intelligence Analyst.  Your job is to produce
a decision-grade meeting brief from the evidence provided.

ABSOLUTE RULES:
1. Every factual claim MUST cite its source using [SOURCE:source_type:source_id:date] format.
2. If you cannot find evidence for something, write "Unknown – no evidence found in available data."
3. NEVER fabricate, infer dates, or hallucinate facts not in the evidence.
4. Be concise and actionable.  This brief will be read 5 minutes before a call.
5. Confidence score: 0.0 if no data, scale linearly with evidence breadth/recency.

Respond with a single JSON object matching the schema below.  Do not include markdown fences."""

USER_PROMPT_TEMPLATE = """
## BRIEF REQUEST
- Person: {person}
- Company: {company}
- Topic: {topic}
- Meeting datetime: {meeting_datetime}

## EVIDENCE ({evidence_count} sources)

### INTERACTIONS (most recent first)
{interactions_text}

### OPEN ACTION ITEMS
{action_items_text}

### CONCERN / OBJECTION SNIPPETS
{concerns_text}

## REQUIRED OUTPUT (JSON)
Return a JSON object with these exact keys:

{{
  "confidence_score": <float 0-1>,
  "relationship_context": {{
    "role": "<role or null>",
    "company": "<company or null>",
    "influence_level": "<low|medium|high or null>",
    "influence_level_inferred": <bool>,
    "relationship_health": "<cold|warm|hot or null>",
    "relationship_health_inferred": <bool>,
    "citations": [<citation objects>]
  }},
  "last_interaction": {{
    "date": "<ISO date or null>",
    "summary": "<what happened>",
    "commitments": ["<commitment1>", ...],
    "citations": [<citation objects>]
  }},
  "interaction_history": [
    {{
      "date": "<ISO date>",
      "summary": "<summary>",
      "commitments": [],
      "citations": [<citation objects>]
    }}
  ],
  "open_loops": [
    {{
      "description": "<action item>",
      "owner": "<who>",
      "due_date": "<date or null>",
      "status": "open",
      "citations": [<citation objects>]
    }}
  ],
  "watchouts": [
    {{
      "description": "<risk/concern>",
      "severity": "low|medium|high",
      "citations": [<citation objects>]
    }}
  ],
  "meeting_objectives": [
    {{
      "objective": "<what to achieve>",
      "measurable_outcome": "<how to measure success>",
      "citations": [<citation objects>]
    }}
  ],
  "leverage_plan": {{
    "questions": ["<q1>", "<q2>", "<q3>"],
    "proof_points": ["<pp1>", "<pp2>"],
    "tension_to_surface": "<tension>",
    "ask": "<the ask>",
    "citations": [<citation objects>]
  }},
  "agenda": {{
    "variants": [
      {{
        "duration_minutes": 20,
        "blocks": [{{"minutes": 5, "label": "Opening", "notes": "..."}}]
      }},
      {{
        "duration_minutes": 30,
        "blocks": [...]
      }},
      {{
        "duration_minutes": 45,
        "blocks": [...]
      }}
    ]
  }}
}}

Each citation object:
{{
  "source_type": "fireflies|gmail",
  "source_id": "<id>",
  "timestamp": "<ISO datetime>",
  "excerpt": "<exact text from evidence>",
  "snippet_hash": "<sha256 of excerpt>",
  "link": null
}}

If you have NO evidence for a section, use:
- For strings: "Unknown – no evidence found in available data"
- For lists: empty []
- For objects: include the key with null/empty values
- confidence_score: 0.0
"""


def _format_interactions(evidence: RetrievedEvidence) -> str:
    if not evidence.interactions:
        return "No interactions found."
    parts = []
    for ix in evidence.interactions[:20]:  # Cap at 20 for prompt size
        parts.append(
            f"---\n"
            f"Source: {ix['source_type']} | ID: {ix['source_id']} | Date: {ix['date']}\n"
            f"Title: {ix['title']}\n"
            f"Summary: {ix['summary']}\n"
            f"Participants: {', '.join(ix['participants'])}\n"
            f"Action items: {json.dumps(ix['action_items'])}\n"
            f"Body preview:\n{ix['body_preview']}\n"
        )
    return "\n".join(parts)


def _format_action_items(evidence: RetrievedEvidence) -> str:
    if not evidence.action_items:
        return "No action items found."
    parts = []
    for ai in evidence.action_items[:30]:
        parts.append(
            f"- {ai['description']} "
            f"[{ai['source_type']}:{ai['source_id']}:{ai['date']}]"
        )
    return "\n".join(parts)


def _format_concerns(evidence: RetrievedEvidence) -> str:
    if not evidence.concern_snippets:
        return "No concerns/objections found."
    parts = []
    for c in evidence.concern_snippets[:20]:
        parts.append(
            f"- [{c['keyword']}] \"{c['snippet']}\" "
            f"[{c['source_type']}:{c['source_id']}:{c['date']}]"
        )
    return "\n".join(parts)


def _compute_snippet_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _parse_citation(raw: dict) -> Citation:
    """Parse a citation dict from the LLM response, filling in snippet_hash."""
    excerpt = raw.get("excerpt", "")
    return Citation(
        source_type=SourceType(raw.get("source_type", "fireflies")),
        source_id=raw.get("source_id", "unknown"),
        timestamp=raw.get("timestamp") or datetime.utcnow().isoformat(),
        excerpt=excerpt,
        snippet_hash=raw.get("snippet_hash") or _compute_snippet_hash(excerpt),
        link=raw.get("link"),
    )


def _parse_citations(raw_list: list[dict] | None) -> list[Citation]:
    if not raw_list:
        return []
    return [_parse_citation(c) for c in raw_list]


def _build_evidence_appendix(evidence: RetrievedEvidence) -> list[EvidenceItem]:
    items = []
    for record in evidence.all_source_records:
        items.append(
            EvidenceItem(
                source_type=SourceType(record.source_type),
                source_id=record.source_id,
                title=record.title,
                date=record.date,
                link=record.link,
                excerpt_preview=(record.body or "")[:200],
            )
        )
    return items


def generate_brief(
    person: str | None,
    company: str | None,
    topic: str | None,
    meeting_datetime: datetime | None,
    evidence: RetrievedEvidence,
) -> BriefOutput:
    """Generate a Pre-Call Intelligence Brief from retrieved evidence.

    If there is no evidence at all, produces a minimal brief with
    confidence_score=0 and "Unknown" placeholders.
    """
    # Build header
    header = HeaderSection(
        person=person,
        company=company,
        topic=topic,
        meeting_datetime=meeting_datetime,
        data_sources_used=list(
            {r.source_type for r in evidence.all_source_records}
        ),
    )

    if not evidence.has_data:
        header.confidence_score = 0.0
        return BriefOutput(
            header=header,
            relationship_context=RelationshipContext(
                role="Unknown – no evidence found in available data",
            ),
            last_interaction=None,
            open_loops=[],
            watchouts=[],
            meeting_objectives=[
                MeetingObjective(
                    objective="Unknown – no evidence found in available data",
                    measurable_outcome="Unknown – no evidence found in available data",
                )
            ],
            leverage_plan=LeveragePlan(
                questions=["Unknown – no evidence found in available data"],
            ),
            agenda=Agenda(variants=[]),
            appendix_evidence=[],
        )

    # Build LLM prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        person=person or "Unknown",
        company=company or "Unknown",
        topic=topic or "General",
        meeting_datetime=meeting_datetime.isoformat() if meeting_datetime else "Not specified",
        evidence_count=evidence.source_count,
        interactions_text=_format_interactions(evidence),
        action_items_text=_format_action_items(evidence),
        concerns_text=_format_concerns(evidence),
    )

    # Call LLM
    try:
        llm = LLMClient()
        raw_json = llm.chat_json(SYSTEM_PROMPT, user_prompt)
    except Exception:
        logger.exception("LLM call failed – producing degraded brief from raw evidence")
        return _build_fallback_brief(header, evidence)

    # Parse LLM response into BriefOutput
    return _parse_llm_response(raw_json, header, evidence)


def _parse_llm_response(
    raw: dict,
    header: HeaderSection,
    evidence: RetrievedEvidence,
) -> BriefOutput:
    """Parse the LLM JSON response into a validated BriefOutput."""
    header.confidence_score = float(raw.get("confidence_score", 0.0))

    # Relationship context
    rc_raw = raw.get("relationship_context", {})
    relationship_context = RelationshipContext(
        role=rc_raw.get("role"),
        company=rc_raw.get("company"),
        influence_level=rc_raw.get("influence_level"),
        influence_level_inferred=rc_raw.get("influence_level_inferred", False),
        relationship_health=rc_raw.get("relationship_health"),
        relationship_health_inferred=rc_raw.get("relationship_health_inferred", False),
        citations=_parse_citations(rc_raw.get("citations")),
    )

    # Last interaction
    li_raw = raw.get("last_interaction")
    last_interaction = None
    if li_raw:
        last_interaction = InteractionRecord(
            date=li_raw.get("date"),
            summary=li_raw.get("summary", "Unknown"),
            commitments=li_raw.get("commitments", []),
            citations=_parse_citations(li_raw.get("citations")),
        )

    # Interaction history
    interaction_history = []
    for ih_raw in raw.get("interaction_history", []):
        interaction_history.append(
            InteractionRecord(
                date=ih_raw.get("date"),
                summary=ih_raw.get("summary", ""),
                commitments=ih_raw.get("commitments", []),
                citations=_parse_citations(ih_raw.get("citations")),
            )
        )

    # Open loops
    open_loops = []
    for ol_raw in raw.get("open_loops", []):
        open_loops.append(
            OpenLoop(
                description=ol_raw.get("description", ""),
                owner=ol_raw.get("owner"),
                due_date=ol_raw.get("due_date"),
                status=ol_raw.get("status", "open"),
                citations=_parse_citations(ol_raw.get("citations")),
            )
        )

    # Watchouts
    watchouts = []
    for w_raw in raw.get("watchouts", []):
        watchouts.append(
            Watchout(
                description=w_raw.get("description", ""),
                severity=w_raw.get("severity", "medium"),
                citations=_parse_citations(w_raw.get("citations")),
            )
        )

    # Meeting objectives
    objectives = []
    for mo_raw in raw.get("meeting_objectives", []):
        objectives.append(
            MeetingObjective(
                objective=mo_raw.get("objective", ""),
                measurable_outcome=mo_raw.get("measurable_outcome", ""),
                citations=_parse_citations(mo_raw.get("citations")),
            )
        )

    # Leverage plan
    lp_raw = raw.get("leverage_plan", {})
    leverage_plan = LeveragePlan(
        questions=lp_raw.get("questions", []),
        proof_points=lp_raw.get("proof_points", []),
        tension_to_surface=lp_raw.get("tension_to_surface"),
        ask=lp_raw.get("ask"),
        citations=_parse_citations(lp_raw.get("citations")),
    )

    # Agenda
    agenda_raw = raw.get("agenda", {})
    variants = []
    for v in agenda_raw.get("variants", []):
        blocks = [
            AgendaBlock(
                minutes=b.get("minutes", 5),
                label=b.get("label", ""),
                notes=b.get("notes"),
            )
            for b in v.get("blocks", [])
        ]
        variants.append(
            AgendaVariant(duration_minutes=v.get("duration_minutes", 30), blocks=blocks)
        )
    agenda = Agenda(variants=variants)

    return BriefOutput(
        header=header,
        relationship_context=relationship_context,
        last_interaction=last_interaction,
        interaction_history=interaction_history,
        open_loops=open_loops,
        watchouts=watchouts,
        meeting_objectives=objectives,
        leverage_plan=leverage_plan,
        agenda=agenda,
        appendix_evidence=_build_evidence_appendix(evidence),
    )


def _build_fallback_brief(
    header: HeaderSection,
    evidence: RetrievedEvidence,
) -> BriefOutput:
    """Build a minimal brief from raw evidence when LLM is unavailable."""
    header.confidence_score = 0.1

    last = None
    if evidence.last_interaction:
        li = evidence.last_interaction
        last = InteractionRecord(
            date=li.get("date"),
            summary=li.get("summary") or "See raw evidence",
            commitments=[],
            citations=[
                Citation(
                    source_type=SourceType(li["source_type"]),
                    source_id=li["source_id"],
                    timestamp=li.get("date") or datetime.utcnow().isoformat(),
                    excerpt=(li.get("summary") or "")[:200],
                    snippet_hash=_compute_snippet_hash(li.get("summary") or ""),
                )
            ],
        )

    open_loops = []
    for ai in evidence.action_items[:10]:
        open_loops.append(
            OpenLoop(
                description=ai["description"],
                citations=[
                    Citation(
                        source_type=SourceType(ai["source_type"]),
                        source_id=ai["source_id"],
                        timestamp=ai.get("date") or datetime.utcnow().isoformat(),
                        excerpt=ai["description"][:200],
                        snippet_hash=_compute_snippet_hash(ai["description"]),
                    )
                ],
            )
        )

    return BriefOutput(
        header=header,
        last_interaction=last,
        open_loops=open_loops,
        appendix_evidence=_build_evidence_appendix(evidence),
    )
