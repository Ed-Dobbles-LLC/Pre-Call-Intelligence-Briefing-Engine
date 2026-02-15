"""Brief generation: convert retrieved evidence into a strategic intelligence dossier.

This is the core intelligence layer.  It:
1. Assembles a comprehensive prompt with all retrieved evidence
2. Calls the LLM with strict instructions to produce a Strategic Operating Model
3. Post-processes the response to enforce citation integrity and evidence tagging
4. Produces both BriefOutput (JSON) and markdown

Every claim is tagged with evidence discipline markers:
- VERIFIED_MEETING: explicitly stated in meeting transcript
- VERIFIED_PUBLIC: explicitly documented in public source
- INFERRED_HIGH: high-confidence inference from evidence
- INFERRED_LOW: low-confidence inference
- UNKNOWN: no supporting evidence
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
    BehavioralForecast,
    BriefOutput,
    Citation,
    CognitivePattern,
    ConversationStrategy,
    EngineImprovement,
    EvidenceIndexEntry,
    EvidenceItem,
    EvidenceTag,
    HeaderSection,
    IncentiveStructure,
    InformationGap,
    InteractionRecord,
    LeverageQuestion,
    LeveragePlan,
    MeetingDelta,
    MeetingObjective,
    OpenLoop,
    PowerInfluenceMap,
    ProofPoint,
    RelationshipContext,
    SourceType,
    StrategicTension,
    TaggedClaim,
    Watchout,
    WhatToCoverItem,
)
from app.retrieve.retriever import RetrievedEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Pre-Call Intelligence Analyst. Your job is to produce a \
person-first, evidence-locked pre-call brief that an executive can trust before \
walking into a meeting.

Your output is NOT a company overview, NOT a strategic dossier, NOT a personality \
profile. It is a RELATIONSHIP BRIEF: what happened between us and this person, \
what is open, what to cover, what to ask.

ABSOLUTE RULES:

1. PERSON-FIRST — Every section is about the PERSON and your relationship with them. \
Company context only where it directly explains a person's behavior or commitment.

2. EVIDENCE DISCIPLINE — Tag every claim with exactly one of:
   - VERIFIED_MEETING: explicitly stated in meeting transcript
   - VERIFIED_PUBLIC: explicitly documented in public source
   - INFERRED_HIGH: high-confidence inference from multiple converging signals
   - INFERRED_LOW: low-confidence inference from weak or single signals
   - UNKNOWN: no supporting evidence — label it, do not guess

3. CITATION FORMAT — Every factual claim MUST cite its source:
   [SOURCE:source_type:source_id:date]

4. ZERO HALLUCINATION — If you cannot support a claim with evidence, write \
"Unknown" and add a Resolution Question. An explicit gap is more valuable than \
plausible fiction.

5. SEPARATE FACT vs INFERENCE — Facts have citations. Inferences MUST be labeled \
with evidence_tag and cite the upstream evidence they derive from.

6. NO GENERIC CONTENT — Before including any statement, test: "Could this apply \
to 50% of executives in this role?" If yes, delete it. No "likely", "may", "could", \
"generally", "typically" unless citing specific evidence.

7. NO SCENARIO PLANNING — No behavioral forecasts, no "if X then Y" predictions \
unless directly supported by cited evidence from past interactions.

8. NO CORPORATE FLUFF — No buzzwords, no personality adjectives, no flattery. \
No "strategic leader", "data-driven", "passionate about".

9. CONCISE — Target 1-2 pages. Bullets over paragraphs. Evidence over adjectives. \
Gaps over guesses. If you have nothing evidenced, say "Unknown" and move on.

Respond with a single JSON object. Do not include markdown fences."""


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

## REQUIRED OUTPUT — PRE-CALL INTELLIGENCE BRIEF (JSON)

Return a JSON object. Every bullet in sections B-F MUST have citations.
If evidence is missing, output "Unknown" + a Resolution Question — do NOT guess.

For every tagged_claim object, use:
{{"claim": "<text>", "evidence_tag": "VERIFIED_MEETING|VERIFIED_PUBLIC|INFERRED_HIGH|INFERRED_LOW|UNKNOWN", "citations": [<citation objects>]}}

{{
  "confidence_score": <float 0-1 — explain what drives this number>,
  "confidence_drivers": ["<driver1>", "<driver2>"],

  "relationship_context": {{
    "role": "<role or null — ONLY if evidenced>",
    "company": "<company or null>",
    "influence_level": "<low|medium|high or null>",
    "influence_level_inferred": <bool>,
    "relationship_health": "<cold|warm|hot or null>",
    "relationship_health_inferred": <bool>,
    "citations": [<citation objects>]
  }},

  "last_interaction": {{
    "date": "<ISO date or null>",
    "summary": "<3-5 bullet 'what happened' — person-first, cited>",
    "commitments": ["<their stated commitments — ONLY if evidenced>"],
    "citations": [<citation objects>]
  }},

  "interaction_history": [
    {{"date": "<ISO date>", "summary": "<summary>", "commitments": [], "citations": [<citation objects>]}}
  ],

  "open_loops": [
    {{"description": "<action item>", "owner": "<who>", "due_date": "<date or null>", "status": "open", "citations": [<citation objects>]}}
  ],

  "watchouts": [
    {{
      "description": "<ONLY evidenced risks: objections raised, tension moments, things I said that create risk>",
      "severity": "low|medium|high",
      "citations": [<citation objects>]
    }}
  ],

  "what_to_cover": [
    {{
      "item": "<3-7 bullets: tie to open loops, their priorities, or unresolved objections>",
      "rationale": "<why this matters — cite the source evidence>",
      "citations": [<citation objects>]
    }}
  ],

  "leverage_questions": [
    {{
      "question": "<leverage question>",
      "rationale": "<cites the upstream evidence that makes this question powerful>",
      "citations": [<citation objects>]
    }}
  ],

  "proof_points": [
    {{
      "point": "<proof point to deploy>",
      "why_it_matters": "<tie to their stated priorities — cite source>",
      "citations": [<citation objects>]
    }}
  ],

  "tension_to_surface_detail": {{
    "claim": "<1 tension to surface — cite the trigger>",
    "evidence_tag": "VERIFIED_MEETING|INFERRED_HIGH",
    "citations": [<citation objects>]
  }},

  "direct_ask": {{
    "claim": "<1 direct ask/decision to seek — cite why now>",
    "evidence_tag": "VERIFIED_MEETING|INFERRED_HIGH",
    "citations": [<citation objects>]
  }},

  "leverage_plan": {{
    "questions": ["<q1>", "<q2>", "<q3>"],
    "proof_points": ["<pp1>", "<pp2>"],
    "tension_to_surface": "<tension>",
    "ask": "<the ask>",
    "citations": [<citation objects>]
  }},

  "agenda": {{
    "variants": [
      {{"duration_minutes": 20, "blocks": [{{"minutes": 5, "label": "<label>", "notes": "<cites driver evidence>"}}]}},
      {{"duration_minutes": 30, "blocks": [...]}},
      {{"duration_minutes": 45, "blocks": [...]}}
    ]
  }},

  "information_gaps": [
    {{
      "gap": "<what is unknown>",
      "strategic_impact": "<why this gap matters>",
      "how_to_resolve": "<method to fill this gap>",
      "suggested_question": "<exact question to ask on the call>"
    }}
  ],

  "engine_improvements": {{
    "missing_signals": ["<signal type that was missing from the evidence>"],
    "recommended_data_sources": ["<specific data source that would improve accuracy>"],
    "capture_fields": ["<structured field the meeting tool should capture in future calls>"]
  }}
}}

Each citation object:
{{
  "source_type": "fireflies|gmail",
  "source_id": "<id>",
  "timestamp": "<ISO datetime>",
  "excerpt": "<exact quote from evidence, <=25 words>",
  "snippet_hash": "<sha256 of excerpt>",
  "link": null
}}

QUALITY GATE — Before returning, verify:
1. Is every bullet in last_interaction, open_loops, watchouts, what_to_cover, and leverage \
cited? If not, either add a citation or remove the bullet.
2. Does any statement apply to 50% of executives? Delete it.
3. Are there personality adjectives without evidence? Delete them.
4. Is there ANY speculative scenario planning without cited evidence? Delete it.
5. For every UNKNOWN, did you add a Resolution Question in information_gaps? If not, add one.
6. Are confidence_drivers explicit about what drives uncertainty?

If you have NO evidence for a section, use:
- For strings: "Unknown"
- For lists: empty []
- For objects: null
- confidence_score: 0.0
- Tag claims as UNKNOWN
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


def _parse_evidence_tag(raw: str | None) -> EvidenceTag:
    """Parse an evidence tag string from LLM output."""
    if not raw:
        return EvidenceTag.unknown
    tag_map = {
        "VERIFIED_MEETING": EvidenceTag.verified_meeting,
        "VERIFIED_PUBLIC": EvidenceTag.verified_public,
        "INFERRED_HIGH": EvidenceTag.inferred_high,
        "INFERRED_LOW": EvidenceTag.inferred_low,
        "UNKNOWN": EvidenceTag.unknown,
    }
    return tag_map.get(raw.upper().replace(" ", "_").replace("-", "_").replace("–", "_"),
                       EvidenceTag.unknown)


def _parse_tagged_claim(raw: dict | None) -> TaggedClaim | None:
    """Parse a tagged claim from LLM output."""
    if not raw:
        return None
    return TaggedClaim(
        claim=raw.get("claim", "Unknown"),
        evidence_tag=_parse_evidence_tag(raw.get("evidence_tag")),
        citations=_parse_citations(raw.get("citations")),
    )


def _parse_tagged_claims(raw_list: list[dict] | None) -> list[TaggedClaim]:
    """Parse a list of tagged claims from LLM output."""
    if not raw_list:
        return []
    claims = []
    for raw in raw_list:
        claim = _parse_tagged_claim(raw)
        if claim:
            claims.append(claim)
    return claims


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


def _build_evidence_index(evidence: RetrievedEvidence) -> list[EvidenceIndexEntry]:
    """Build the evidence index with <=25 word excerpts and snippet hashes."""
    entries = []
    for record in evidence.all_source_records:
        body = record.body or record.summary or ""
        # Take first ~25 words as excerpt
        words = body.split()[:25]
        excerpt = " ".join(words)
        if len(words) == 25:
            excerpt += "..."
        entries.append(
            EvidenceIndexEntry(
                source_type=SourceType(record.source_type),
                source_id=record.source_id,
                timestamp=record.date,
                excerpt=excerpt,
                snippet_hash=_compute_snippet_hash(excerpt),
                link=record.link,
            )
        )
    return entries


def generate_brief(
    person: str | None,
    company: str | None,
    topic: str | None,
    meeting_datetime: datetime | None,
    evidence: RetrievedEvidence,
) -> BriefOutput:
    """Generate a Strategic Intelligence Brief from retrieved evidence.

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
        header.confidence_drivers = ["No interaction data available"]
        header.gate_status = "failed"
        return BriefOutput(
            header=header,
            relationship_context=RelationshipContext(
                role="Unknown – no evidence found in available data",
            ),
            last_interaction=None,
            open_loops=[],
            watchouts=[],
            what_to_cover=[],
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
            information_gaps=[
                InformationGap(
                    gap="No meeting transcripts or emails available for this contact",
                    strategic_impact="Cannot assess prior commitments, open loops, or "
                    "relationship context without interaction data",
                    how_to_resolve="Ingest Fireflies transcripts and/or Gmail threads",
                    suggested_question="What's your current top priority this quarter?",
                )
            ],
            engine_improvements=EngineImprovement(
                missing_signals=["No interaction data available"],
                recommended_data_sources=[
                    "Meeting transcripts via Fireflies",
                    "Email correspondence via Gmail",
                    "LinkedIn profile for public positioning",
                ],
                capture_fields=[
                    "Risk appetite signals",
                    "Growth pressure markers",
                    "Incentive cues",
                ],
            ),
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

    # --- Strategic Operating Model sections ---

    # Strategic positioning
    strategic_positioning = _parse_tagged_claims(raw.get("strategic_positioning"))

    # Power & influence map
    pm_raw = raw.get("power_map", {}) or {}
    power_map = PowerInfluenceMap(
        formal_authority=_parse_tagged_claim(pm_raw.get("formal_authority")),
        informal_influence=_parse_tagged_claim(pm_raw.get("informal_influence")),
        revenue_control=_parse_tagged_claim(pm_raw.get("revenue_control")),
        decision_gate_ownership=_parse_tagged_claim(pm_raw.get("decision_gate_ownership")),
        needs_to_impress=_parse_tagged_claim(pm_raw.get("needs_to_impress")),
        veto_risk=_parse_tagged_claim(pm_raw.get("veto_risk")),
    )

    # Incentive structure
    is_raw = raw.get("incentive_structure", {}) or {}
    incentive_structure = IncentiveStructure(
        short_term=_parse_tagged_claims(is_raw.get("short_term")),
        medium_term=_parse_tagged_claims(is_raw.get("medium_term")),
        career=_parse_tagged_claims(is_raw.get("career")),
        risk_exposure=_parse_tagged_claims(is_raw.get("risk_exposure")),
        personal_wins=_parse_tagged_claims(is_raw.get("personal_wins")),
        personal_losses=_parse_tagged_claims(is_raw.get("personal_losses")),
    )

    # Cognitive patterns
    cognitive_patterns = []
    for cp_raw in raw.get("cognitive_patterns", []):
        cognitive_patterns.append(
            CognitivePattern(
                pattern_type=cp_raw.get("pattern_type", "Unknown"),
                observation=cp_raw.get("observation", ""),
                evidence_quote=cp_raw.get("evidence_quote"),
                evidence_tag=_parse_evidence_tag(cp_raw.get("evidence_tag")),
                citations=_parse_citations(cp_raw.get("citations")),
            )
        )

    # Strategic tensions
    strategic_tensions = []
    for st_raw in raw.get("strategic_tensions", []):
        strategic_tensions.append(
            StrategicTension(
                tension=st_raw.get("tension", ""),
                evidence=st_raw.get("evidence", ""),
                evidence_tag=_parse_evidence_tag(st_raw.get("evidence_tag")),
                citations=_parse_citations(st_raw.get("citations")),
            )
        )

    # Behavioral forecasts
    behavioral_forecasts = []
    for bf_raw in raw.get("behavioral_forecasts", []):
        behavioral_forecasts.append(
            BehavioralForecast(
                scenario=bf_raw.get("scenario", ""),
                predicted_reaction=bf_raw.get("predicted_reaction", ""),
                reasoning=bf_raw.get("reasoning", ""),
                citations=_parse_citations(bf_raw.get("citations")),
            )
        )

    # Information gaps (with resolution questions)
    information_gaps = []
    for ig_raw in raw.get("information_gaps", []):
        information_gaps.append(
            InformationGap(
                gap=ig_raw.get("gap", ""),
                strategic_impact=ig_raw.get("strategic_impact", ""),
                how_to_resolve=ig_raw.get("how_to_resolve", ""),
                suggested_question=ig_raw.get("suggested_question", ""),
            )
        )

    # --- Person-first sections ---

    # What to cover
    what_to_cover = []
    for wtc_raw in raw.get("what_to_cover", []):
        what_to_cover.append(
            WhatToCoverItem(
                item=wtc_raw.get("item", ""),
                rationale=wtc_raw.get("rationale", ""),
                citations=_parse_citations(wtc_raw.get("citations")),
            )
        )

    # Leverage questions (detailed, per-item citations)
    leverage_questions = []
    for lq_raw in raw.get("leverage_questions", []):
        leverage_questions.append(
            LeverageQuestion(
                question=lq_raw.get("question", ""),
                rationale=lq_raw.get("rationale", ""),
                citations=_parse_citations(lq_raw.get("citations")),
            )
        )

    # Proof points (detailed, per-item citations)
    proof_points_detail = []
    for pp_raw in raw.get("proof_points", []):
        if isinstance(pp_raw, dict):
            proof_points_detail.append(
                ProofPoint(
                    point=pp_raw.get("point", ""),
                    why_it_matters=pp_raw.get("why_it_matters", ""),
                    citations=_parse_citations(pp_raw.get("citations")),
                )
            )

    # Tension to surface (detailed)
    tension_detail = _parse_tagged_claim(raw.get("tension_to_surface_detail"))

    # Direct ask
    direct_ask = _parse_tagged_claim(raw.get("direct_ask"))

    # Confidence drivers
    header.confidence_drivers = raw.get("confidence_drivers", [])

    # Conversation strategy
    cs_raw = raw.get("conversation_strategy", {}) or {}
    conversation_strategy = ConversationStrategy(
        leverage_angles=_parse_tagged_claims(cs_raw.get("leverage_angles")),
        stress_tests=_parse_tagged_claims(cs_raw.get("stress_tests")),
        credibility_builders=_parse_tagged_claims(cs_raw.get("credibility_builders")),
        contrarian_wedge=_parse_tagged_claim(cs_raw.get("contrarian_wedge")),
        collaboration_vector=_parse_tagged_claim(cs_raw.get("collaboration_vector")),
    )

    # Meeting delta
    md_raw = raw.get("meeting_delta", {}) or {}
    meeting_delta = MeetingDelta(
        alignments=_parse_tagged_claims(md_raw.get("alignments")),
        divergences=_parse_tagged_claims(md_raw.get("divergences")),
    )

    # Engine improvements
    ei_raw = raw.get("engine_improvements", {}) or {}
    engine_improvements = EngineImprovement(
        missing_signals=ei_raw.get("missing_signals", []),
        recommended_data_sources=ei_raw.get("recommended_data_sources", []),
        capture_fields=ei_raw.get("capture_fields", []),
    )

    return BriefOutput(
        header=header,
        relationship_context=relationship_context,
        last_interaction=last_interaction,
        interaction_history=interaction_history,
        open_loops=open_loops,
        watchouts=watchouts,
        what_to_cover=what_to_cover,
        meeting_objectives=objectives,
        leverage_plan=leverage_plan,
        leverage_questions=leverage_questions,
        proof_points=proof_points_detail,
        tension_to_surface_detail=tension_detail,
        direct_ask=direct_ask,
        agenda=agenda,
        information_gaps=information_gaps,
        evidence_index=_build_evidence_index(evidence),
        appendix_evidence=_build_evidence_appendix(evidence),
        strategic_positioning=strategic_positioning,
        power_map=power_map,
        incentive_structure=incentive_structure,
        cognitive_patterns=cognitive_patterns,
        strategic_tensions=strategic_tensions,
        behavioral_forecasts=behavioral_forecasts,
        conversation_strategy=conversation_strategy,
        meeting_delta=meeting_delta,
        engine_improvements=engine_improvements,
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
        information_gaps=[
            InformationGap(
                gap="LLM unavailable — strategic analysis could not be generated",
                strategic_impact="Only raw evidence is available; no behavioral or "
                "incentive analysis was performed",
            )
        ],
        engine_improvements=EngineImprovement(
            missing_signals=["LLM-based analysis unavailable"],
            recommended_data_sources=["Verify OpenAI API key configuration"],
        ),
    )
