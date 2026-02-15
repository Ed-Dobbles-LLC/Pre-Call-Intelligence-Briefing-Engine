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
    EvidenceItem,
    EvidenceTag,
    HeaderSection,
    IncentiveStructure,
    InformationGap,
    InteractionRecord,
    LeveragePlan,
    MeetingDelta,
    MeetingObjective,
    OpenLoop,
    PowerInfluenceMap,
    RelationshipContext,
    SourceType,
    StrategicTension,
    TaggedClaim,
    Watchout,
)
from app.retrieve.retriever import RetrievedEvidence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a Strategic Intelligence Analyst producing a decision-grade \
executive dossier. You are NOT writing a summary. You are building an executive \
intelligence layer.

Your output must be:
- Decision-grade: every claim helps an executive negotiate, pressure-test, or decide
- Behavioral: based on observed patterns, not personality adjectives
- Strategic: focused on incentives, power dynamics, and tensions
- Signal-dense: no padding, no filler, no corporate fluff
- Explicit about uncertainty: every claim tagged with evidence confidence

ABSOLUTE RULES:

1. EVIDENCE DISCIPLINE — Tag every claim with exactly one of:
   - VERIFIED_MEETING: explicitly stated in meeting transcript
   - VERIFIED_PUBLIC: explicitly documented in public source
   - INFERRED_HIGH: high-confidence analytical inference from multiple signals
   - INFERRED_LOW: low-confidence inference from weak or single signals
   - UNKNOWN: no supporting evidence — label it, do not guess

2. CITATION FORMAT — Every factual claim MUST cite its source:
   [SOURCE:source_type:source_id:date]

3. NO HALLUCINATION — If you cannot support a claim with evidence, write "Unknown" \
and tag it UNKNOWN. An explicit gap is more valuable than plausible fiction.

4. NO GENERIC OUTPUT — Before including any statement, apply this test: \
"Could this statement apply to 50% of executives in this industry?" If yes, delete it.

5. NO PERSONALITY ADJECTIVES — No "strategic leader", "data-driven", "passionate". \
Only structural observations backed by evidence.

6. NO CORPORATE FLUFF — No buzzwords, no motivational tone, no flattery.

7. BE CONCISE — Bullets over paragraphs. Evidence over adjectives. Gaps over guesses.

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

## REQUIRED OUTPUT — STRATEGIC OPERATING MODEL (JSON)

Return a JSON object with ALL of these keys. For every tagged_claim object, use:
{{"claim": "<text>", "evidence_tag": "VERIFIED_MEETING|VERIFIED_PUBLIC|INFERRED_HIGH|INFERRED_LOW|UNKNOWN", "citations": [<citation objects>]}}

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
    "commitments": ["<commitment1>"],
    "citations": [<citation objects>]
  }},

  "interaction_history": [
    {{"date": "<ISO date>", "summary": "<summary>", "commitments": [], "citations": [<citation objects>]}}
  ],

  "open_loops": [
    {{"description": "<action item>", "owner": "<who>", "due_date": "<date or null>", "status": "open", "citations": [<citation objects>]}}
  ],

  "watchouts": [
    {{"description": "<risk/concern>", "severity": "low|medium|high", "citations": [<citation objects>]}}
  ],

  "meeting_objectives": [
    {{"objective": "<what to achieve>", "measurable_outcome": "<how to measure>", "citations": [<citation objects>]}}
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
      {{"duration_minutes": 20, "blocks": [{{"minutes": 5, "label": "Opening", "notes": "..."}}]}},
      {{"duration_minutes": 30, "blocks": [...]}},
      {{"duration_minutes": 45, "blocks": [...]}}
    ]
  }},

  "strategic_positioning": [
    <5-7 tagged_claim objects answering: What game is this person playing? What incentives/constraints are visible? What outcomes are they measured on? No adjectives, only structural observations.>
  ],

  "power_map": {{
    "formal_authority": <tagged_claim or null>,
    "informal_influence": <tagged_claim or null>,
    "revenue_control": <tagged_claim or null>,
    "decision_gate_ownership": <tagged_claim or null>,
    "needs_to_impress": <tagged_claim or null>,
    "veto_risk": <tagged_claim or null>
  }},

  "incentive_structure": {{
    "short_term": [<tagged_claims — what they need in next 0-3 months>],
    "medium_term": [<tagged_claims — 3-12 month horizon>],
    "career": [<tagged_claims — career trajectory incentives>],
    "risk_exposure": [<tagged_claims — what could go wrong for them>],
    "personal_wins": [<tagged_claims — where they personally benefit>],
    "personal_losses": [<tagged_claims — where they personally lose>]
  }},

  "cognitive_patterns": [
    {{
      "pattern_type": "<e.g. Repeated language | Framing device | Growth vs control bias | Abstraction level>",
      "observation": "<specific observation>",
      "evidence_quote": "<verbatim quote if available>",
      "evidence_tag": "VERIFIED_MEETING|INFERRED_HIGH|etc",
      "citations": [<citation objects>]
    }}
  ],

  "strategic_tensions": [
    {{
      "tension": "<e.g. Revenue target realism vs delivery capacity>",
      "evidence": "<what from the evidence reveals this tension>",
      "evidence_tag": "VERIFIED_MEETING|INFERRED_HIGH|etc",
      "citations": [<citation objects>]
    }}
  ],

  "behavioral_forecasts": [
    {{
      "scenario": "If <specific scenario>",
      "predicted_reaction": "Likely reaction: <specific prediction>",
      "reasoning": "<evidence that supports this prediction>",
      "citations": [<citation objects>]
    }}
  ],

  "information_gaps": [
    {{
      "gap": "<what is unknown>",
      "strategic_impact": "<why this gap matters for decision-making>"
    }}
  ],

  "conversation_strategy": {{
    "leverage_angles": [<3 tagged_claims — angles mapped to their incentive structure>],
    "stress_tests": [<2 tagged_claims — pressure-test angles>],
    "credibility_builders": [<2 tagged_claims — what builds trust with them>],
    "contrarian_wedge": <1 tagged_claim — intelligent challenge that earns respect>,
    "collaboration_vector": <1 tagged_claim — highest-upside partnership angle>
  }},

  "meeting_delta": {{
    "alignments": [<tagged_claims — where public persona matches meeting signals>],
    "divergences": [<tagged_claims — where public persona contradicts meeting signals>]
  }},

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
  "excerpt": "<exact text from evidence>",
  "snippet_hash": "<sha256 of excerpt>",
  "link": null
}}

QUALITY GATE — Before returning, verify:
1. Could any strategic_positioning bullet apply to 50% of executives? If yes, rewrite it.
2. Does every behavioral_forecast cite specific evidence? If not, tag INFERRED_LOW.
3. Are there any personality adjectives without evidence? If yes, delete them.
4. Does every conversation_strategy angle map to a specific incentive? If not, fix it.
5. Are information_gaps strategically material? Remove generic gaps like "career history missing".

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
            information_gaps=[
                InformationGap(
                    gap="No meeting transcripts or emails available for this contact",
                    strategic_impact="Cannot assess incentives, power dynamics, or behavioral "
                    "patterns without interaction data",
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

    # Information gaps
    information_gaps = []
    for ig_raw in raw.get("information_gaps", []):
        information_gaps.append(
            InformationGap(
                gap=ig_raw.get("gap", ""),
                strategic_impact=ig_raw.get("strategic_impact", ""),
            )
        )

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
        meeting_objectives=objectives,
        leverage_plan=leverage_plan,
        agenda=agenda,
        appendix_evidence=_build_evidence_appendix(evidence),
        strategic_positioning=strategic_positioning,
        power_map=power_map,
        incentive_structure=incentive_structure,
        cognitive_patterns=cognitive_patterns,
        strategic_tensions=strategic_tensions,
        behavioral_forecasts=behavioral_forecasts,
        information_gaps=information_gaps,
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
