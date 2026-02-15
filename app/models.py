"""Pydantic models for the Pre-Call Intelligence Brief output schema.

These models define the canonical JSON shape of a person-first, evidence-locked
pre-call brief. Every claim-bearing section carries Citation objects so the
consumer can trace every bullet back to a stored source record.

The brief prioritises prior interactions (transcripts + emails) over general
company speculation. Every claim is tagged with evidence discipline markers.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Citation & Evidence Discipline
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    fireflies = "fireflies"
    gmail = "gmail"


class EvidenceTag(str, Enum):
    """Evidence discipline tag — every claim must carry one."""
    verified_meeting = "VERIFIED_MEETING"
    verified_public = "VERIFIED_PUBLIC"
    inferred_high = "INFERRED_HIGH"
    inferred_low = "INFERRED_LOW"
    unknown = "UNKNOWN"


class Citation(BaseModel):
    """Pointer back to the evidence behind a claim."""
    source_type: SourceType
    source_id: str = Field(..., description="Fireflies transcript ID or Gmail message ID")
    timestamp: datetime
    excerpt: str = Field(..., description="Verbatim excerpt supporting the claim")
    snippet_hash: str = Field(..., description="SHA-256 of the excerpt for dedup / audit")
    link: Optional[str] = Field(None, description="Deep-link if available")
    excerpt_start: Optional[int] = Field(None, description="Start char offset in source")
    excerpt_end: Optional[int] = Field(None, description="End char offset in source")


class TaggedClaim(BaseModel):
    """A single claim with evidence discipline tagging."""
    claim: str
    evidence_tag: EvidenceTag = EvidenceTag.unknown
    citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Legacy brief sections (preserved for backward compatibility)
# ---------------------------------------------------------------------------

class HeaderSection(BaseModel):
    person: Optional[str] = None
    company: Optional[str] = None
    topic: Optional[str] = None
    meeting_datetime: Optional[datetime] = None
    brief_generated_at: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="0 = no data, 1 = rich evidence across multiple sources",
    )
    data_sources_used: list[str] = Field(default_factory=list)
    # Quality gate scores (populated by pipeline)
    identity_lock_score: float = Field(
        0.0, ge=0.0, le=100.0,
        description="0-100 identity confidence from disambiguation",
    )
    evidence_coverage_pct: float = Field(
        0.0, ge=0.0, le=100.0,
        description="% of substantive sentences with evidence citations",
    )
    genericness_score: float = Field(
        0.0, ge=0.0, le=100.0,
        description="0=clean, 100=all generic filler",
    )
    confidence_drivers: list[str] = Field(
        default_factory=list,
        description="Explicit reasons behind the confidence score",
    )
    gate_status: str = Field(
        "not_run",
        description="passed | failed | constrained | not_run",
    )


class RelationshipContext(BaseModel):
    role: Optional[str] = None
    company: Optional[str] = None
    influence_level: Optional[str] = None
    influence_level_inferred: bool = False
    relationship_health: Optional[str] = None
    relationship_health_inferred: bool = False
    citations: list[Citation] = Field(default_factory=list)


class InteractionRecord(BaseModel):
    date: Optional[datetime] = None
    summary: str
    commitments: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)


class OpenLoop(BaseModel):
    description: str
    owner: Optional[str] = None
    due_date: Optional[str] = None
    status: str = "open"
    citations: list[Citation] = Field(default_factory=list)


class Watchout(BaseModel):
    description: str
    severity: str = "medium"  # low | medium | high
    citations: list[Citation] = Field(default_factory=list)


class MeetingObjective(BaseModel):
    objective: str
    measurable_outcome: str
    citations: list[Citation] = Field(default_factory=list)


class LeveragePlan(BaseModel):
    questions: list[str] = Field(default_factory=list, max_length=3)
    proof_points: list[str] = Field(default_factory=list, max_length=2)
    tension_to_surface: Optional[str] = None
    ask: Optional[str] = None
    citations: list[Citation] = Field(default_factory=list)


class AgendaVariant(BaseModel):
    duration_minutes: int
    blocks: list[AgendaBlock] = Field(default_factory=list)


class AgendaBlock(BaseModel):
    minutes: int
    label: str
    notes: Optional[str] = None


# Forward-ref fix (AgendaVariant references AgendaBlock defined after it)
AgendaVariant.model_rebuild()


class Agenda(BaseModel):
    variants: list[AgendaVariant] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    source_type: SourceType
    source_id: str
    title: Optional[str] = None
    date: Optional[datetime] = None
    link: Optional[str] = None
    excerpt_preview: Optional[str] = None


# ---------------------------------------------------------------------------
# Strategic Operating Model sections
# ---------------------------------------------------------------------------

class PowerInfluenceMap(BaseModel):
    """Power & Influence Map — formal/informal authority structure."""
    formal_authority: Optional[TaggedClaim] = None
    informal_influence: Optional[TaggedClaim] = None
    revenue_control: Optional[TaggedClaim] = None
    decision_gate_ownership: Optional[TaggedClaim] = None
    needs_to_impress: Optional[TaggedClaim] = None
    veto_risk: Optional[TaggedClaim] = None


class IncentiveStructure(BaseModel):
    """Incentive analysis — what drives this person's decisions."""
    short_term: list[TaggedClaim] = Field(default_factory=list)
    medium_term: list[TaggedClaim] = Field(default_factory=list)
    career: list[TaggedClaim] = Field(default_factory=list)
    risk_exposure: list[TaggedClaim] = Field(default_factory=list)
    personal_wins: list[TaggedClaim] = Field(default_factory=list)
    personal_losses: list[TaggedClaim] = Field(default_factory=list)


class CognitivePattern(BaseModel):
    """An observed cognitive or rhetorical pattern backed by evidence."""
    pattern_type: str = Field(
        ..., description="e.g. 'Repeated language', 'Framing device', 'Bias signal'"
    )
    observation: str
    evidence_quote: Optional[str] = None
    evidence_tag: EvidenceTag = EvidenceTag.unknown
    citations: list[Citation] = Field(default_factory=list)


class StrategicTension(BaseModel):
    """A live strategic tension identified from evidence."""
    tension: str
    evidence: str
    evidence_tag: EvidenceTag = EvidenceTag.unknown
    citations: list[Citation] = Field(default_factory=list)


class BehavioralForecast(BaseModel):
    """Scenario-based behavioral prediction."""
    scenario: str = Field(..., description="If X happens")
    predicted_reaction: str = Field(..., description="Likely reaction")
    reasoning: str = Field(..., description="Evidence-backed reasoning")
    citations: list[Citation] = Field(default_factory=list)


class InformationGap(BaseModel):
    """A strategically material information gap with resolution path."""
    gap: str
    strategic_impact: str = Field(..., description="Why this gap matters for strategy")
    how_to_resolve: str = Field("", description="Method to fill this gap")
    suggested_question: str = Field("", description="Exact question to ask on the call")


class ConversationStrategy(BaseModel):
    """Executive conversation strategy mapped to incentive structure."""
    leverage_angles: list[TaggedClaim] = Field(default_factory=list)
    stress_tests: list[TaggedClaim] = Field(default_factory=list)
    credibility_builders: list[TaggedClaim] = Field(default_factory=list)
    contrarian_wedge: Optional[TaggedClaim] = None
    collaboration_vector: Optional[TaggedClaim] = None


class MeetingDelta(BaseModel):
    """Comparison: public persona vs. meeting signals."""
    alignments: list[TaggedClaim] = Field(default_factory=list)
    divergences: list[TaggedClaim] = Field(default_factory=list)


class EngineImprovement(BaseModel):
    """Post-generation meta: what would improve future intelligence."""
    missing_signals: list[str] = Field(default_factory=list)
    recommended_data_sources: list[str] = Field(default_factory=list)
    capture_fields: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Person-first brief models
# ---------------------------------------------------------------------------

class WhatToCoverItem(BaseModel):
    """A specific evidence-backed agenda item for the upcoming call."""
    item: str
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)


class LeverageQuestion(BaseModel):
    """A leverage question with upstream evidence citation."""
    question: str
    rationale: str = ""
    citations: list[Citation] = Field(default_factory=list)


class ProofPoint(BaseModel):
    """A proof point to deploy, citing why it matters to them."""
    point: str
    why_it_matters: str = ""
    citations: list[Citation] = Field(default_factory=list)


class EvidenceIndexEntry(BaseModel):
    """Entry in the evidence index: every source with excerpt."""
    source_type: SourceType
    source_id: str
    timestamp: Optional[datetime] = None
    excerpt: str = Field("", description="Verbatim excerpt <=25 words")
    snippet_hash: str = ""
    link: Optional[str] = None


class VerifyFirstItem(BaseModel):
    """An identity fact to confirm when identity lock is weak."""
    fact: str
    current_confidence: str = "low"
    source: str = ""


# ---------------------------------------------------------------------------
# Public Visibility Report
# ---------------------------------------------------------------------------

class VisibilityEntry(BaseModel):
    """A single public visibility finding (talk, podcast, etc.)."""
    category: str = Field(..., description="ted|tedx|keynote|conference|summit|podcast|webinar|youtube_talk|panel|interview_video")
    title: str = ""
    url: str = ""
    date: str = ""
    snippet: str = ""
    tier: int = 3  # 1=primary, 2=secondary, 3=low


class PublicVisibilityReport(BaseModel):
    """Results of the 10-query public visibility sweep."""
    sweep_executed: bool = False
    categories_searched: list[str] = Field(default_factory=list)
    entries: list[VisibilityEntry] = Field(default_factory=list)
    total_results: int = 0
    ted_tedx_found: bool = False
    podcast_webinar_found: bool = False
    conference_keynote_found: bool = False


# ---------------------------------------------------------------------------
# Deal Probability Score
# ---------------------------------------------------------------------------

class DealProbabilityFactor(BaseModel):
    """A single weighted factor in the deal probability calculation."""
    factor: str
    weight_range: str = Field(..., description="e.g. '0-20', '-0-15'")
    score: float = 0.0
    reasoning: str = ""


class DealProbabilityScore(BaseModel):
    """Weighted 0-100 deal probability with factor breakdown."""
    total_score: float = Field(0.0, ge=0.0, le=100.0)
    factors: list[DealProbabilityFactor] = Field(default_factory=list)
    positive_total: float = 0.0
    negative_total: float = 0.0
    confidence_level: str = "low"  # low | medium | high


# ---------------------------------------------------------------------------
# Influence Strategy Recommendation
# ---------------------------------------------------------------------------

class InfluenceStrategy(BaseModel):
    """Structured influence strategy recommendation."""
    primary_leverage: Optional[str] = None
    secondary_leverage: Optional[str] = None
    message_framing: Optional[str] = None
    psychological_tempo: Optional[str] = None
    pressure_points: list[str] = Field(default_factory=list)
    avoidance_points: list[str] = Field(default_factory=list)
    early_warning_signs: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Canonical Brief Output
# ---------------------------------------------------------------------------

class BriefOutput(BaseModel):
    """The canonical Pre-Call Intelligence Brief.

    Person-first, evidence-locked. Every claim traces to a source record.
    Organised as sections A-I per the pre-call brief spec.

    Legacy strategic sections preserved for backward compatibility with
    the deep-profile endpoint.
    """
    header: HeaderSection

    # --- A) Header is above ---

    # --- B) Relationship & Interaction Snapshot (PERSON-FIRST) ---
    relationship_context: RelationshipContext = Field(default_factory=RelationshipContext)
    last_interaction: Optional[InteractionRecord] = None
    interaction_history: list[InteractionRecord] = Field(default_factory=list)

    # --- C) Open Loops & Commitments ---
    open_loops: list[OpenLoop] = Field(default_factory=list)

    # --- D) Watchouts ---
    watchouts: list[Watchout] = Field(default_factory=list)

    # --- E) What I Must Cover ---
    what_to_cover: list[WhatToCoverItem] = Field(default_factory=list)
    meeting_objectives: list[MeetingObjective] = Field(default_factory=list)

    # --- F) Leverage Plan ---
    leverage_plan: LeveragePlan = Field(default_factory=LeveragePlan)
    leverage_questions: list[LeverageQuestion] = Field(default_factory=list)
    proof_points: list[ProofPoint] = Field(default_factory=list)
    tension_to_surface_detail: Optional[TaggedClaim] = None
    direct_ask: Optional[TaggedClaim] = None

    # --- G) Agenda ---
    agenda: Agenda = Field(default_factory=Agenda)

    # --- H) Unknowns That Matter ---
    information_gaps: list[InformationGap] = Field(default_factory=list)

    # --- I) Evidence Index ---
    evidence_index: list[EvidenceIndexEntry] = Field(default_factory=list)
    appendix_evidence: list[EvidenceItem] = Field(default_factory=list)

    # --- Identity verification (when lock score < 70) ---
    verify_first: list[VerifyFirstItem] = Field(default_factory=list)

    # --- Strategic Operating Model (for deep-profile) ---
    strategic_positioning: list[TaggedClaim] = Field(default_factory=list)
    power_map: PowerInfluenceMap = Field(default_factory=PowerInfluenceMap)
    incentive_structure: IncentiveStructure = Field(default_factory=IncentiveStructure)
    cognitive_patterns: list[CognitivePattern] = Field(default_factory=list)
    strategic_tensions: list[StrategicTension] = Field(default_factory=list)
    behavioral_forecasts: list[BehavioralForecast] = Field(default_factory=list)
    conversation_strategy: ConversationStrategy = Field(default_factory=ConversationStrategy)
    meeting_delta: MeetingDelta = Field(default_factory=MeetingDelta)
    engine_improvements: EngineImprovement = Field(default_factory=EngineImprovement)

    # --- Deep-profile new sections ---
    public_visibility: PublicVisibilityReport = Field(
        default_factory=PublicVisibilityReport
    )
    deal_probability: DealProbabilityScore = Field(
        default_factory=DealProbabilityScore
    )
    influence_strategy: InfluenceStrategy = Field(
        default_factory=InfluenceStrategy
    )


# ---------------------------------------------------------------------------
# Internal data models (normalised artifacts)
# ---------------------------------------------------------------------------

class NormalizedTranscript(BaseModel):
    """Normalised representation of a Fireflies transcript."""
    source_id: str
    title: Optional[str] = None
    date: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    participants: list[str] = Field(default_factory=list)
    summary: Optional[str] = None
    action_items: list[str] = Field(default_factory=list)
    sentences: list[TranscriptSentence] = Field(default_factory=list)
    raw_json: Optional[dict] = None


class TranscriptSentence(BaseModel):
    speaker: Optional[str] = None
    text: str
    start_time: Optional[float] = None
    end_time: Optional[float] = None


NormalizedTranscript.model_rebuild()


class NormalizedEmail(BaseModel):
    """Normalised representation of a Gmail message."""
    source_id: str
    thread_id: Optional[str] = None
    subject: Optional[str] = None
    date: Optional[datetime] = None
    from_address: Optional[str] = None
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] = Field(default_factory=list)
    body_plain: Optional[str] = None
    snippet: Optional[str] = None
    labels: list[str] = Field(default_factory=list)
    raw_json: Optional[dict] = None
