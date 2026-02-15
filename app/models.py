"""Pydantic models for the Brief output schema and internal data structures.

These models define the canonical JSON shape of a Strategic Intelligence Brief.
Every claim-bearing section carries a list of Citation objects so the
consumer can trace every bullet back to a stored source record.

The brief is structured as a Strategic Operating Model, not a generic
profile. Every claim is tagged with evidence discipline markers.
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
    """A strategically material information gap."""
    gap: str
    strategic_impact: str = Field(..., description="Why this gap matters for strategy")


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
# Canonical Brief Output
# ---------------------------------------------------------------------------

class BriefOutput(BaseModel):
    """The canonical Strategic Intelligence Brief.

    Contains both legacy sections (for backward compatibility) and the
    new Strategic Operating Model sections. When strategic sections are
    populated, they take precedence in rendering.
    """
    header: HeaderSection

    # --- Legacy sections (backward compat) ---
    relationship_context: RelationshipContext = Field(default_factory=RelationshipContext)
    last_interaction: Optional[InteractionRecord] = None
    interaction_history: list[InteractionRecord] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    watchouts: list[Watchout] = Field(default_factory=list)
    meeting_objectives: list[MeetingObjective] = Field(default_factory=list)
    leverage_plan: LeveragePlan = Field(default_factory=LeveragePlan)
    agenda: Agenda = Field(default_factory=Agenda)
    appendix_evidence: list[EvidenceItem] = Field(default_factory=list)

    # --- Strategic Operating Model ---
    strategic_positioning: list[TaggedClaim] = Field(default_factory=list)
    power_map: PowerInfluenceMap = Field(default_factory=PowerInfluenceMap)
    incentive_structure: IncentiveStructure = Field(default_factory=IncentiveStructure)
    cognitive_patterns: list[CognitivePattern] = Field(default_factory=list)
    strategic_tensions: list[StrategicTension] = Field(default_factory=list)
    behavioral_forecasts: list[BehavioralForecast] = Field(default_factory=list)
    information_gaps: list[InformationGap] = Field(default_factory=list)
    conversation_strategy: ConversationStrategy = Field(default_factory=ConversationStrategy)
    meeting_delta: MeetingDelta = Field(default_factory=MeetingDelta)
    engine_improvements: EngineImprovement = Field(default_factory=EngineImprovement)


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
