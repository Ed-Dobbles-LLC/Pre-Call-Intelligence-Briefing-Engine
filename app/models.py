"""Pydantic models for the Brief output schema and internal data structures.

These models define the canonical JSON shape of a Pre-Call Brief.
Every claim-bearing section carries a list of Citation objects so the
consumer can trace every bullet back to a stored source record.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Citation
# ---------------------------------------------------------------------------

class SourceType(str, Enum):
    fireflies = "fireflies"
    gmail = "gmail"


class Citation(BaseModel):
    """Pointer back to the evidence behind a claim."""
    source_type: SourceType
    source_id: str = Field(..., description="Fireflies transcript ID or Gmail message ID")
    timestamp: datetime
    excerpt: str = Field(..., description="Verbatim excerpt supporting the claim")
    snippet_hash: str = Field(..., description="SHA-256 of the excerpt for dedup / audit")
    link: Optional[str] = Field(None, description="Deep-link if available")


# ---------------------------------------------------------------------------
# Brief sections
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


class BriefOutput(BaseModel):
    """The canonical Pre-Call Intelligence Brief."""
    header: HeaderSection
    relationship_context: RelationshipContext = Field(default_factory=RelationshipContext)
    last_interaction: Optional[InteractionRecord] = None
    interaction_history: list[InteractionRecord] = Field(default_factory=list)
    open_loops: list[OpenLoop] = Field(default_factory=list)
    watchouts: list[Watchout] = Field(default_factory=list)
    meeting_objectives: list[MeetingObjective] = Field(default_factory=list)
    leverage_plan: LeveragePlan = Field(default_factory=LeveragePlan)
    agenda: Agenda = Field(default_factory=Agenda)
    appendix_evidence: list[EvidenceItem] = Field(default_factory=list)


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
