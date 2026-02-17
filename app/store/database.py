"""SQLAlchemy models and database initialisation.

Schema is designed for SQLite local dev with a clean migration path to
Supabase Postgres (swap the DATABASE_URL, run migrations).
"""

from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import Session, declarative_base, relationship, sessionmaker

from app.config import settings

Base = declarative_base()

# Module-level flag: True when pgvector extension is confirmed available.
_pgvector_available: bool = False


def pgvector_available() -> bool:
    """Return whether the pgvector extension was successfully enabled."""
    return _pgvector_available


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

class EntityRecord(Base):
    """A resolved person or company we track."""
    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(512), nullable=False, index=True)
    entity_type = Column(String(32), nullable=False, default="person")  # person | company
    emails = Column(Text, default="[]")  # JSON list
    aliases = Column(Text, default="[]")  # JSON list
    domains = Column(Text, default="[]")  # JSON list / profile_data blob
    # Canonical fields — written by PDL enrichment, read by Entity Lock
    canonical_company = Column(String(512), nullable=True)
    canonical_title = Column(String(512), nullable=True)
    canonical_location = Column(String(512), nullable=True)
    pdl_person_id = Column(String(256), nullable=True)
    pdl_match_confidence = Column(Float, nullable=True)
    enriched_at = Column(DateTime, nullable=True)
    enrichment_json = Column(Text, nullable=True)  # Full PDL response JSON
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def get_emails(self) -> list[str]:
        return json.loads(self.emails) if self.emails else []

    def set_emails(self, value: list[str]) -> None:
        self.emails = json.dumps(value)

    def get_aliases(self) -> list[str]:
        return json.loads(self.aliases) if self.aliases else []

    def set_aliases(self, value: list[str]) -> None:
        self.aliases = json.dumps(value)

    def get_domains(self) -> list[str]:
        return json.loads(self.domains) if self.domains else []

    def set_domains(self, value: list[str]) -> None:
        self.domains = json.dumps(value)


# ---------------------------------------------------------------------------
# Source records (Fireflies + Gmail)
# ---------------------------------------------------------------------------

class SourceRecord(Base):
    """Raw + normalised artifact from an external source."""
    __tablename__ = "source_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_type = Column(String(32), nullable=False)  # fireflies | gmail
    source_id = Column(String(512), nullable=False, unique=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=True)
    title = Column(String(1024), nullable=True)
    date = Column(DateTime, nullable=True)
    participants = Column(Text, default="[]")  # JSON list
    summary = Column(Text, nullable=True)
    action_items = Column(Text, default="[]")  # JSON list
    body = Column(Text, nullable=True)
    raw_json = Column(Text, nullable=True)
    normalized_json = Column(Text, nullable=True)
    link = Column(String(2048), nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)

    entity = relationship("EntityRecord", backref="source_records")

    __table_args__ = (
        Index("ix_source_type_date", "source_type", "date"),
        Index("ix_entity_date", "entity_id", "date"),
    )


# ---------------------------------------------------------------------------
# Embeddings (for semantic retrieval)
# ---------------------------------------------------------------------------

class EmbeddingRecord(Base):
    """Stores vector embeddings for source chunks."""
    __tablename__ = "embeddings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_record_id = Column(Integer, ForeignKey("source_records.id"), nullable=False)
    chunk_index = Column(Integer, default=0)
    chunk_text = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)  # JSON list[float]; for Supabase use pgvector
    model = Column(String(128), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    source_record = relationship("SourceRecord", backref="embeddings")


# ---------------------------------------------------------------------------
# Projects
# ---------------------------------------------------------------------------

class ProjectRecord(Base):
    """Tracks deals, job interviews, partnerships, and internal initiatives."""
    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(512), nullable=False)
    project_type = Column(String(64), nullable=False, default="other")
    stage = Column(String(64), nullable=False, default="identified")
    description = Column(Text, nullable=True)
    entity_ids = Column(Text, default="[]")      # JSON list of contact entity IDs
    source_ids = Column(Text, default="[]")      # JSON list of source_record IDs
    metadata_json = Column(Text, default="{}")   # Flexible metadata
    classifier_source = Column(String(32), default="rule")  # llm | rule | manual
    classifier_confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        Index("ix_projects_type", "project_type"),
        Index("ix_projects_stage", "stage"),
    )

    def get_entity_ids(self) -> list[int]:
        return json.loads(self.entity_ids) if self.entity_ids else []

    def set_entity_ids(self, ids: list[int]) -> None:
        self.entity_ids = json.dumps(ids)

    def add_entity_id(self, entity_id: int) -> None:
        ids = self.get_entity_ids()
        if entity_id not in ids:
            ids.append(entity_id)
            self.entity_ids = json.dumps(ids)

    def get_metadata(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}

    def set_metadata(self, data: dict) -> None:
        self.metadata_json = json.dumps(data)


# Stage pipelines by project type
PROJECT_STAGE_PIPELINES = {
    "job_interview": ["identified", "applied", "screening", "interviewing", "offer", "closed_won", "closed_lost"],
    "sales_deal": ["identified", "qualified", "proposal", "negotiation", "closed_won", "closed_lost"],
    "partnership": ["identified", "exploring", "terms", "active", "completed", "stalled"],
    "internal": ["identified", "planning", "in_progress", "review", "completed"],
    "other": ["identified", "in_progress", "completed"],
}


# ---------------------------------------------------------------------------
# Action Items
# ---------------------------------------------------------------------------

class ActionItemRecord(Base):
    """Action items extracted from transcripts and emails."""
    __tablename__ = "action_items"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(1024), nullable=False)
    description = Column(Text, nullable=True)
    source_type = Column(String(32), nullable=True)
    source_id = Column(String(512), nullable=True)
    source_record_id = Column(Integer, ForeignKey("source_records.id"), nullable=True)
    entity_id = Column(Integer, ForeignKey("entities.id"), nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    priority = Column(String(16), default="medium")
    status = Column(String(32), default="open")
    due_date = Column(DateTime, nullable=True)
    assigned_to = Column(String(512), nullable=True)
    extracted_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    metadata_json = Column(Text, default="{}")

    entity = relationship("EntityRecord", backref="action_items")
    project = relationship("ProjectRecord", backref="action_items")
    source_record = relationship("SourceRecord", backref="action_items_records")

    __table_args__ = (
        Index("ix_action_items_status", "status"),
        Index("ix_action_items_priority", "priority"),
        Index("ix_action_items_entity", "entity_id"),
        Index("ix_action_items_project", "project_id"),
    )

    def get_metadata(self) -> dict:
        return json.loads(self.metadata_json) if self.metadata_json else {}


# ---------------------------------------------------------------------------
# Calendar Events
# ---------------------------------------------------------------------------

class CalendarEventRecord(Base):
    """Persisted calendar events linked to contacts and projects."""
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    calendar_event_id = Column(String(512), nullable=False, unique=True)
    title = Column(String(1024), nullable=True)
    description = Column(Text, nullable=True)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    location = Column(String(1024), nullable=True)
    organizer_email = Column(String(512), nullable=True)
    attendees_json = Column(Text, default="[]")
    entity_ids = Column(Text, default="[]")
    project_id = Column(Integer, ForeignKey("projects.id"), nullable=True)
    status = Column(String(32), default="upcoming")
    synced_at = Column(DateTime, default=datetime.utcnow)
    metadata_json = Column(Text, default="{}")

    project = relationship("ProjectRecord", backref="calendar_events")

    __table_args__ = (
        Index("ix_calendar_events_start", "start_time"),
        Index("ix_calendar_events_status", "status"),
    )

    def get_entity_ids(self) -> list[int]:
        return json.loads(self.entity_ids) if self.entity_ids else []

    def set_entity_ids(self, ids: list[int]) -> None:
        self.entity_ids = json.dumps(ids)


# ---------------------------------------------------------------------------
# Brief audit log
# ---------------------------------------------------------------------------

class BriefLog(Base):
    """Immutable record of every brief produced."""
    __tablename__ = "brief_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    person = Column(String(512), nullable=True)
    company = Column(String(512), nullable=True)
    topic = Column(String(512), nullable=True)
    meeting_datetime = Column(DateTime, nullable=True)
    brief_json = Column(Text, nullable=False)
    brief_markdown = Column(Text, nullable=False)
    confidence_score = Column(Float, default=0.0)
    source_record_ids = Column(Text, default="[]")
    # Quality gate scores
    identity_lock_score = Column(Float, default=0.0)
    evidence_coverage_pct = Column(Float, default=0.0)
    genericness_score = Column(Float, default=0.0)
    gate_status = Column(String(32), default="not_run")
    created_at = Column(DateTime, default=datetime.utcnow)


# ---------------------------------------------------------------------------
# Engine / session helpers
# ---------------------------------------------------------------------------

_engine_cache: dict[str, "Engine"] = {}


def get_engine(url: str | None = None):
    url = url or settings.effective_database_url
    if url in _engine_cache:
        return _engine_cache[url]
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    engine = create_engine(url, echo=False, connect_args=connect_args)
    _engine_cache[url] = engine
    return engine


_session_factory_cache: dict[str, sessionmaker] = {}


def get_session_factory(url: str | None = None) -> sessionmaker:
    url = url or settings.effective_database_url
    if url in _session_factory_cache:
        return _session_factory_cache[url]
    engine = get_engine(url)
    factory = sessionmaker(bind=engine)
    _session_factory_cache[url] = factory
    return factory


def init_db(url: str | None = None) -> None:
    """Create all tables if they don't exist.

    On Postgres, probes for the pgvector extension but never attempts to
    CREATE it (which requires superuser on managed hosts like Railway).
    The embeddings table always uses a TEXT column with JSON-serialised
    vectors so the application works identically with or without pgvector.
    """
    global _pgvector_available
    import logging
    _log = logging.getLogger(__name__)

    engine = get_engine(url)
    effective_url = url or settings.effective_database_url
    if not effective_url.startswith("sqlite"):
        try:
            with engine.connect() as conn:
                row = conn.execute(
                    text(
                        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
                    )
                ).fetchone()
                if row:
                    _pgvector_available = True
                    _log.info("pgvector extension detected")
                else:
                    _log.info(
                        "pgvector extension not installed — embeddings will "
                        "use TEXT column with in-memory cosine similarity"
                    )
        except Exception as exc:
            _log.warning("Could not probe for pgvector extension: %s", exc)
    Base.metadata.create_all(engine)


def get_session(url: str | None = None) -> Session:
    factory = get_session_factory(url)
    return factory()
