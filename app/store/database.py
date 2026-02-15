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

def get_engine(url: str | None = None):
    url = url or settings.effective_database_url
    connect_args = {}
    if url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(url, echo=False, connect_args=connect_args)


def get_session_factory(url: str | None = None) -> sessionmaker:
    engine = get_engine(url)
    return sessionmaker(bind=engine)


def _apply_column_migrations(engine, effective_url: str) -> None:
    """Add columns that may be missing on existing Postgres databases.

    ``create_all()`` only creates *new* tables; it never ALTERs existing
    ones.  This function closes the gap by issuing idempotent
    ``ALTER TABLE … ADD COLUMN IF NOT EXISTS`` statements for every
    column introduced in migrations 004-007.

    On SQLite the clauses are wrapped in try/except because SQLite does
    not support ``IF NOT EXISTS`` on ``ADD COLUMN``.
    """
    import logging

    _log = logging.getLogger(__name__)

    is_pg = not effective_url.startswith("sqlite")

    alter_stmts = [
        # Migration 004 — gate scores on brief_logs
        "ALTER TABLE brief_logs ADD COLUMN {ine} identity_lock_score FLOAT DEFAULT 0.0",
        "ALTER TABLE brief_logs ADD COLUMN {ine} evidence_coverage_pct FLOAT DEFAULT 0.0",
        "ALTER TABLE brief_logs ADD COLUMN {ine} genericness_score FLOAT DEFAULT 0.0",
        "ALTER TABLE brief_logs ADD COLUMN {ine} gate_status VARCHAR(32) DEFAULT 'not_run'",
        # Migration 007 — canonical / PDL columns on entities
        "ALTER TABLE entities ADD COLUMN {ine} canonical_company VARCHAR(512)",
        "ALTER TABLE entities ADD COLUMN {ine} canonical_title VARCHAR(512)",
        "ALTER TABLE entities ADD COLUMN {ine} canonical_location VARCHAR(512)",
        "ALTER TABLE entities ADD COLUMN {ine} pdl_person_id VARCHAR(256)",
        "ALTER TABLE entities ADD COLUMN {ine} pdl_match_confidence REAL",
        "ALTER TABLE entities ADD COLUMN {ine} enriched_at TIMESTAMP",
        "ALTER TABLE entities ADD COLUMN {ine} enrichment_json TEXT",
    ]

    index_stmts = [
        "CREATE INDEX IF NOT EXISTS ix_brief_logs_gate_status ON brief_logs (gate_status)",
        "CREATE INDEX IF NOT EXISTS ix_entities_pdl_person_id ON entities (pdl_person_id)",
        "CREATE INDEX IF NOT EXISTS ix_entities_enriched_at ON entities (enriched_at)",
    ]

    ine = "IF NOT EXISTS" if is_pg else ""

    try:
        with engine.begin() as conn:
            for tmpl in alter_stmts:
                stmt = tmpl.format(ine=ine)
                try:
                    conn.execute(text(stmt))
                except Exception:
                    # SQLite: column already exists raises OperationalError
                    pass
            for stmt in index_stmts:
                try:
                    conn.execute(text(stmt))
                except Exception:
                    pass
        _log.info("Column migrations verified/applied")
    except Exception as exc:
        _log.warning("Column migration check failed: %s", exc)


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
    _apply_column_migrations(engine, effective_url)


def get_session(url: str | None = None) -> Session:
    factory = get_session_factory(url)
    return factory()
