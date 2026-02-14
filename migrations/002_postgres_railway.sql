-- Pre-Call Intelligence Briefing Engine
-- Migration 002: PostgreSQL schema for Railway Postgres
-- Run this against your Railway Postgres database
--
-- NOTE: This migration works with OR without pgvector.
-- The embeddings.embedding column is TEXT (JSON-serialised float arrays).
-- If you have pgvector installed and want native vector indexing, run
-- migration 003_enable_pgvector.sql after this one.

-- Entities: resolved persons and companies
CREATE TABLE IF NOT EXISTS entities (
    id              SERIAL PRIMARY KEY,
    name            VARCHAR(512)    NOT NULL,
    entity_type     VARCHAR(32)     NOT NULL DEFAULT 'person',
    emails          TEXT            DEFAULT '[]',
    aliases         TEXT            DEFAULT '[]',
    domains         TEXT            DEFAULT '[]',
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS ix_entities_type ON entities(entity_type);

-- Source records: raw + normalised artifacts from Fireflies / Gmail
CREATE TABLE IF NOT EXISTS source_records (
    id              SERIAL PRIMARY KEY,
    source_type     VARCHAR(32)     NOT NULL,
    source_id       VARCHAR(512)    NOT NULL UNIQUE,
    entity_id       INTEGER         REFERENCES entities(id),
    title           VARCHAR(1024),
    date            TIMESTAMP,
    participants    TEXT            DEFAULT '[]',
    summary         TEXT,
    action_items    TEXT            DEFAULT '[]',
    body            TEXT,
    raw_json        TEXT,
    normalized_json TEXT,
    link            VARCHAR(2048),
    ingested_at     TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_source_type_date ON source_records(source_type, date);
CREATE INDEX IF NOT EXISTS ix_entity_date ON source_records(entity_id, date);
CREATE INDEX IF NOT EXISTS ix_source_id ON source_records(source_id);

-- Embeddings: stored as JSON text (compatible with all Postgres hosts)
-- The application reads/writes JSON-serialised float arrays to this column
-- and performs cosine similarity in Python.  See 003_enable_pgvector.sql
-- for an optional upgrade to native vector(1536) + IVFFlat indexing.
CREATE TABLE IF NOT EXISTS embeddings (
    id                  SERIAL PRIMARY KEY,
    source_record_id    INTEGER NOT NULL REFERENCES source_records(id),
    chunk_index         INTEGER DEFAULT 0,
    chunk_text          TEXT    NOT NULL,
    embedding           TEXT    NOT NULL,
    model               VARCHAR(128),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_emb_source ON embeddings(source_record_id);

-- Brief audit log
CREATE TABLE IF NOT EXISTS brief_logs (
    id                  SERIAL PRIMARY KEY,
    person              VARCHAR(512),
    company             VARCHAR(512),
    topic               VARCHAR(512),
    meeting_datetime    TIMESTAMP,
    brief_json          TEXT    NOT NULL,
    brief_markdown      TEXT    NOT NULL,
    confidence_score    REAL    DEFAULT 0.0,
    source_record_ids   TEXT    DEFAULT '[]',
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
