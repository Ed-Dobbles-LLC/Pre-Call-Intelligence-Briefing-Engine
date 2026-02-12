-- Pre-Call Intelligence Briefing Engine
-- Migration 001: Initial Schema
-- Compatible with both SQLite and PostgreSQL/Supabase

-- Entities: resolved persons and companies
CREATE TABLE IF NOT EXISTS entities (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,  -- For Postgres: SERIAL PRIMARY KEY
    name            VARCHAR(512)    NOT NULL,
    entity_type     VARCHAR(32)     NOT NULL DEFAULT 'person',  -- person | company
    emails          TEXT            DEFAULT '[]',       -- JSON array of email addresses
    aliases         TEXT            DEFAULT '[]',       -- JSON array of name aliases
    domains         TEXT            DEFAULT '[]',       -- JSON array of company domains
    created_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_entities_name ON entities(name);
CREATE INDEX IF NOT EXISTS ix_entities_type ON entities(entity_type);

-- Source records: raw + normalised artifacts from Fireflies / Gmail
CREATE TABLE IF NOT EXISTS source_records (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type     VARCHAR(32)     NOT NULL,           -- fireflies | gmail
    source_id       VARCHAR(512)    NOT NULL UNIQUE,    -- external system ID
    entity_id       INTEGER         REFERENCES entities(id),
    title           VARCHAR(1024),
    date            TIMESTAMP,
    participants    TEXT            DEFAULT '[]',       -- JSON array
    summary         TEXT,
    action_items    TEXT            DEFAULT '[]',       -- JSON array
    body            TEXT,                               -- full text body
    raw_json        TEXT,                               -- original API response
    normalized_json TEXT,                               -- our normalised Pydantic model
    link            VARCHAR(2048),
    ingested_at     TIMESTAMP       DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_source_type_date ON source_records(source_type, date);
CREATE INDEX IF NOT EXISTS ix_entity_date ON source_records(entity_id, date);
CREATE INDEX IF NOT EXISTS ix_source_id ON source_records(source_id);

-- Embeddings: vector store for semantic retrieval
-- For Supabase, replace TEXT embedding column with: embedding vector(1536)
CREATE TABLE IF NOT EXISTS embeddings (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_record_id    INTEGER NOT NULL REFERENCES source_records(id),
    chunk_index         INTEGER DEFAULT 0,
    chunk_text          TEXT    NOT NULL,
    embedding           TEXT    NOT NULL,   -- JSON array of floats; Supabase: vector(1536)
    model               VARCHAR(128),
    created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_emb_source ON embeddings(source_record_id);

-- Brief audit log: immutable record of every brief produced
CREATE TABLE IF NOT EXISTS brief_logs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
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

-- ==========================================================================
-- SUPABASE MIGRATION NOTES
-- ==========================================================================
-- To migrate from SQLite to Supabase Postgres:
--
-- 1. Replace AUTOINCREMENT with SERIAL
-- 2. Replace the embeddings.embedding TEXT column with:
--       embedding vector(1536)
--    (requires the pgvector extension: CREATE EXTENSION IF NOT EXISTS vector;)
-- 3. Add a GiST or IVFFlat index on the embedding column:
--       CREATE INDEX ix_emb_vector ON embeddings USING ivfflat (embedding vector_cosine_ops);
-- 4. Update DATABASE_URL in .env to your Supabase connection string
-- 5. The application code auto-detects Postgres vs SQLite via the URL prefix
