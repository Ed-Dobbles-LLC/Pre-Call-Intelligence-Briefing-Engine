-- Pre-Call Intelligence Briefing Engine
-- Migration 003: Optional pgvector upgrade
--
-- PREREQUISITES:
--   1. pgvector extension must be installed on your Postgres instance
--   2. You must have privileges to CREATE EXTENSION (superuser or rds_superuser)
--   3. Migration 002 must have been applied first
--
-- This migration converts the embeddings.embedding column from TEXT
-- (JSON-serialised floats) to native vector(1536) and adds an IVFFlat
-- index for fast cosine-similarity search.
--
-- WARNING: If the embeddings table already contains data, the ALTER COLUMN
-- cast will fail because TEXT→vector has no implicit cast.  You must
-- re-populate embeddings after running this migration (the app will
-- re-embed automatically on the next sync).

-- 1. Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Drop existing data (TEXT values cannot be cast to vector)
TRUNCATE TABLE embeddings;

-- 3. Convert column type to native vector
ALTER TABLE embeddings ALTER COLUMN embedding TYPE vector(1536);

-- 4. Add IVFFlat index for cosine-similarity search
--    (lists = 100 is a reasonable default; tune for your dataset size)
CREATE INDEX IF NOT EXISTS ix_emb_vector
    ON embeddings USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
