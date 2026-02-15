-- Migration 004: Citation excerpt boundaries and gate score tracking
--
-- Adds columns to brief_logs for quality gate scores so they can be
-- queried without parsing the full brief JSON.

ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS identity_lock_score FLOAT DEFAULT 0.0;
ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS evidence_coverage_pct FLOAT DEFAULT 0.0;
ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS genericness_score FLOAT DEFAULT 0.0;
ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS gate_status VARCHAR(32) DEFAULT 'not_run';

-- Index for querying briefs by gate status
CREATE INDEX IF NOT EXISTS ix_brief_logs_gate_status ON brief_logs (gate_status);
