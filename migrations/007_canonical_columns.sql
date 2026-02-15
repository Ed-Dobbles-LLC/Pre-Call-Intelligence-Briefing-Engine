-- 007: Add canonical columns for PDL enrichment persistence
--
-- These columns store the PDL-verified canonical values directly
-- on the entities table so Entity Lock can read them without
-- parsing the JSON blob. They survive restarts and are indexed.
--
-- The enrichment_json column stores the full PDL API response
-- for audit/debug purposes.

ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_company VARCHAR(512);
ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_title VARCHAR(512);
ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_location VARCHAR(512);
ALTER TABLE entities ADD COLUMN IF NOT EXISTS pdl_person_id VARCHAR(256);
ALTER TABLE entities ADD COLUMN IF NOT EXISTS pdl_match_confidence REAL;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMP;
ALTER TABLE entities ADD COLUMN IF NOT EXISTS enrichment_json TEXT;

-- Index for finding enriched contacts
CREATE INDEX IF NOT EXISTS ix_entities_pdl_person_id ON entities(pdl_person_id);
CREATE INDEX IF NOT EXISTS ix_entities_enriched_at ON entities(enriched_at);
