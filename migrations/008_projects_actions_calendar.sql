-- Migration 008: Projects, Action Items, and Calendar Events tables
-- Supports both SQLite and PostgreSQL

-- Projects table: tracks deals, job interviews, partnerships, internal initiatives
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(512) NOT NULL,
    project_type VARCHAR(64) NOT NULL DEFAULT 'other',  -- job_interview | sales_deal | partnership | internal | other
    stage VARCHAR(64) NOT NULL DEFAULT 'identified',     -- stage pipeline varies by type
    description TEXT,
    entity_ids TEXT DEFAULT '[]',       -- JSON list of associated contact entity IDs
    source_ids TEXT DEFAULT '[]',       -- JSON list of source_record IDs that contributed
    metadata_json TEXT DEFAULT '{}',    -- Flexible metadata (company, role, deal_size, etc.)
    classifier_source VARCHAR(32) DEFAULT 'rule',  -- 'llm' | 'rule' | 'manual'
    classifier_confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS ix_projects_type ON projects (project_type);
CREATE INDEX IF NOT EXISTS ix_projects_stage ON projects (stage);

-- Action Items table: extracted from transcripts and emails
CREATE TABLE IF NOT EXISTS action_items (
    id SERIAL PRIMARY KEY,
    title VARCHAR(1024) NOT NULL,
    description TEXT,
    source_type VARCHAR(32),          -- 'fireflies' | 'gmail' | 'manual'
    source_id VARCHAR(512),           -- source_record source_id
    source_record_id INTEGER REFERENCES source_records(id),
    entity_id INTEGER REFERENCES entities(id),
    project_id INTEGER REFERENCES projects(id),
    priority VARCHAR(16) DEFAULT 'medium',  -- 'critical' | 'high' | 'medium' | 'low'
    status VARCHAR(32) DEFAULT 'open',       -- 'open' | 'in_progress' | 'done' | 'dismissed'
    due_date TIMESTAMP,
    assigned_to VARCHAR(512),
    extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    metadata_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_action_items_status ON action_items (status);
CREATE INDEX IF NOT EXISTS ix_action_items_priority ON action_items (priority);
CREATE INDEX IF NOT EXISTS ix_action_items_entity ON action_items (entity_id);
CREATE INDEX IF NOT EXISTS ix_action_items_project ON action_items (project_id);

-- Calendar Events table: persisted calendar events linked to contacts and projects
CREATE TABLE IF NOT EXISTS calendar_events (
    id SERIAL PRIMARY KEY,
    calendar_event_id VARCHAR(512) NOT NULL UNIQUE,
    title VARCHAR(1024),
    description TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    location VARCHAR(1024),
    organizer_email VARCHAR(512),
    attendees_json TEXT DEFAULT '[]',    -- JSON list of {email, name, response_status}
    entity_ids TEXT DEFAULT '[]',        -- JSON list of matched contact entity IDs
    project_id INTEGER REFERENCES projects(id),
    status VARCHAR(32) DEFAULT 'upcoming',  -- 'upcoming' | 'completed' | 'cancelled'
    synced_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_json TEXT DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS ix_calendar_events_start ON calendar_events (start_time);
CREATE INDEX IF NOT EXISTS ix_calendar_events_status ON calendar_events (status);
