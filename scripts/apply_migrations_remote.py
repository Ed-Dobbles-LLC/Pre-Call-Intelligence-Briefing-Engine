"""One-shot script to apply migrations 004-007 to a remote Postgres database.

Usage:
    python scripts/apply_migrations_remote.py "postgresql://user:pass@host:port/db"

This is the same logic as run_migrations.py but takes the URL as a CLI argument
so you can point it at the Railway DATABASE_PUBLIC_URL from your local machine.
"""

from __future__ import annotations

import sys

from sqlalchemy import create_engine, text

if len(sys.argv) < 2:
    print("Usage: python scripts/apply_migrations_remote.py DATABASE_PUBLIC_URL")
    sys.exit(1)

url = sys.argv[1]
if url.startswith("postgres://"):
    url = url.replace("postgres://", "postgresql://", 1)

engine = create_engine(url, connect_args={"connect_timeout": 15})

# --- 1. Kill stale idle connections that may hold locks ----------------------
print("[1/5] Killing stale idle connections...")
with engine.begin() as conn:
    rows = conn.execute(text("""
        SELECT pg_terminate_backend(pid)
        FROM pg_stat_activity
        WHERE state = 'idle'
          AND pid != pg_backend_pid()
    """)).fetchall()
    print(f"      Terminated {len(rows)} idle connections")

# --- 2. Migration 004 -------------------------------------------------------
print("[2/5] Applying migration 004 (brief_logs gate columns)...")
stmts_004 = [
    "ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS identity_lock_score FLOAT DEFAULT 0.0",
    "ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS evidence_coverage_pct FLOAT DEFAULT 0.0",
    "ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS genericness_score FLOAT DEFAULT 0.0",
    "ALTER TABLE brief_logs ADD COLUMN IF NOT EXISTS gate_status VARCHAR(32) DEFAULT 'not_run'",
    "CREATE INDEX IF NOT EXISTS ix_brief_logs_gate_status ON brief_logs (gate_status)",
]
for stmt in stmts_004:
    with engine.begin() as conn:
        conn.execute(text(stmt))
print("      Done")

# --- 3. Migration 005 & 006 (documentation only) ----------------------------
print("[3/5] Migrations 005 & 006 are documentation-only — skipping")

# --- 4. Migration 007 -------------------------------------------------------
print("[4/5] Applying migration 007 (canonical + PDL columns on entities)...")
stmts_007 = [
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_company VARCHAR(512)",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_title VARCHAR(512)",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS canonical_location VARCHAR(512)",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS pdl_person_id VARCHAR(256)",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS pdl_match_confidence REAL",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS enriched_at TIMESTAMP",
    "ALTER TABLE entities ADD COLUMN IF NOT EXISTS enrichment_json TEXT",
    "CREATE INDEX IF NOT EXISTS ix_entities_pdl_person_id ON entities(pdl_person_id)",
    "CREATE INDEX IF NOT EXISTS ix_entities_enriched_at ON entities(enriched_at)",
]
for stmt in stmts_007:
    with engine.begin() as conn:
        conn.execute(text(stmt))
print("      Done")

# --- 5. Verify --------------------------------------------------------------
print("[5/5] Verifying schema...")
with engine.connect() as conn:
    rows = conn.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = 'entities'
          AND column_name IN (
            'canonical_company', 'canonical_title', 'canonical_location',
            'pdl_person_id', 'pdl_match_confidence'
          )
        ORDER BY column_name
    """)).fetchall()
    columns = [r[0] for r in rows]
    print(f"      Found columns: {columns}")
    if len(columns) == 5:
        print("      ALL 5 COLUMNS PRESENT — migration successful!")
    else:
        print(f"      ERROR: expected 5 columns, found {len(columns)}")
        sys.exit(1)
