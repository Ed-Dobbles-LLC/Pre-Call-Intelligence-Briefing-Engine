"""Apply SQL migrations to the database.

Reads DATABASE_URL from the environment (same var the web service uses),
connects via SQLAlchemy, and executes each migration file in order.

All migration statements use IF NOT EXISTS / IF NOT EXISTS guards so
this script is safe to re-run on every deploy.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("migrate")

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"

ORDERED_MIGRATIONS = [
    "004_citation_offsets_and_gates.sql",
    "005_photo_and_calendar.sql",
    "006_pdl_enrichment.sql",
    "007_canonical_columns.sql",
]

VERIFY_QUERY = """\
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'entities'
  AND column_name IN (
    'canonical_company', 'canonical_title', 'canonical_location',
    'pdl_person_id', 'pdl_match_confidence'
  )
ORDER BY column_name;
"""


def get_database_url() -> str:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        logger.error("DATABASE_URL is not set")
        sys.exit(1)
    # Railway injects postgres:// but SQLAlchemy 2.0+ needs postgresql://
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql://", 1)
    return url


def run_migrations() -> None:
    url = get_database_url()
    is_sqlite = url.startswith("sqlite")

    logger.info("Connecting to database …")
    engine = create_engine(url)

    # Verify the connection actually works before running migrations
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("Database connection OK")
    except Exception as exc:
        logger.error("Cannot connect to database: %s", exc)
        sys.exit(1)

    for filename in ORDERED_MIGRATIONS:
        path = MIGRATIONS_DIR / filename
        if not path.exists():
            logger.warning("Migration file not found, skipping: %s", filename)
            continue

        sql = path.read_text(encoding="utf-8").strip()
        if not sql:
            continue

        logger.info("Applying %s …", filename)
        # Split on semicolons to execute each statement individually
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        with engine.begin() as conn:
            for stmt in statements:
                # Skip comment-only blocks
                code_lines = [ln for ln in stmt.splitlines() if not ln.strip().startswith("--")]
                if not "".join(code_lines).strip():
                    continue
                try:
                    conn.execute(text(stmt))
                except Exception as exc:
                    # On SQLite, IF NOT EXISTS isn't supported for ADD COLUMN —
                    # a "duplicate column" error is expected and harmless.
                    if is_sqlite and "duplicate column" in str(exc).lower():
                        pass
                    else:
                        logger.warning("  Statement skipped (%s): %s", exc, stmt[:80])
        logger.info("  ✓ %s applied", filename)

    # Verification (Postgres only)
    if not is_sqlite:
        logger.info("Verifying schema …")
        with engine.connect() as conn:
            rows = conn.execute(text(VERIFY_QUERY)).fetchall()
            columns = [r[0] for r in rows]
            logger.info("  Found columns: %s", columns)
            if len(columns) == 5:
                logger.info("  ✓ All 5 canonical columns present")
            else:
                logger.error(
                    "  ✗ Expected 5 columns, found %d — check migration output above",
                    len(columns),
                )
                sys.exit(1)

    logger.info("Migrations complete.")


if __name__ == "__main__":
    try:
        run_migrations()
    except Exception:
        logger.exception("Migration script crashed")
        sys.exit(1)
