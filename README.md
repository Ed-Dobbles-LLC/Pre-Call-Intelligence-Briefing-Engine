# Pre-Call Intelligence Briefing Engine

A command-line tool that produces **decision-grade Pre-Call Briefs** by ingesting Fireflies transcripts and Gmail emails, resolving entities, and generating cited intelligence reports.

## Quick Start

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. Configure environment
cp .env.example .env
# Edit .env with your API keys

# 3. Generate a brief
brief --person "Jane Doe" --company "Acme Corp" --when "2026-02-15 14:00" --topic "Q1 Review"
```

Output files are written to `./out/`:
- `brief_Jane_Doe_20260215.md` – human-readable markdown
- `brief_Jane_Doe_20260215.json` – structured JSON with citations

## Architecture

```
Input (CLI)
  │
  ├─ Entity Resolution ─── Map person/company → emails, aliases, domains
  │
  ├─ Ingestion
  │   ├─ Fireflies API ──── Fetch transcripts → normalise → store
  │   └─ Gmail API ──────── Fetch emails → normalise → store
  │
  ├─ Retrieval ──────────── Query stored artifacts for relevant evidence
  │
  ├─ Brief Generation ───── LLM synthesises evidence into cited brief
  │
  └─ Output ─────────────── JSON + Markdown files, audit log
```

### Pipeline Failure Modes

| Step | Failure | Handling |
|------|---------|----------|
| Fireflies API | Rate limit / auth error | Log warning, continue with stored data |
| Gmail API | OAuth expired / no credentials | Log warning, continue with stored data |
| Entity resolution | No match found | Create new entity, proceed |
| LLM call | API error / timeout | Fall back to raw-evidence brief |
| Embedding | API error | Skip embeddings, brief still works |

## Project Structure

```
app/
├── clients/           # API clients (Fireflies, Gmail, OpenAI)
├── ingest/            # Ingestion + normalisation
├── normalize/         # Entity resolution + embeddings
├── store/             # SQLAlchemy models + DB init
├── retrieve/          # Evidence retrieval
├── brief/             # Brief generation + rendering + pipeline
├── cli/               # Click CLI entry point
├── config.py          # Pydantic settings
└── models.py          # Canonical Pydantic schemas
tests/
migrations/
```

## CLI Options

```
brief --person "Name"        # Person to brief on
      --company "Company"    # Company to brief on
      --when "YYYY-MM-DD HH:MM"  # Meeting datetime
      --topic "Topic"        # Meeting topic
      --skip-ingestion       # Use only stored data (no API calls)
      --verbose              # Debug logging
```

At least `--person` or `--company` is required.

## Configuration

All secrets are loaded from environment variables (or `.env` file):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM + embeddings |
| `FIREFLIES_API_KEY` | For ingestion | Fireflies.ai API key |
| `GMAIL_CREDENTIALS_PATH` | For Gmail | Path to OAuth2 credentials JSON |
| `GMAIL_TOKEN_PATH` | For Gmail | Path to store OAuth token |
| `DATABASE_URL` | No | Default: `sqlite:///./briefing_engine.db` |
| `OUTPUT_DIR` | No | Default: `./out` |

## Running Tests

```bash
pip install -e ".[dev]"
pytest -v
```

## Migration to Supabase

1. Set `DATABASE_URL` to your Supabase Postgres connection string
2. Enable pgvector: `CREATE EXTENSION IF NOT EXISTS vector;`
3. Run `migrations/001_initial_schema.sql` (see Supabase notes at bottom)
4. The application auto-detects Postgres vs SQLite
