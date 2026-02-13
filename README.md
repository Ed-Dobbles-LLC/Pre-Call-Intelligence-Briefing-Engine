# Pre-Call Intelligence Briefing Engine

A tool that produces **decision-grade Pre-Call Briefs** by ingesting Fireflies transcripts and Gmail emails, resolving entities, and generating cited intelligence reports. Available as both a **CLI** and a **REST API** (deployed on Railway).

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
Input (CLI or API)
  │
  ├─ Entity Resolution ─── Map person/company → emails, aliases, domains
  │
  ├─ Ingestion
  │   ├─ Fireflies API ──── Fetch transcripts → normalise → store
  │   └─ Gmail API ──────── Fetch emails → normalise → store
  │
  ├─ Embeddings ──────────── Chunk + embed new records (OpenAI)
  │
  ├─ Retrieval ──────────── Keyword + semantic search for evidence
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

## Web API

The engine is also available as a FastAPI service (deployed on Railway via Docker).

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check and config status (no auth) |
| `POST` | `/brief` | Generate a full brief (JSON response) |
| `POST` | `/brief/markdown` | Generate a brief, return markdown only |
| `POST` | `/brief/json` | Generate a brief, return structured JSON only |

### Authentication

Set `BRIEFING_API_KEY` to require a Bearer token on all `/brief` endpoints.
When the key is not set, endpoints are open (useful for local dev).

```bash
# Example request
curl -X POST https://your-app.railway.app/brief \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"person": "Jane Doe", "company": "Acme Corp"}'
```

### Request body

```json
{
  "person": "Jane Doe",
  "company": "Acme Corp",
  "topic": "Q1 Review",
  "meeting_when": "2026-02-15 14:00",
  "skip_ingestion": false
}
```

At least `person` or `company` is required.

## Configuration

All secrets are loaded from environment variables (or `.env` file):

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM + embeddings |
| `FIREFLIES_API_KEY` | For ingestion | Fireflies.ai API key |
| `GMAIL_CREDENTIALS_PATH` | For Gmail | Path to OAuth2 credentials JSON |
| `GMAIL_TOKEN_PATH` | For Gmail | Path to store OAuth token |
| `GOOGLE_CLIENT_ID` | For Gmail (env) | OAuth client ID (alternative to credentials file) |
| `GOOGLE_CLIENT_SECRET` | For Gmail (env) | OAuth client secret |
| `GOOGLE_REFRESH_TOKEN` | For Gmail (env) | OAuth refresh token |
| `BRIEFING_API_KEY` | Recommended | Bearer token for API auth (open if unset) |
| `DATABASE_URL` | No | Default: `sqlite:///./briefing_engine.db` |
| `OUTPUT_DIR` | No | Default: `./out` |

Missing keys are logged as warnings at startup so you can diagnose configuration issues quickly.

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
