# CLAUDE.md — Pre-Call Intelligence Briefing Engine

## Project Overview

A decision-grade meeting briefing tool that generates cited intelligence reports by ingesting Fireflies transcripts and Gmail emails, resolving entities, and synthesizing evidence with an LLM. Every claim in a generated brief traces back to a stored source document (transcript or email) via `Citation` objects.

Dual interface: **CLI** (`brief` command via Click) and **REST API** (FastAPI on Uvicorn, deployed on Railway via Docker).

## Tech Stack

- **Python 3.11+** — required minimum version
- **FastAPI + Uvicorn** — async web server (`app/api.py`)
- **Click** — CLI framework (`app/cli/main.py`)
- **Pydantic 2.x + pydantic-settings** — data models and config
- **SQLAlchemy 2.0** — ORM (sync sessions, not async)
- **OpenAI API** — GPT-4o for brief generation, text-embedding-3-small for embeddings
- **httpx** — HTTP client (Fireflies GraphQL, Apollo.io REST)
- **google-api-python-client** — Gmail OAuth2 integration
- **Ruff** — linter/formatter (line-length 100, target py311)
- **pytest + pytest-asyncio** — test framework

## Repository Structure

```
app/
  api.py              # FastAPI app — all HTTP endpoints
  config.py           # Pydantic Settings (env vars / .env)
  models.py           # Canonical Pydantic schemas (BriefOutput, Citation, etc.)
  brief/
    pipeline.py       # Main orchestrator: ingest → resolve → retrieve → generate
    generator.py      # LLM brief generation with citation extraction
    renderer.py       # Markdown rendering of BriefOutput
  clients/
    apollo.py         # Apollo.io People Enrichment API client
    fireflies.py      # Fireflies.ai GraphQL API client
    gmail.py          # Gmail API OAuth2 client
    openai_client.py  # OpenAI chat + embeddings wrapper
  ingest/
    fireflies_ingest.py  # Fetch + normalize + store Fireflies transcripts
    gmail_ingest.py      # Fetch + normalize + store Gmail messages
  normalize/
    entity_resolver.py   # Person/company entity resolution (email, name, alias)
    embeddings.py        # Chunk text + generate OpenAI embeddings
  retrieve/
    retriever.py         # Keyword + semantic search for evidence assembly
  store/
    database.py          # SQLAlchemy models, engine, session, init_db()
  sync/
    auto_sync.py         # Background thread: periodic Fireflies/Gmail sync
  cli/
    main.py              # Click CLI entry point
  static/
    index.html           # Web dashboard (single-page)
tests/
  conftest.py            # Fixtures: test DB, sample payloads
  test_api.py            # FastAPI endpoint tests
  test_citations.py      # Citation extraction + formatting
  test_entity_matching.py # Entity resolution logic
  test_ingestion.py      # Fireflies/Gmail normalization
  test_retrieval.py      # Evidence retrieval + search
migrations/
  001_initial_schema.sql # Base schema (SQLite + Postgres)
  002_postgres_railway.sql # Railway/Postgres-specific indexes
scripts/
  setup_gmail_oauth.py   # Interactive Gmail OAuth credential setup
```

## Key Entry Points

| What | Location |
|------|----------|
| FastAPI app | `app/api.py:app` |
| CLI entry point | `app/cli/main.py:cli` (installed as `brief` command) |
| Pipeline orchestrator | `app/brief/pipeline.py:run_pipeline()` |
| Database init | `app/store/database.py:init_db()` |
| Config singleton | `app/config.py:settings` |

## Build & Run Commands

```bash
# Install (with dev deps)
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run API server locally
uvicorn app.api:app --host 0.0.0.0 --port 8000

# Run CLI
brief --person "Jane Doe" --company "Acme Corp" --when "2026-02-15 14:00" --topic "Q1 Review"

# Lint / format
ruff check .
ruff format .
```

## Testing

- **Framework**: pytest 8.x with pytest-asyncio (asyncio_mode = "auto")
- **Test database**: Each test gets a fresh SQLite DB via the `setup_test_db` autouse fixture in `tests/conftest.py`
- **Environment**: Tests set empty API keys (`OPENAI_API_KEY=""`, etc.) and disable auth (`BRIEFING_API_KEY=""`)
- **No external calls**: Tests validate normalization, entity matching, retrieval, and API routing without hitting live APIs
- **Run**: `pytest tests/ -v` — all tests must pass before merging

## Configuration

All config is via environment variables, loaded through Pydantic Settings (`app/config.py:Settings`). Copy `.env.example` to `.env` for local development.

**Required for brief generation:**
- `OPENAI_API_KEY` — LLM + embeddings

**Required for data ingestion:**
- `FIREFLIES_API_KEY` — Fireflies transcript access
- Gmail: either `GMAIL_CREDENTIALS_PATH` + `GMAIL_TOKEN_PATH` (file-based OAuth) or `GOOGLE_CLIENT_ID` + `GOOGLE_CLIENT_SECRET` + `GOOGLE_REFRESH_TOKEN` (env-based OAuth)

**Optional:**
- `APOLLO_API_KEY` — contact enrichment (job titles, photos, LinkedIn)
- `BRIEFING_API_KEY` — Bearer token for API auth (unauthenticated if unset)
- `DATABASE_URL` — defaults to `sqlite:///./briefing_engine.db`; use `postgresql://` for production
- `RETRIEVAL_WINDOW_DAYS` — default 90
- `OUTPUT_DIR` — default `./out`
- `LOG_LEVEL` — default `INFO`

## Database

- **Local**: SQLite (default, zero-config)
- **Production**: PostgreSQL on Supabase (with pgvector for embeddings)
- SQLAlchemy models in `app/store/database.py`: `EntityRecord`, `SourceRecord`, `EmbeddingRecord`, `BriefLog`
- Tables are auto-created by `init_db()` at startup
- SQL migrations in `migrations/` for Postgres-specific features

## Pipeline Architecture

The core pipeline in `app/brief/pipeline.py:run_pipeline()` follows this flow:

1. **Entity Resolution** — map person/company names to stored entities (`entity_resolver.py`)
2. **Ingestion** — fetch from Fireflies/Gmail APIs, normalize, store (`ingest/`)
3. **Embedding** — generate semantic vectors for new records (`embeddings.py`)
4. **Retrieval** — keyword + cosine similarity search over stored evidence (`retriever.py`)
5. **Brief Generation** — LLM synthesizes evidence with inline citations (`generator.py`)
6. **Rendering** — convert `BriefOutput` to markdown (`renderer.py`)

Every step gracefully degrades: missing API keys skip that source; LLM failures fall back to raw-evidence briefs.

## Citation System

Citations are the core differentiator. The LLM is prompted to use `[SOURCE:source_type:source_id:date]` format. Post-processing maps these back to `Citation` objects containing `source_type`, `source_id`, `timestamp`, `excerpt`, and `snippet_hash`. Every section of `BriefOutput` carries its own `citations` list.

## Deployment

- **Docker**: `Dockerfile` builds a `python:3.11-slim` image; entry point is `uvicorn app.api:app`
- **Railway**: `railway.toml` configures Dockerfile build, health check at `/health`, restart on failure
- **CI/CD**: `.github/workflows/deploy-railway.yml` — runs `pytest` on all pushes/PRs; deploys to Railway on merge to `main`

## Code Conventions

- **Type hints** throughout — use `from __future__ import annotations` for forward references
- **Module-level docstrings** on all files
- **Logging** via stdlib `logging` with module-level `logger = logging.getLogger(__name__)`
- **Ruff** for linting: 100-char line limit, Python 3.11 target
- **Pydantic models** for all data boundaries (API requests/responses, config, brief output)
- **Graceful degradation** — missing external service configs log warnings but don't crash
- **No `async` ORM sessions** — SQLAlchemy sessions are synchronous despite FastAPI being async

## API Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/health` | No | Config/status check |
| POST | `/brief` | Yes | Full brief (JSON + markdown) |
| POST | `/brief/markdown` | Yes | Markdown-only brief |
| POST | `/brief/json` | Yes | Structured JSON brief |
| GET | `/next-steps` | Yes | Action items from emails |
| POST | `/sync` | Yes | Manual Fireflies/Gmail resync |
| GET | `/profiles` | Yes | All contact profiles |
| GET | `/stats` | Yes | Dashboard statistics |
| GET | `/briefs/recent` | Yes | Audit log of recent briefs |
| GET | `/` | No | Web dashboard (static HTML) |

Auth is Bearer token via `BRIEFING_API_KEY`. Disabled when the env var is unset.

## Common Pitfalls

- The `DATABASE_URL` must use `postgresql://` (not `postgres://`) for SQLAlchemy 2.0+. The `Settings.effective_database_url` property handles this conversion for Railway-injected URLs.
- Gmail OAuth requires either credential files on disk or the three `GOOGLE_*` env vars. The `scripts/setup_gmail_oauth.py` script helps generate tokens interactively.
- The `AgendaVariant` model references `AgendaBlock` before its definition — `model_rebuild()` is called to fix forward refs. Don't reorder these classes in `models.py`.
- Background sync runs in a thread with a lock (`auto_sync.py`). It starts automatically if `FIREFLIES_API_KEY` is set.
