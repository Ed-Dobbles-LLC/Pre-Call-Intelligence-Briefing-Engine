# CLAUDE.md — Pre-Call Intelligence Briefing Engine

## Project Overview

A decision-grade meeting briefing and contact intelligence tool. Two output modes:

- **Mode A (Meeting-Prep Brief)** — fast, internal-only, no web required. Produces actionable prep from meeting transcripts and emails.
- **Mode B (Deep Research Dossier)** — web-required, fail-closed, 12-section structural intelligence dossier with strategic inference layers.

Every claim traces to a source via evidence tags (`[VERIFIED-*]`, `[INFERRED-*]`, `[STRATEGIC MODEL]`, `[UNKNOWN]`). The system is fail-closed: missing evidence halts output rather than fabricating.

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
    renderer.py       # Markdown rendering of BriefOutput (Mode A)
    profiler.py       # Mode B: Deep Research dossier prompt + LLM generation
    evidence_graph.py # Evidence Graph, fail-closed gates, dossier mode logic
    qa.py             # Entity lock scoring, QA gates, evidence coverage
  clients/
    apollo.py         # Apollo.io People Enrichment API client
    fireflies.py      # Fireflies.ai GraphQL API client
    gmail.py          # Gmail API OAuth2 client
    openai_client.py  # OpenAI chat + embeddings wrapper
  services/
    photo_resolution.py  # Photo resolution service (Gravatar, Clearbit, uploaded)
    enrichment_service.py # PDL enrichment with photo download + caching
    linkedin_pdf.py      # LinkedIn PDF ingestion (text extraction + headshot crop)
    artifact_dossier.py  # PDF-first artifact dossier generation
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
    auto_sync.py         # Background thread: periodic Fireflies/Gmail sync + profile management
  cli/
    main.py              # Click CLI entry point
  static/
    index.html           # Web dashboard (single-page)
tests/
  conftest.py            # Fixtures: test DB, sample payloads
  test_api.py            # FastAPI endpoint tests
  test_citations.py      # Citation extraction + formatting
  test_entity_matching.py # Entity resolution logic
  test_evidence_graph.py # Evidence graph, gates, coverage, dossier mode tests
  test_gates.py          # QA gates, entity lock, gate status tests
  test_profiler.py       # Deep Research prompt template + generation tests
  test_deep_research.py  # Full deep research pipeline + API tests
  test_ingestion.py      # Fireflies/Gmail normalization
  test_retrieval.py      # Evidence retrieval + search
migrations/
  001_initial_schema.sql # Base schema (SQLite + Postgres)
  002_postgres_railway.sql # Railway/Postgres-specific indexes
scripts/
  setup_gmail_oauth.py   # Interactive Gmail OAuth credential setup
image_cache/             # Locally cached photos (LinkedIn crops, PDL, uploaded)
```

## Key Entry Points

| What | Location |
|------|----------|
| FastAPI app | `app/api.py:app` |
| CLI entry point | `app/cli/main.py:cli` (installed as `brief` command) |
| Pipeline orchestrator | `app/brief/pipeline.py:run_pipeline()` |
| Deep Research (Mode B) | `app/brief/profiler.py:generate_deep_profile()` |
| Evidence Graph engine | `app/brief/evidence_graph.py` |
| QA / Entity Lock | `app/brief/qa.py` |
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

### Mode A: Meeting-Prep Brief

The core pipeline in `app/brief/pipeline.py:run_pipeline()` follows this flow:

1. **Entity Resolution** — map person/company names to stored entities (`entity_resolver.py`)
2. **Ingestion** — fetch from Fireflies/Gmail APIs, normalize, store (`ingest/`)
3. **Embedding** — generate semantic vectors for new records (`embeddings.py`)
4. **Retrieval** — keyword + cosine similarity search over stored evidence (`retriever.py`)
5. **Brief Generation** — LLM synthesizes evidence with inline citations (`generator.py`)
6. **Rendering** — convert `BriefOutput` to markdown (`renderer.py`)

Every step gracefully degrades: missing API keys skip that source; LLM failures fall back to raw-evidence briefs.

### Mode B: Deep Research Dossier

The deep research pipeline in `app/api.py:deep_research_endpoint()` follows this flow:

1. **Auto-enrich** — PDL enrichment if not recently enriched (`enrichment_service.py`)
2. **Search Plan** — generate targeted search queries for the subject
3. **Evidence Graph** — initialize and populate with meeting/email/PDF/web nodes (`evidence_graph.py`)
4. **SerpAPI Retrieval** — execute visibility sweep (15+ queries) + bio/press searches
5. **Entity Lock** — score identity confidence 0-100 (`qa.py:EntityLock`)
6. **Dossier Mode** — determine FULL/CONSTRAINED/HALTED based on gates
7. **LLM Synthesis** — generate 12-section dossier via `profiler.py:generate_deep_profile()`
8. **QA Gates** — enforce evidence coverage, visibility sweep, entity lock thresholds
9. **Persist** — save dossier markdown and metadata to profile

### Fail-Closed Gates (Mode B)

| Gate | Condition | Result |
|------|-----------|--------|
| Visibility Sweep | < 8 queries executed | **HALT** — no dossier generated |
| Evidence Coverage | < adaptive threshold (85%/70%/60%) | **HALT** — no dossier generated |
| Entity Lock | < 60 | **CONSTRAIN** — suppress INFERRED-H/M and STRATEGIC MODEL |
| No Public Results | 0 SerpAPI results | **HALT** — no dossier generated |

## Evidence Tagging System

Every non-trivial claim in a dossier carries one evidence tag:

| Tag | Meaning |
|-----|---------|
| `[VERIFIED–MEETING]` | Explicitly stated in meeting transcript/email |
| `[VERIFIED–PUBLIC]` | Documented in a cited public source (URL required) |
| `[VERIFIED-PDF]` | Stated in uploaded LinkedIn PDF or resume |
| `[INFERRED–H]` | High-confidence inference from multiple converging signals |
| `[INFERRED–M]` | Medium-confidence inference from limited signals |
| `[INFERRED–L]` | Low-confidence inference from weak or single signal |
| `[STRATEGIC MODEL]` | Structured reasoning derived from verified/inferred evidence |
| `[UNKNOWN]` | No supporting evidence — gap explicitly declared |

**STRATEGIC MODEL** sections do not require per-sentence tags. The section header must cite the upstream evidence nodes (e.g., `[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]`). Only enabled when `identity_lock_score >= 60`.

## Deep Research Dossier Sections (Mode B)

The 12 required sections in order:

1. **Executive Summary** — verified facts + decision-useful inference
2. **Identity & Disambiguation** — identifiers, name collision risks
3. **Career Timeline** — chronological roles with evidence tags
4. **Public Statements & Positions** — organized by topic, direct quotes
5. **Public Visibility** — TED/keynote/podcast/conference sweep results
6. **Quantified Claims Inventory** — separated into Personal Ownership / Engagement Outcome / Marketing-Level claims with claim style pattern analysis
7. **Rhetorical & Decision Patterns** — language bias, decision style, red flags
8. **Structural Pressure Model** — mandate, pressures, vendor posture
9. **Structural Incentive & Power Model** *(strategic)* — revenue mandate, org power, political capital, competitive positioning, growth pressure, failure consequences, budget authority, build vs buy
10. **Competitive Positioning Context** *(strategic)* — market position, competitors, consulting vs product mix, AI maturity, organizational role type classification
11. **How to Win This Decision-Maker** *(strategic)* — what makes them look smart, what they're measured on, failure cost, narrative resonance, threat/strengthen positioning, what NOT to do
12. **Primary Source Index** — all sources with URLs

Sections 9-11 use `[STRATEGIC MODEL]` tags and are suppressed when `identity_lock_score < 60`.

## Citation System (Mode A)

Citations are the core differentiator for Mode A briefs. The LLM is prompted to use `[SOURCE:source_type:source_id:date]` format. Post-processing maps these back to `Citation` objects containing `source_type`, `source_id`, `timestamp`, `excerpt`, and `snippet_hash`. Every section of `BriefOutput` carries its own `citations` list.

## Entity Lock Scoring

Entity lock (`app/brief/qa.py:EntityLock`) scores identity confidence 0-100:

| Signal | Points |
|--------|--------|
| LinkedIn URL present | +10 (weak) |
| LinkedIn verified by retrieval | +30 (strong) |
| Meeting confirms identity | +20 |
| Employer in public source | +20 |
| Multiple independent sources agree | +20 |
| Title in public source | +10 |
| Location in public source | +10 |

**Thresholds** (inference gating, NOT the score calculation):
- `>= 60`: FULL mode — strategic inference + STRATEGIC MODEL enabled
- `50-59`: CONSTRAINED — suppress INFERRED-H/M and STRATEGIC MODEL blocks
- `< 50`: CONSTRAINED — suppress ALL INFERRED tags and STRATEGIC MODEL blocks

Note: `EntityLock.is_locked` property still uses 70 as its semantic label threshold. The inference gating threshold (used by `determine_dossier_mode()` and `filter_prose_by_mode()`) is 60.

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
| POST | `/profiles/{id}/deep-research` | Yes | Full Deep Research pipeline (Mode B) |
| POST | `/profiles/{id}/ingest-linkedin-pdf` | Yes | Upload + process LinkedIn PDF |
| POST | `/profiles/{id}/upload-photo` | Yes | Upload JPG/PNG photo for contact |
| POST | `/profiles/resolve-photos` | Yes | Resolve photos for all contacts |
| POST | `/profiles/{id}/photo-render-failed` | Yes | Client callback for failed photo render |
| GET | `/api/local-image/{path}` | No | Serve cached images from image_cache/ |
| GET | `/api/image-proxy` | No | Proxy + cache remote images |
| GET | `/` | No | Web dashboard (single-page) |

Auth is Bearer token via `BRIEFING_API_KEY`. Disabled when the env var is unset.

## Common Pitfalls

- The `DATABASE_URL` must use `postgresql://` (not `postgres://`) for SQLAlchemy 2.0+. The `Settings.effective_database_url` property handles this conversion for Railway-injected URLs.
- Gmail OAuth requires either credential files on disk or the three `GOOGLE_*` env vars. The `scripts/setup_gmail_oauth.py` script helps generate tokens interactively.
- The `AgendaVariant` model references `AgendaBlock` before its definition — `model_rebuild()` is called to fix forward refs. Don't reorder these classes in `models.py`.
- Background sync runs in a thread with a lock (`auto_sync.py`). It starts automatically if `FIREFLIES_API_KEY` is set.
- **Entity lock threshold vs label**: `ENTITY_LOCK_THRESHOLD` (60) is used by `determine_dossier_mode()` for inference gating. `EntityLock.is_locked` (70) is a semantic label. Don't confuse them — changing one doesn't change the other.
- **Photo URLs**: Locally cached photos must use `/api/local-image/./image_cache/...` URLs (not raw filesystem paths like `./image_cache/...`) so the browser can load them. See `upload-photo` and `ingest-linkedin-pdf` endpoints.
- **Sync carry-forward**: When adding new profile fields stored in `EntityRecord.domains` JSON, add them to the carry-forward allowlist in `auto_sync.py:_update_profiles()` or they will be wiped on the next Fireflies sync cycle.
- **USER_PROMPT_TEMPLATE** has 12 format placeholders including `{inference_gate_instruction}`. All must be provided when calling `.format()` — tests that format the template directly must include this field.
