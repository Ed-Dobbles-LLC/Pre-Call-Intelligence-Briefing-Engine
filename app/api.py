"""FastAPI web API for the Pre-Call Intelligence Briefing Engine.

Exposes the briefing pipeline as HTTP endpoints for Railway / cloud deployment.
Serves a built-in web dashboard at the root URL.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.brief.pipeline import run_pipeline
from app.config import settings, validate_config
from app.models import BriefOutput
from app.store.database import BriefLog, get_session, init_db
from app.sync.auto_sync import (
    get_all_profiles,
    get_dashboard_stats,
    start_background_sync,
    sync_fireflies_transcripts,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_config()
    logger.info("Initialising database...")
    try:
        init_db()
    except Exception:
        logger.exception("Database init failed – running in degraded mode")
    # Start background auto-sync for Fireflies transcripts
    if settings.fireflies_api_key:
        start_background_sync(interval_minutes=30)
        logger.info("Auto-sync enabled (every 30 minutes)")
    else:
        logger.info("Auto-sync disabled (no Fireflies API key)")

    logger.info("Briefing Engine API ready")
    yield


app = FastAPI(
    title="Pre-Call Intelligence Briefing Engine",
    version="0.1.0",
    description="Generate decision-grade Pre-Call Briefs from Fireflies transcripts and Gmail emails.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_bearer_scheme = HTTPBearer(auto_error=False)


def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
):
    """Require a valid Bearer token when BRIEFING_API_KEY is set."""
    expected = settings.briefing_api_key
    if not expected:
        return  # auth disabled – no key configured
    if not credentials or credentials.credentials != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class BriefRequest(BaseModel):
    person: Optional[str] = Field(None, description="Person name to brief on")
    company: Optional[str] = Field(None, description="Company name to brief on")
    topic: Optional[str] = Field(None, description="Meeting topic")
    meeting_when: Optional[str] = Field(
        None,
        description="Meeting datetime (YYYY-MM-DD HH:MM or ISO format)",
        examples=["2026-02-15 14:00"],
    )
    skip_ingestion: bool = Field(
        False,
        description="If true, skip Fireflies/Gmail API calls and use only stored data",
    )


class BriefResponse(BaseModel):
    brief: BriefOutput
    markdown: str
    confidence_score: float
    source_count: int


class HealthResponse(BaseModel):
    status: str
    version: str
    database: str
    fireflies_configured: bool
    gmail_configured: bool
    openai_configured: bool


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check – verifies the service is running and shows config status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database="sqlite" if settings.is_sqlite else "postgres",
        fireflies_configured=bool(settings.fireflies_api_key),
        gmail_configured=bool(settings.gmail_credentials_path),
        openai_configured=bool(settings.openai_api_key),
    )


@app.post("/brief", response_model=BriefResponse, dependencies=[Depends(verify_api_key)])
def generate_brief_endpoint(request: BriefRequest):
    """Generate a Pre-Call Intelligence Brief.

    Requires at least `person` or `company` to be provided.
    """
    if not request.person and not request.company:
        raise HTTPException(
            status_code=422,
            detail="At least 'person' or 'company' must be provided.",
        )

    try:
        result = run_pipeline(
            person=request.person,
            company=request.company,
            topic=request.topic,
            meeting_when=request.meeting_when,
            skip_ingestion=request.skip_ingestion,
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Brief generation failed. Check server logs.")

    return BriefResponse(
        brief=result.brief,
        markdown=result.markdown,
        confidence_score=result.brief.header.confidence_score,
        source_count=len(result.brief.appendix_evidence),
    )


@app.post("/brief/markdown", response_class=PlainTextResponse, dependencies=[Depends(verify_api_key)])
def generate_brief_markdown(request: BriefRequest):
    """Generate a brief and return only the markdown (for quick consumption)."""
    if not request.person and not request.company:
        raise HTTPException(
            status_code=422,
            detail="At least 'person' or 'company' must be provided.",
        )

    try:
        result = run_pipeline(
            person=request.person,
            company=request.company,
            topic=request.topic,
            meeting_when=request.meeting_when,
            skip_ingestion=request.skip_ingestion,
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Brief generation failed.")

    return result.markdown


@app.post("/brief/json", dependencies=[Depends(verify_api_key)])
def generate_brief_json(request: BriefRequest):
    """Generate a brief and return only the structured JSON."""
    if not request.person and not request.company:
        raise HTTPException(
            status_code=422,
            detail="At least 'person' or 'company' must be provided.",
        )

    try:
        result = run_pipeline(
            person=request.person,
            company=request.company,
            topic=request.topic,
            meeting_when=request.meeting_when,
            skip_ingestion=request.skip_ingestion,
        )
    except Exception:
        logger.exception("Pipeline failed")
        raise HTTPException(status_code=500, detail="Brief generation failed.")

    return result.brief.model_dump()


# ---------------------------------------------------------------------------
# Sync & Profiles
# ---------------------------------------------------------------------------

@app.post("/sync", dependencies=[Depends(verify_api_key)])
def trigger_sync():
    """Trigger a manual sync of Fireflies transcripts and profile rebuild."""
    result = sync_fireflies_transcripts()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/profiles", dependencies=[Depends(verify_api_key)])
def list_profiles():
    """Return all auto-generated contact profiles."""
    return get_all_profiles()


@app.get("/stats", dependencies=[Depends(verify_api_key)])
def dashboard_stats():
    """Return summary statistics for the dashboard."""
    return get_dashboard_stats()


# ---------------------------------------------------------------------------
# Recent briefs (for dashboard)
# ---------------------------------------------------------------------------

@app.get("/briefs/recent", dependencies=[Depends(verify_api_key)])
def list_recent_briefs(limit: int = Query(20, ge=1, le=100)):
    """Return recent brief audit log entries (newest first)."""
    session = get_session()
    try:
        rows = (
            session.query(BriefLog)
            .order_by(BriefLog.created_at.desc())
            .limit(limit)
            .all()
        )
        return [
            {
                "id": row.id,
                "person": row.person,
                "company": row.company,
                "topic": row.topic,
                "confidence_score": row.confidence_score,
                "brief_json": json.loads(row.brief_json) if row.brief_json else None,
                "brief_markdown": row.brief_markdown,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ]
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Dashboard (static files)
# ---------------------------------------------------------------------------

_static_dir = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def serve_dashboard():
    """Serve the web dashboard."""
    return FileResponse(_static_dir / "index.html")


# Mount static assets last so it doesn't shadow API routes
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
