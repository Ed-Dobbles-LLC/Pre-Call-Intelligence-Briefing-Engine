"""FastAPI web API for the Pre-Call Intelligence Briefing Engine.

Exposes the briefing pipeline as HTTP endpoints for Railway / cloud deployment.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from app.brief.pipeline import run_pipeline
from app.config import settings
from app.models import BriefOutput
from app.store.database import init_db

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initialising database...")
    init_db()
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


@app.post("/brief", response_model=BriefResponse)
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


@app.post("/brief/markdown", response_class=PlainTextResponse)
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


@app.post("/brief/json")
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
