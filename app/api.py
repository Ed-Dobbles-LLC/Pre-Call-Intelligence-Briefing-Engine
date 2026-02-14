"""FastAPI web API for the Pre-Call Intelligence Briefing Engine.

Exposes the briefing pipeline as HTTP endpoints for Railway / cloud deployment.
Serves a built-in web dashboard at the root URL.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.brief.pipeline import run_pipeline
from app.brief.profiler import build_interactions_summary, generate_deep_profile
from app.config import settings, validate_config
from app.models import BriefOutput
from app.store.database import BriefLog, EntityRecord, get_session, init_db
from app.clients.apollo import ApolloClient, normalize_candidate
from app.sync.auto_sync import (
    _extract_next_steps,
    async_sync_fireflies,
    get_all_profiles,
    get_dashboard_stats,
    start_background_sync,
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
    apollo_configured: bool


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
        gmail_configured=bool(settings.google_client_id and settings.google_refresh_token),
        openai_configured=bool(settings.openai_api_key),
        apollo_configured=bool(settings.apollo_api_key),
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

@app.get("/next-steps", dependencies=[Depends(verify_api_key)])
def get_next_steps():
    """Return extracted next steps from recent scheduling emails and profile data."""
    return _extract_next_steps()


@app.post("/sync", dependencies=[Depends(verify_api_key)])
async def trigger_sync():
    """Trigger a manual sync of Fireflies transcripts and profile rebuild."""
    result = await async_sync_fireflies()
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@app.get("/profiles", dependencies=[Depends(verify_api_key)])
def list_profiles():
    """Return all auto-generated contact profiles."""
    return get_all_profiles()


class ConfirmLinkedInRequest(BaseModel):
    candidate_index: int = Field(
        ..., ge=-1,
        description="Index into linkedin_candidates, or -1 for 'none of the above'",
    )


@app.get("/profiles/pending-review", dependencies=[Depends(verify_api_key)])
def profiles_pending_review():
    """Return profiles that need LinkedIn disambiguation.

    Includes both pending_review (have candidates) and no_match (need manual search).
    """
    all_p = get_all_profiles()
    return [
        p for p in all_p
        if p.get("linkedin_status") in ("pending_review", "no_match")
    ]


@app.post("/profiles/{profile_id}/confirm-linkedin", dependencies=[Depends(verify_api_key)])
def confirm_linkedin(profile_id: int, request: ConfirmLinkedInRequest):
    """Confirm or reject a LinkedIn candidate for a profile.

    candidate_index = -1 means 'none of the above' (mark as no_match).
    """
    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")
        candidates = profile_data.get("linkedin_candidates", [])

        if request.candidate_index == -1:
            profile_data["linkedin_status"] = "no_match"
            profile_data["linkedin_candidates"] = []
            entity.domains = json.dumps(profile_data)
            session.commit()
            return {"status": "ok", "linkedin_status": "no_match"}

        if request.candidate_index >= len(candidates):
            raise HTTPException(status_code=422, detail="Invalid candidate index")

        chosen = candidates[request.candidate_index]

        profile_data["photo_url"] = chosen.get("photo_url", "")
        profile_data["linkedin_url"] = chosen.get("linkedin_url", "")
        profile_data["title"] = chosen.get("title", "")
        profile_data["headline"] = chosen.get("headline", "")
        profile_data["seniority"] = chosen.get("seniority", "")
        profile_data["location"] = ", ".join(
            filter(None, [chosen.get("city", ""), chosen.get("state", "")])
        )
        if chosen.get("company_name") and not profile_data.get("company"):
            profile_data["company"] = chosen["company_name"]
        elif chosen.get("company_name"):
            profile_data["company_full"] = chosen["company_name"]
        profile_data["company_industry"] = chosen.get("company_industry", "")
        profile_data["company_size"] = chosen.get("company_size")
        profile_data["company_linkedin"] = chosen.get("company_linkedin", "")

        profile_data["linkedin_status"] = "confirmed"
        profile_data["linkedin_candidates"] = []

        entity.domains = json.dumps(profile_data)
        session.commit()
        return {"status": "ok", "linkedin_status": "confirmed"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Failed to confirm LinkedIn for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Failed to update profile")
    finally:
        session.close()


class SearchLinkedInRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search terms (name, company, etc.)")


@app.post("/profiles/{profile_id}/search-linkedin", dependencies=[Depends(verify_api_key)])
async def search_linkedin(profile_id: int, request: SearchLinkedInRequest):
    """Search Apollo for LinkedIn candidates using custom search terms.

    Updates the profile with new candidates and sets status to pending_review.
    """
    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        client = ApolloClient()
        if not client.api_key:
            raise HTTPException(status_code=400, detail="Apollo API key not configured")

        # Split query into name and optional company (separated by @)
        parts = request.query.split("@", 1)
        name = parts[0].strip()
        company = parts[1].strip() if len(parts) > 1 else None

        candidates_raw = await client.search_people(
            name=name,
            organization_name=company,
            per_page=5,
        )

        candidates = [
            normalize_candidate(c)
            for c in candidates_raw
            if c.get("linkedin_url") or c.get("photo_url")
        ]

        profile_data = json.loads(entity.domains or "{}")
        profile_data["linkedin_candidates"] = candidates
        profile_data["linkedin_status"] = "pending_review" if candidates else "no_match"
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {"status": "ok", "candidates": candidates}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("LinkedIn search failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Search failed")
    finally:
        session.close()


class SetLinkedInRequest(BaseModel):
    linkedin_url: str = Field(..., min_length=1, description="LinkedIn profile URL")


@app.post("/profiles/{profile_id}/set-linkedin", dependencies=[Depends(verify_api_key)])
def set_linkedin(profile_id: int, request: SetLinkedInRequest):
    """Manually set a LinkedIn URL for a profile."""
    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")
        profile_data["linkedin_url"] = request.linkedin_url
        profile_data["linkedin_status"] = "confirmed"
        profile_data["linkedin_candidates"] = []
        entity.domains = json.dumps(profile_data)
        session.commit()
        return {"status": "ok", "linkedin_status": "confirmed"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Failed to set LinkedIn for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Failed to update profile")
    finally:
        session.close()


@app.post("/profiles/{profile_id}/deep-profile", dependencies=[Depends(verify_api_key)])
def generate_profile_research(profile_id: int):
    """Generate a deep intelligence profile for a verified contact.

    Uses the LLM to produce a structured executive dossier with career analysis,
    strategic patterns, conversation playbook, and risk signals.
    Only works for profiles with linkedin_status == 'confirmed'.
    """
    if not settings.openai_api_key:
        raise HTTPException(status_code=400, detail="OpenAI API key not configured")

    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")

        if profile_data.get("linkedin_status") != "confirmed":
            raise HTTPException(
                status_code=422,
                detail="Profile must be verified before generating deep research",
            )

        interactions_summary = build_interactions_summary(profile_data)

        result = generate_deep_profile(
            name=entity.name,
            title=profile_data.get("title", ""),
            company=profile_data.get("company", ""),
            linkedin_url=profile_data.get("linkedin_url", ""),
            location=profile_data.get("location", ""),
            industry=profile_data.get("company_industry", ""),
            company_size=profile_data.get("company_size"),
            interactions_summary=interactions_summary,
        )

        profile_data["deep_profile"] = result
        profile_data["deep_profile_generated_at"] = datetime.utcnow().isoformat()
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok",
            "deep_profile": result,
            "generated_at": profile_data["deep_profile_generated_at"],
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Deep profile generation failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Deep profile generation failed")
    finally:
        session.close()


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
    return FileResponse(
        _static_dir / "index.html",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Mount static assets last so it doesn't shadow API routes
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")
