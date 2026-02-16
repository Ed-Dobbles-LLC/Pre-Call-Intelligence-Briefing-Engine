"""FastAPI web API for the Pre-Call Intelligence Briefing Engine.

Exposes the briefing pipeline as HTTP endpoints for Railway / cloud deployment.
Serves a built-in web dashboard at the root URL.
"""

from __future__ import annotations

import json
import logging
import os
import threading
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
from app.brief.evidence_graph import (
    DossierMode,
    EvidenceGraph,
    build_failure_report,
    build_meeting_prep_brief,
    compute_visibility_coverage_confidence,
    determine_dossier_mode,
    extract_highest_signal_artifacts,
    filter_prose_by_mode,
)
from app.brief.qa import (
    enforce_fail_closed_gates,
    generate_dossier_qa_report,
    render_qa_report_markdown,
    score_disambiguation,
)
from app.clients.serpapi import (
    SerpAPIClient,
    VISIBILITY_CATEGORIES,
    format_visibility_results_for_prompt,
    format_web_results_for_prompt,
    generate_search_plan,
)
from app.config import settings, validate_config
from app.models import BriefOutput
from app.store.database import BriefLog, EntityRecord, get_session, init_db
from app.clients.apollo import ApolloClient, normalize_candidate, normalize_enrichment
from app.sync.auto_sync import (
    _extract_next_steps,
    _re_enrich_confirmed_profiles,
    async_sync_fireflies,
    get_all_profiles,
    get_dashboard_stats,
    repair_linkedin_status,
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
    # Repair linkedin_status only when explicitly opted-in via env var.
    # This avoids blocking startup or crashlooping on lock-contention /
    # schema-mismatch.  Run in a background thread so it never blocks
    # the lifespan yield (and therefore the health-check).
    if os.getenv("RUN_REPAIR_ON_STARTUP", "").lower() in ("1", "true", "yes"):
        def _deferred_repair():
            try:
                repaired = repair_linkedin_status()
                if repaired:
                    logger.info("Repaired %d profiles with missing linkedin_status", repaired)
            except Exception:
                logger.exception("LinkedIn status repair failed")

        threading.Thread(target=_deferred_repair, daemon=True).start()
        logger.info("LinkedIn status repair scheduled (background thread)")
    else:
        logger.info("LinkedIn status repair skipped (RUN_REPAIR_ON_STARTUP not set)")

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
    try:
        result = await async_sync_fireflies()
    except Exception as exc:
        logger.exception("Sync endpoint: unexpected error")
        raise HTTPException(status_code=500, detail=f"Sync failed unexpectedly: {exc}")
    if "error" in result:
        # Return 409 for "already in progress" vs 400 for config errors
        status = 409 if "already in progress" in result["error"].lower() else 400
        raise HTTPException(status_code=status, detail=result["error"])
    return result


# ---------------------------------------------------------------------------
# Calendar + Photo Resolution Endpoints
# ---------------------------------------------------------------------------


@app.post("/sync/calendar", dependencies=[Depends(verify_api_key)])
async def trigger_calendar_sync():
    """Trigger a manual Google Calendar ingest for next 7 days.

    Fetches upcoming events, matches attendees to contacts,
    creates stubs for unknowns, and stores meeting associations.
    """
    if not settings.google_client_id or not settings.google_refresh_token:
        raise HTTPException(
            status_code=400,
            detail="Google Calendar requires GOOGLE_CLIENT_ID + GOOGLE_REFRESH_TOKEN",
        )

    from app.ingest.calendar_ingest import run_calendar_ingest
    result = run_calendar_ingest(days=7)
    return result.to_dict()


@app.post("/sync/calendar/enrich", dependencies=[Depends(verify_api_key)])
async def trigger_meeting_enrichment():
    """Run Gmail context enrichment for all upcoming meetings.

    For each upcoming meeting attendee, fetches recent email threads
    and extracts summaries, last contact dates, and open commitments.
    """
    from app.ingest.gmail_meeting_enrichment import enrich_all_upcoming_meetings
    enrichments = enrich_all_upcoming_meetings()
    return {"status": "ok", "enrichments": len(enrichments), "results": enrichments}


@app.post("/profiles/repair", dependencies=[Depends(verify_api_key)])
async def repair_profiles():
    """Repair corrupted profile data from bad PDF ingestion.

    Scans all profiles and:
    1. Detects/clears garbled text (headline, raw_text, sections)
    2. Resets corrupted PDF-crop photos and re-resolves via fallback chain
    3. Clears corrupted artifact dossiers built from garbled text
    4. Auto-triggers photo resolution for contacts that lost photos

    Safe to run multiple times — idempotent.
    """
    from app.services.linkedin_pdf import _is_garbled_text
    from app.services.photo_resolution import PhotoSource, resolve_photo_for_profile

    session = get_session()
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person"
        ).all()

        repaired = 0
        photos_resolved = 0
        details = []

        for entity in entities:
            profile_data = json.loads(entity.domains or "{}")
            changes = []

            # --- Fix garbled headline ---
            headline = profile_data.get("headline", "")
            if headline and _is_garbled_text(headline):
                profile_data["headline"] = ""
                changes.append("cleared garbled headline")

            # --- Fix garbled location ---
            location = profile_data.get("location", "")
            if location and _is_garbled_text(location):
                profile_data["location"] = ""
                changes.append("cleared garbled location")

            # --- Fix garbled PDF raw text ---
            raw_text = profile_data.get("linkedin_pdf_raw_text", "")
            if raw_text and _is_garbled_text(raw_text):
                profile_data["linkedin_pdf_raw_text"] = ""
                profile_data["linkedin_pdf_sections"] = {}
                profile_data["linkedin_pdf_text_length"] = 0
                profile_data["linkedin_pdf_text_usable"] = False
                profile_data["linkedin_pdf_experience"] = []
                profile_data["linkedin_pdf_education"] = []
                profile_data["linkedin_pdf_skills"] = []
                changes.append("cleared garbled PDF text/sections/experience")

            # --- Fix corrupted PDF-crop photos ---
            photo_source = profile_data.get("photo_source", "")
            if photo_source == PhotoSource.LINKEDIN_PDF_CROP:
                profile_data["photo_url"] = ""
                profile_data["photo_source"] = ""
                profile_data["photo_status"] = ""
                changes.append("cleared corrupted PDF crop photo")

            # --- Fix corrupted artifact dossier ---
            artifact_md = profile_data.get("artifact_dossier_markdown", "")
            if artifact_md and _is_garbled_text(artifact_md[:500]):
                profile_data["artifact_dossier_markdown"] = ""
                profile_data["artifact_dossier_generated_at"] = ""
                changes.append("cleared garbled artifact dossier")

            # --- Auto-resolve photos for contacts without photos ---
            if not profile_data.get("photo_url"):
                emails = json.loads(entity.emails or "[]")
                email = emails[0] if emails else ""
                try:
                    # Build a mini-profile dict for the resolution service
                    resolve_input = {
                        "name": entity.name,
                        "email": email,
                        "linkedin_url": profile_data.get("linkedin_url", ""),
                        "company_domain": "",
                        "photo_url": "",
                        "photo_source": "",
                        "photo_status": "",
                    }
                    resolve_photo_for_profile(resolve_input)
                    if resolve_input.get("photo_url"):
                        profile_data["photo_url"] = resolve_input["photo_url"]
                        profile_data["photo_source"] = resolve_input.get(
                            "photo_source", ""
                        )
                        profile_data["photo_status"] = resolve_input.get(
                            "photo_status", ""
                        )
                        profile_data["photo_last_checked_at"] = resolve_input.get(
                            "photo_last_checked_at", ""
                        )
                        photos_resolved += 1
                        changes.append(
                            f"resolved photo via {resolve_input.get('photo_source', 'fallback')}"
                        )
                except Exception as e:
                    logger.debug("Photo resolution failed for %s: %s", entity.name, e)

            if changes:
                entity.domains = json.dumps(profile_data)
                repaired += 1
                details.append({
                    "id": entity.id,
                    "name": entity.name,
                    "repairs": changes,
                })

        session.commit()
        return {
            "status": "ok",
            "profiles_scanned": len(entities),
            "profiles_repaired": repaired,
            "photos_resolved": photos_resolved,
            "details": details,
        }
    except Exception:
        session.rollback()
        logger.exception("Profile repair failed")
        raise HTTPException(status_code=500, detail="Profile repair failed")
    finally:
        session.close()


@app.post("/profiles/resolve-photos", dependencies=[Depends(verify_api_key)])
async def resolve_all_photos():
    """Run PhotoResolutionService on all contacts.

    RULE: Never wipe an existing photo_url. If a contact has a photo and it
    hasn't been marked FAILED_RENDER, the resolver preserves it as-is.
    Also runs backfill to restore UNKNOWN status on profiles that had their
    status incorrectly set to MISSING.
    """
    from app.services.photo_resolution import PhotoResolutionService, backfill_photo_status
    session = get_session()
    service = PhotoResolutionService()
    updated = 0
    backfilled = 0
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person"
        ).all()

        for entity in entities:
            profile_data = json.loads(entity.domains or "{}")
            emails = entity.get_emails()

            # Backfill: restore UNKNOWN status for profiles with URLs but MISSING status
            old_status = profile_data.get("photo_status", "")
            backfill_photo_status(profile_data)
            if profile_data.get("photo_status") != old_status:
                backfilled += 1

            result = service.resolve(
                contact_id=entity.id,
                contact_name=entity.name,
                email=emails[0] if emails else "",
                linkedin_url=profile_data.get("linkedin_url", ""),
                company_domain=profile_data.get("company_domain", ""),
                existing_photo_url=profile_data.get("photo_url", ""),
                existing_photo_source=profile_data.get("photo_source", ""),
                existing_photo_status=profile_data.get("photo_status", ""),
            )
            profile_data["photo_url"] = result.photo_url
            profile_data["photo_source"] = result.photo_source
            profile_data["photo_status"] = result.photo_status
            profile_data["photo_last_checked_at"] = result.resolved_at
            if result.error:
                profile_data["photo_last_error"] = result.error
            entity.domains = json.dumps(profile_data)
            updated += 1

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Photo resolution failed")
        raise HTTPException(status_code=500, detail="Photo resolution failed")
    finally:
        session.close()

    return {
        "status": "ok",
        "contacts_processed": updated,
        "backfilled_status": backfilled,
        "resolution_logs": [
            {"contact": log.contact_name, "source": log.resolved_source, "error": log.error}
            for log in service.resolution_logs
        ],
    }


@app.post("/profiles/{profile_id}/photo-render-failed", dependencies=[Depends(verify_api_key)])
async def report_photo_render_failed(profile_id: int):
    """Client-side callback: photo failed to render in browser.

    Sets photo_status to FAILED_RENDER so the next resolve-photos run
    will attempt re-resolution instead of preserving the broken URL.
    """
    session = get_session()
    try:
        entity = session.query(EntityRecord).filter(
            EntityRecord.id == profile_id,
            EntityRecord.entity_type == "person",
        ).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")
        profile_data["photo_status"] = "FAILED_RENDER"
        entity.domains = json.dumps(profile_data)
        session.commit()
        return {"status": "ok", "photo_status": "FAILED_RENDER"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Photo render-failed update failed")
        raise HTTPException(status_code=500, detail="Update failed")
    finally:
        session.close()


@app.get("/api/image-proxy")
async def image_proxy(url: str = Query(..., description="Remote image URL to proxy")):
    """Server-side image proxy with caching by URL hash.

    Feature-flagged: LinkedIn CDN proxying is only enabled when
    LINKEDIN_PROXY_ENABLED=true. Other domains are proxied freely.
    Stores fetched images in a local cache directory.
    """
    import hashlib
    import httpx
    from fastapi.responses import Response

    if not url or not url.startswith("http"):
        raise HTTPException(status_code=400, detail="Invalid URL")

    LINKEDIN_HOSTS = ["media.licdn.com", "media-exp1.licdn.com", "static.licdn.com"]
    is_linkedin = any(h in url.lower() for h in LINKEDIN_HOSTS)

    if is_linkedin and not settings.linkedin_proxy_enabled:
        raise HTTPException(
            status_code=403,
            detail="LinkedIn image proxying is disabled. Set LINKEDIN_PROXY_ENABLED=true to enable.",
        )

    if is_linkedin:
        logger.warning(
            "Proxying LinkedIn CDN image — reliability/ToS risk. URL: %s",
            url[:100],
        )

    # Check cache
    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    cache_dir = Path("./image_cache")
    cache_dir.mkdir(exist_ok=True)
    cache_path = cache_dir / url_hash

    if cache_path.exists():
        content = cache_path.read_bytes()
        # Guess content type from extension or default
        ct = "image/jpeg"
        if url.lower().endswith(".png"):
            ct = "image/png"
        elif url.lower().endswith(".webp"):
            ct = "image/webp"
        return Response(content=content, media_type=ct)

    # Fetch remote image
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(url, follow_redirects=True)
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Remote server returned {resp.status_code}",
                )
            content_type = resp.headers.get("content-type", "image/jpeg")
            if not content_type.startswith("image/"):
                raise HTTPException(status_code=502, detail="Remote URL is not an image")

            # Cache it
            cache_path.write_bytes(resp.content)
            return Response(content=resp.content, media_type=content_type)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Failed to fetch image: {e}")


@app.post(
    "/profiles/{profile_id}/ingest-linkedin-pdf",
    dependencies=[Depends(verify_api_key)],
)
async def ingest_linkedin_pdf_endpoint(
    profile_id: int,
    pdf_data: dict | None = None,
):
    """Ingest a LinkedIn profile PDF for a contact.

    Accepts JSON body: {"pdf_base64": "<base64 encoded PDF>"}.

    Extracts:
    1. Structured text (stored for artifact-first dossier)
    2. Headshot image (updates contact photo if crop succeeds)

    Photo update rules:
    - If crop succeeds: set photo_source=linkedin_pdf_crop, photo_status=RESOLVED
    - If crop fails: do NOT overwrite existing photo
    - Never regress: only update photo if it improves the situation
    """
    import base64

    from app.services.linkedin_pdf import ingest_linkedin_pdf
    from app.services.photo_resolution import PhotoSource, PhotoStatus

    if not pdf_data or not pdf_data.get("pdf_base64"):
        raise HTTPException(
            status_code=422,
            detail="Request body must include 'pdf_base64' field with base64-encoded PDF",
        )

    try:
        pdf_bytes = base64.b64decode(pdf_data["pdf_base64"])
    except Exception:
        raise HTTPException(status_code=422, detail="Invalid base64 encoding")

    if not pdf_bytes:
        raise HTTPException(status_code=422, detail="Empty PDF data")

    session = get_session()
    try:
        entity = session.query(EntityRecord).filter(
            EntityRecord.id == profile_id,
        ).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")

        # Run ingestion pipeline
        result = ingest_linkedin_pdf(
            pdf_bytes=pdf_bytes,
            contact_id=profile_id,
            contact_name=entity.name,
        )

        # Store PDF metadata in profile
        profile_data["linkedin_pdf_path"] = result.pdf_path
        profile_data["linkedin_pdf_hash"] = result.pdf_hash
        profile_data["linkedin_pdf_ingested_at"] = result.ingested_at
        profile_data["linkedin_pdf_page_count"] = result.text_result.page_count

        # Check if text extraction produced usable (non-garbled) text
        from app.services.linkedin_pdf import _is_garbled_text
        raw_text = result.text_result.raw_text
        text_is_usable = bool(raw_text) and not _is_garbled_text(raw_text)
        profile_data["linkedin_pdf_text_usable"] = text_is_usable

        if text_is_usable:
            profile_data["linkedin_pdf_text_length"] = len(raw_text)
            # Store structured text sections for dossier
            profile_data["linkedin_pdf_sections"] = result.text_result.sections
            profile_data["linkedin_pdf_raw_text"] = raw_text[:50000]

            # Update profile fields from PDF if not already set
            if result.text_result.headline and not profile_data.get("headline"):
                profile_data["headline"] = result.text_result.headline
            if result.text_result.location and not profile_data.get("location"):
                profile_data["location"] = result.text_result.location

            # Store experience/education for dossier
            if result.text_result.experience:
                profile_data["linkedin_pdf_experience"] = result.text_result.experience
            if result.text_result.education:
                profile_data["linkedin_pdf_education"] = result.text_result.education
            if result.text_result.skills:
                profile_data["linkedin_pdf_skills"] = result.text_result.skills
        else:
            # Don't store garbled text — it corrupts dossier output
            profile_data["linkedin_pdf_text_length"] = 0
            profile_data["linkedin_pdf_raw_text"] = ""
            profile_data["linkedin_pdf_sections"] = {}
            logger.warning(
                "PDF text extraction returned garbled/empty output for profile %d "
                "(%d raw chars); not storing text",
                profile_id, len(raw_text),
            )

        # Update photo if crop succeeded — NEVER regress
        # PDF crops are unreliable for browser-saved PDFs, so only update
        # if there's no existing photo or the existing photo is from a
        # low-priority source (gravatar, company logo, initials, or unknown).
        # NEVER overwrite enrichment provider (Apollo/PDL), cached proxy,
        # or user-uploaded photos with a PDF crop.
        photo_updated = False
        if result.crop_result.success:
            existing_status = profile_data.get("photo_status", "")
            existing_source = profile_data.get("photo_source", "")

            should_update_photo = (
                not profile_data.get("photo_url")
                or existing_status in (
                    PhotoStatus.MISSING, PhotoStatus.FAILED,
                    PhotoStatus.FAILED_RENDER, PhotoStatus.BLOCKED,
                    PhotoStatus.EXPIRED,
                )
                or existing_source in (
                    PhotoSource.GRAVATAR, PhotoSource.COMPANY_LOGO,
                    PhotoSource.INITIALS, "",
                )
            )

            if should_update_photo:
                # Serve cropped image via local image proxy
                profile_data["photo_url"] = (
                    f"/api/local-image/{result.crop_result.image_path}"
                )
                profile_data["photo_source"] = PhotoSource.LINKEDIN_PDF_CROP
                profile_data["photo_status"] = PhotoStatus.RESOLVED
                profile_data["photo_last_checked_at"] = result.ingested_at
                photo_updated = True

        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok",
            "pdf_stored": bool(result.pdf_path),
            "text_extracted": text_is_usable,
            "text_garbled": not text_is_usable and bool(raw_text),
            "text_length": len(raw_text) if text_is_usable else 0,
            "page_count": result.text_result.page_count,
            "sections_found": (
                list(result.text_result.sections.keys()) if text_is_usable else []
            ),
            "headshot_cropped": result.crop_result.success,
            "crop_method": result.crop_result.method,
            "crop_error": (
                result.crop_result.error if not result.crop_result.success else None
            ),
            "photo_updated": photo_updated,
            "experience_entries": (
                len(result.text_result.experience) if text_is_usable else 0
            ),
            "education_entries": (
                len(result.text_result.education) if text_is_usable else 0
            ),
            "skills_count": (
                len(result.text_result.skills) if text_is_usable else 0
            ),
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("LinkedIn PDF ingestion failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="PDF ingestion failed")
    finally:
        session.close()


@app.post(
    "/profiles/{profile_id}/upload-photo",
    dependencies=[Depends(verify_api_key)],
)
async def upload_photo_endpoint(profile_id: int, photo_data: dict | None = None):
    """Upload a JPG/PNG photo for a contact.

    Accepts JSON body: {"photo_base64": "<base64 encoded image>"}.
    Stores the image locally and sets photo_source=uploaded, photo_status=RESOLVED.
    """
    import base64
    import hashlib

    from app.services.photo_resolution import PhotoSource, PhotoStatus

    if not photo_data or "photo_base64" not in photo_data:
        raise HTTPException(status_code=400, detail="Missing photo_base64 in request body")

    session = get_session()
    try:
        entity = session.query(EntityRecord).filter(EntityRecord.id == profile_id).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        raw_b64 = photo_data["photo_base64"]
        try:
            photo_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid base64 data")

        if len(photo_bytes) < 100:
            raise HTTPException(status_code=400, detail="Image too small — likely corrupted")

        # Detect format from magic bytes
        ext = "jpg"
        if photo_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            ext = "png"
        elif photo_bytes[:4] == b"RIFF" and photo_bytes[8:12] == b"WEBP":
            ext = "webp"

        # Save to image_cache
        cache_dir = Path("./image_cache")
        cache_dir.mkdir(exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"uploaded_{profile_id}_{timestamp}.{ext}"
        dest = cache_dir / filename
        dest.write_bytes(photo_bytes)

        local_path = f"./image_cache/{filename}"
        photo_url = f"/api/local-image/{local_path}"
        photo_hash = hashlib.sha256(photo_bytes).hexdigest()[:16]

        # Update profile
        profile_data = {}
        try:
            profile_data = json.loads(entity.domains or "{}")
        except (json.JSONDecodeError, TypeError):
            pass

        profile_data["photo_url"] = photo_url
        profile_data["photo_source"] = PhotoSource.UPLOADED
        profile_data["photo_status"] = PhotoStatus.RESOLVED
        profile_data["photo_uploaded_at"] = datetime.utcnow().isoformat()
        profile_data["photo_hash"] = photo_hash

        entity.domains = json.dumps(profile_data)
        session.commit()

        logger.info(
            "Photo uploaded for profile %d: %s (%d bytes)",
            profile_id, filename, len(photo_bytes),
        )

        return {
            "status": "ok",
            "photo_url": photo_url,
            "photo_source": "uploaded",
            "file_size": len(photo_bytes),
            "format": ext,
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Photo upload failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Photo upload failed")
    finally:
        session.close()


@app.get("/api/local-image/{file_path:path}")
async def serve_local_image(file_path: str):
    """Serve a locally cached image (e.g., LinkedIn PDF crop).

    Only serves files from the image_cache directory for security.
    """
    from fastapi.responses import Response

    safe_path = Path(file_path)
    # Security: reject path traversal
    if ".." in str(safe_path):
        raise HTTPException(status_code=403, detail="Invalid path")

    if not safe_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Only serve from allowed directories
    allowed_prefixes = [str(Path("./image_cache").resolve())]
    resolved = str(safe_path.resolve())
    if not any(resolved.startswith(p) for p in allowed_prefixes):
        raise HTTPException(status_code=403, detail="Access denied")

    content = safe_path.read_bytes()
    ct = "image/jpeg"
    if file_path.endswith(".png"):
        ct = "image/png"
    elif file_path.endswith(".webp"):
        ct = "image/webp"
    return Response(content=content, media_type=ct)


@app.get("/debug/photos")
def debug_photos():
    """Debug endpoint: photo resolution status across all contacts."""
    from app.services.photo_resolution import PhotoResolutionService
    profiles = get_all_profiles()
    service = PhotoResolutionService()
    stats = service.get_debug_stats(profiles)
    # Add PDF crop stats
    pdf_crops = sum(
        1 for p in profiles
        if p.get("photo_source") == "linkedin_pdf_crop"
    )
    crop_attempts = sum(1 for p in profiles if p.get("linkedin_pdf_path"))
    stats["linkedin_pdf_crops"] = pdf_crops
    stats["linkedin_pdf_uploads"] = crop_attempts
    stats["crop_success_rate"] = (
        f"{pdf_crops}/{crop_attempts}" if crop_attempts else "0/0"
    )
    return stats


@app.get("/debug/calendar")
def debug_calendar():
    """Debug endpoint: calendar integration status across all contacts."""
    profiles = get_all_profiles()
    events_count = 0
    matched = 0
    stubs = 0
    unmatched = []

    for p in profiles:
        upcoming = p.get("upcoming_meetings", [])
        events_count += len(upcoming)
        for m in upcoming:
            if m.get("match_reason") == "new_stub":
                stubs += 1
            elif m.get("match_reason"):
                matched += 1

        if p.get("source") == "calendar_ingest" and p.get("research_status") == "QUEUED":
            unmatched.append({
                "name": p.get("name"),
                "email": p.get("email"),
                "reason": "created_as_stub",
            })

    return {
        "events_attached": events_count,
        "matched_contacts": matched,
        "created_stubs": stubs,
        "unmatched_attendees": unmatched,
        "calendar_configured": bool(
            settings.google_client_id and settings.google_refresh_token
        ),
    }


@app.post(
    "/profiles/{profile_id}/enrich",
    dependencies=[Depends(verify_api_key)],
)
async def enrich_profile(profile_id: int):
    """Enrich a contact using People Data Labs (PDL).

    Calls PDL's person/enrich endpoint with the best available identifier
    (email > linkedin_url > name+company > name+location).

    If photo_url is returned, downloads it server-side and stores locally.
    NEVER wipes an existing RESOLVED photo unless download succeeds.

    Returns: success, fields_updated, photo_updated, match_confidence.
    """
    from app.services.enrichment_service import enrich_contact

    if not settings.pdl_enabled:
        raise HTTPException(
            status_code=400,
            detail="PDL enrichment is not enabled. Set PDL_ENABLED=true.",
        )
    if not settings.pdl_api_key:
        raise HTTPException(
            status_code=400,
            detail="PDL_API_KEY is not configured.",
        )

    session = get_session()
    try:
        entity = session.query(EntityRecord).filter(
            EntityRecord.id == profile_id,
        ).first()
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")

        result = await enrich_contact(
            profile_data=profile_data,
            contact_id=profile_id,
            contact_name=entity.name,
            entity=entity,
        )

        if result["success"]:
            entity.domains = json.dumps(profile_data)
            session.commit()

        return result

    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("PDL enrichment failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Enrichment failed")
    finally:
        session.close()


@app.get("/debug/enrichment")
def debug_enrichment():
    """Debug endpoint: PDL enrichment status with per-contact detail.

    Shows:
    - Global PDL status and rate limiter state
    - Per-contact enrichment records from the database (canonical columns)
    - Recent in-memory enrichment attempts with identifiers used
    """
    from app.clients.pdl_client import get_enrichment_log, get_rate_limiter

    log = get_enrichment_log()
    limiter = get_rate_limiter()

    total_enriched = sum(1 for e in log if e.get("status") == "success")
    total_failed = sum(1 for e in log if e.get("status") != "success")

    # Per-contact enrichment from database
    per_contact: list[dict] = []
    session = get_session()
    try:
        enriched_entities = session.query(EntityRecord).filter(
            EntityRecord.enriched_at.isnot(None),
        ).order_by(EntityRecord.enriched_at.desc()).limit(50).all()

        for ent in enriched_entities:
            profile_data = json.loads(ent.domains or "{}")
            per_contact.append({
                "contact_id": ent.id,
                "name": ent.name,
                "canonical_company": ent.canonical_company,
                "canonical_title": ent.canonical_title,
                "canonical_location": ent.canonical_location,
                "pdl_person_id": ent.pdl_person_id,
                "pdl_match_confidence": ent.pdl_match_confidence,
                "enriched_at": ent.enriched_at.isoformat() if ent.enriched_at else None,
                "request_identifier_used": profile_data.get(
                    "enrichment_request_identifier", "unknown"
                ),
                "fields_in_json": {
                    "company": profile_data.get("company", ""),
                    "title": profile_data.get("title", ""),
                    "location": profile_data.get("location", ""),
                    "linkedin_url": profile_data.get("linkedin_url", ""),
                },
                "persisted_to_columns": bool(ent.pdl_person_id),
            })
    finally:
        session.close()

    return {
        "pdl_enabled": settings.pdl_enabled,
        "pdl_configured": bool(settings.pdl_api_key),
        "total_enriched": total_enriched,
        "total_failed": total_failed,
        "enriched_contacts": per_contact,
        "last_10_attempts": log[-10:],
        "rate_limiter_state": limiter.state,
    }


@app.get("/debug/serp")
def debug_serp():
    """Debug endpoint: SerpAPI configuration and recent deep-research results.

    Shows:
    - SerpAPI configuration status
    - Per-contact deep research status (from profile data)
    - Retrieval ledger row counts
    - Visibility sweep execution status
    """
    serpapi_configured = bool(settings.serpapi_api_key)

    per_contact: list[dict] = []
    session = get_session()
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
        ).all()

        for ent in entities:
            profile_data = json.loads(ent.domains or "{}")
            dr_status = profile_data.get("deep_research_status", "NOT_STARTED")
            if dr_status == "NOT_STARTED" and not profile_data.get("retrieval_ledger"):
                continue  # Skip contacts with no research activity

            per_contact.append({
                "contact_id": ent.id,
                "name": ent.name,
                "deep_research_status": dr_status,
                "entity_lock_score": profile_data.get("entity_lock_score"),
                "retrieval_ledger_rows": len(profile_data.get("retrieval_ledger", [])),
                "visibility_ledger_rows": len(profile_data.get("visibility_ledger", [])),
                "visibility_sweep_executed": (
                    profile_data.get("visibility_report", {}).get("sweep_executed", False)
                ),
                "evidence_nodes": len(profile_data.get("evidence_nodes", [])),
                "generated_at": profile_data.get("deep_profile_generated_at"),
                "fail_closed_gates_passed": (
                    profile_data.get("fail_closed_status", {}).get("gates_passed")
                ),
            })
    finally:
        session.close()

    return {
        "serpapi_configured": serpapi_configured,
        "openai_configured": bool(settings.openai_api_key),
        "contacts_with_research": per_contact,
    }


@app.post("/profiles/{profile_id}/deep-research", dependencies=[Depends(verify_api_key)])
async def deep_research_endpoint(profile_id: int):
    """Execute full Deep Research pipeline for a contact.

    This is the primary deep research endpoint that:
    1. Auto-runs PDL enrichment if not recently enriched
    2. Executes SerpAPI visibility sweep (MANDATORY, fail-closed)
    3. Saves RetrievalLedger rows (every query logged)
    4. Builds EvidenceNodes from web results
    5. Recomputes Entity Lock using PDL + web evidence
    6. Synthesizes dossier (Mode B) via LLM
    7. Runs QA gates and saves results

    Fail-closed rules:
    - If SerpAPI key missing or call fails: returns failure report with
      preserved ledger rows and error reason
    - Does NOT claim sweep executed if it didn't
    - Persists partial results even on failure
    """
    from app.services.enrichment_service import enrich_contact

    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")

        # --- STEP 0: Auto-enrich via PDL if not recently enriched ---
        pdl_ran = False
        enriched_at = profile_data.get("enriched_at", "")
        needs_enrichment = not enriched_at
        if enriched_at:
            try:
                from datetime import timedelta
                last_enriched = datetime.fromisoformat(enriched_at)
                if datetime.utcnow() - last_enriched > timedelta(days=7):
                    needs_enrichment = True
            except (ValueError, TypeError):
                needs_enrichment = True

        if needs_enrichment and settings.pdl_enabled and settings.pdl_api_key:
            try:
                enrich_result = await enrich_contact(
                    profile_data=profile_data,
                    contact_id=profile_id,
                    contact_name=entity.name,
                    entity=entity,
                )
                pdl_ran = True
                # Save enrichment even if deep research fails later
                entity.domains = json.dumps(profile_data)
                session.commit()
                logger.info(
                    "Auto-enriched profile %d before deep research: %s",
                    profile_id, enrich_result.get("fields_updated", []),
                )
            except Exception:
                logger.exception("Auto-enrichment failed for profile %d", profile_id)

        # Read current profile data (may have been updated by enrichment)
        profile_data = json.loads(entity.domains or "{}")

        p_name = entity.name
        p_company = (
            entity.canonical_company
            or profile_data.get("company", "")
        )
        p_title = (
            entity.canonical_title
            or profile_data.get("title", "")
        )
        p_linkedin = profile_data.get("linkedin_url", "")
        p_location = (
            entity.canonical_location
            or profile_data.get("location", "")
        )

        # Mark as RUNNING
        profile_data["deep_research_status"] = DossierMode.RUNNING
        entity.domains = json.dumps(profile_data)
        session.commit()

        # --- STEP 1: Generate auditable search plan ---
        search_plan = generate_search_plan(
            name=p_name,
            company=p_company,
            title=p_title,
            linkedin_url=p_linkedin,
            location=p_location,
        )

        interactions_summary = build_interactions_summary(profile_data)

        # --- Initialize Evidence Graph ---
        graph = EvidenceGraph()

        # Add PDF evidence nodes if LinkedIn PDF was uploaded
        if profile_data.get("linkedin_pdf_raw_text"):
            try:
                from app.services.linkedin_pdf import (
                    LinkedInPDFTextResult,
                    build_evidence_nodes_from_pdf,
                )
                text_result = LinkedInPDFTextResult(
                    raw_text=profile_data.get("linkedin_pdf_raw_text", ""),
                    headline=profile_data.get("headline", ""),
                    location=profile_data.get("location", ""),
                    about=profile_data.get("linkedin_pdf_sections", {}).get("about", ""),
                    experience=profile_data.get("linkedin_pdf_experience", []),
                    education=profile_data.get("linkedin_pdf_education", []),
                    skills=profile_data.get("linkedin_pdf_skills", []),
                    sections=profile_data.get("linkedin_pdf_sections", {}),
                )
                pdf_nodes = build_evidence_nodes_from_pdf(text_result, contact_name=p_name)
                for node_data in pdf_nodes:
                    graph.add_pdf_node(
                        source=node_data["source"],
                        snippet=node_data["snippet"],
                        date=node_data.get("date", "UNKNOWN"),
                        ref=node_data.get("ref", ""),
                    )
                logger.info("Added %d PDF evidence nodes for profile %d", len(pdf_nodes), profile_id)
            except Exception:
                logger.exception("PDF evidence extraction failed for profile %d", profile_id)

        # Add meeting evidence nodes
        for interaction in profile_data.get("interactions", [])[:15]:
            graph.add_meeting_node(
                source=interaction.get("title", "meeting"),
                snippet=interaction.get("summary", "")[:200] or interaction.get("title", ""),
                date=interaction.get("date", "UNKNOWN"),
                ref=interaction.get("type", "meeting"),
            )

        # --- STEP 2: Execute SerpAPI retrieval (with ledger logging) ---
        web_research = ""
        visibility_research = ""
        search_results: dict = {}
        visibility_results: dict = {}
        visibility_sweep_executed = False
        serp_error = ""

        if settings.serpapi_api_key:
            try:
                serp = SerpAPIClient()

                # Person search with ledger
                search_results = await serp.search_person_with_ledger(
                    name=p_name,
                    company=p_company,
                    title=p_title,
                    linkedin_url=p_linkedin,
                    graph=graph,
                )
                web_research = format_web_results_for_prompt(search_results)
                logger.info(
                    "Web search for '%s' returned %d results (ledger: %d rows)",
                    p_name,
                    sum(len(v) for v in search_results.values()),
                    len(graph.ledger),
                )
            except Exception as exc:
                serp_error = f"Web search failed: {exc}"
                logger.exception("Web search failed for '%s'", p_name)

            # --- Visibility Sweep (MANDATORY, fail-closed) ---
            try:
                serp = SerpAPIClient()
                visibility_results = await serp.search_visibility_sweep_with_ledger(
                    name=p_name,
                    company=p_company,
                    graph=graph,
                )
                visibility_research = format_visibility_results_for_prompt(
                    visibility_results
                )
                visibility_sweep_executed = True
                logger.info(
                    "Visibility sweep for '%s': %d results, %d ledger rows",
                    p_name,
                    sum(len(v) for v in visibility_results.values()),
                    len(graph.get_visibility_ledger_rows()),
                )
            except Exception as exc:
                serp_error += f" Visibility sweep failed: {exc}"
                logger.exception("Visibility sweep failed for '%s'", p_name)
        else:
            serp_error = "SerpAPI key not configured (SERPAPI_API_KEY not set)"
            # Log the error to the ledger so we have a record
            graph.log_retrieval(
                query="[SERPAPI_UNAVAILABLE]",
                intent="visibility",
                results=[],
            )

        # --- STEP 3: Entity Lock (with PDL data) ---
        # Build PDL data dict from canonical columns/profile_data
        pdl_data = {
            "canonical_company": (
                entity.canonical_company
                or profile_data.get("canonical_company", "")
            ),
            "canonical_title": (
                entity.canonical_title
                or profile_data.get("canonical_title", "")
            ),
            "canonical_location": (
                entity.canonical_location
                or profile_data.get("canonical_location", "")
            ),
            "pdl_match_confidence": (
                entity.pdl_match_confidence
                or profile_data.get("pdl_match_confidence", 0)
            ),
        }

        apollo_data = {}
        if profile_data.get("apollo_raw"):
            apollo_data = profile_data["apollo_raw"]

        # Build PDF data dict from uploaded LinkedIn PDF
        pdf_experience = profile_data.get("linkedin_pdf_experience", [])
        pdf_company_from_exp = ""
        pdf_title_from_exp = ""
        if pdf_experience:
            # First experience entry is the most recent/current role
            first_exp = pdf_experience[0] if pdf_experience else {}
            pdf_company_from_exp = first_exp.get("company", "")
            pdf_title_from_exp = first_exp.get("title", "")

        linkedin_pdf_data = {
            "company": pdf_company_from_exp,
            "title": pdf_title_from_exp,
            "headline": profile_data.get("headline", ""),
            "location": profile_data.get("location", ""),
            "text_usable": profile_data.get("linkedin_pdf_text_usable", False),
        }

        entity_lock = score_disambiguation(
            name=p_name,
            company=p_company,
            title=p_title,
            linkedin_url=p_linkedin,
            location=p_location,
            search_results=search_results,
            apollo_data=apollo_data,
            has_meeting_data=bool(profile_data.get("interactions")),
            pdl_data=pdl_data,
            pdf_data=linkedin_pdf_data,
        )

        entity_lock_report = {
            "canonical_name": p_name,
            "confirmed_employer": p_company or None,
            "confirmed_title": p_title or None,
            "location": p_location or None,
            "linkedin_url": p_linkedin or None,
            "entity_lock_score": entity_lock.score,
            "lock_status": entity_lock.lock_status,
            "is_locked": entity_lock.is_locked,
            "pdl_enriched": bool(entity.pdl_person_id or profile_data.get("pdl_person_id")),
            "pdl_confidence": pdl_data.get("pdl_match_confidence"),
            "pdl_canonical": {
                "company": pdl_data.get("canonical_company"),
                "title": pdl_data.get("canonical_title"),
                "location": pdl_data.get("canonical_location"),
            },
            "disambiguation_risks": (
                [] if entity_lock.is_locked
                else [f"IDENTITY {entity_lock.lock_status} — review evidence"]
            ),
            "evidence": entity_lock.evidence,
            "signals": {
                "name_match": entity_lock.name_match,
                "company_match": entity_lock.company_match,
                "title_match": entity_lock.title_match,
                "linkedin_url_present": entity_lock.linkedin_url_present,
                "linkedin_confirmed": entity_lock.linkedin_confirmed,
                "linkedin_verified_by_retrieval": entity_lock.linkedin_verified_by_retrieval,
                "location_match": entity_lock.location_match,
                "photo_available": entity_lock.photo_available,
                "multiple_sources_agree": entity_lock.multiple_sources_agree,
                "employer_match": entity_lock.employer_match,
                "meeting_confirmed": entity_lock.meeting_confirmed,
                "secondary_source_match": entity_lock.secondary_source_match,
            },
        }

        # --- PRE-SYNTHESIS GATE ---
        has_public_results = any(len(v) > 0 for v in search_results.values())
        vis_coverage_confidence = compute_visibility_coverage_confidence(graph)

        dossier_mode, mode_reason = determine_dossier_mode(
            entity_lock_score=entity_lock.score,
            visibility_executed=visibility_sweep_executed,
            has_public_results=has_public_results,
            person_name=p_name,
        )

        logger.info(
            "Pre-synthesis gate for '%s': mode=%s, entity_lock=%d, "
            "visibility_executed=%s, serp_error='%s'",
            p_name, dossier_mode, entity_lock.score,
            visibility_sweep_executed, serp_error[:100],
        )

        # --- Handle HALTED mode ---
        if dossier_mode == DossierMode.HALTED:
            failure_report = build_failure_report(
                mode_reason=mode_reason,
                entity_lock_score=entity_lock.score,
                visibility_confidence=vis_coverage_confidence,
                graph=graph,
                person_name=p_name,
            )
            if serp_error:
                failure_report += f"\n\n--- SERP ERROR ---\n{serp_error}"

            fail_closed_status = {
                "gates_passed": False,
                "dossier_mode": dossier_mode,
                "mode_reason": mode_reason,
                "serp_error": serp_error,
                "visibility_ledger_rows": len(graph.get_visibility_ledger_rows()),
                "visibility_confidence": vis_coverage_confidence,
                "entity_lock_score": entity_lock.score,
                "entity_lock_status": entity_lock.lock_status,
                "has_public_results": has_public_results,
                "failure_message": failure_report,
            }

            generated_at = datetime.utcnow().isoformat()
            profile_data["deep_profile"] = failure_report
            profile_data["dossier_mode_b_markdown"] = failure_report
            profile_data["deep_profile_generated_at"] = generated_at
            profile_data["deep_research_status"] = DossierMode.FAILED
            profile_data["entity_lock_score"] = entity_lock.score
            profile_data["entity_lock_report"] = entity_lock_report
            profile_data["search_plan"] = search_plan
            profile_data["fail_closed_status"] = fail_closed_status
            profile_data["evidence_graph"] = graph.to_dict()
            profile_data["retrieval_ledger"] = [r.model_dump() for r in graph.ledger]
            profile_data["visibility_ledger"] = [
                r.model_dump() for r in graph.get_visibility_ledger_rows()
            ]
            entity.domains = json.dumps(profile_data)
            session.commit()

            return {
                "status": "halted",
                "mode": DossierMode.DEEP_RESEARCH,
                "dossier_mode": dossier_mode,
                "deep_research_status": DossierMode.FAILED,
                "deep_profile": failure_report,
                "generated_at": generated_at,
                "entity_lock": entity_lock_report,
                "qa_report": None,
                "search_plan": search_plan,
                "visibility_report": {
                    "sweep_executed": visibility_sweep_executed,
                    "serp_error": serp_error,
                },
                "fail_closed_status": fail_closed_status,
                "evidence_graph": graph.to_dict(),
                "retrieval_ledger_count": len(graph.ledger),
                "visibility_ledger_count": len(graph.get_visibility_ledger_rows()),
                "pdl_enriched": pdl_ran,
            }

        # --- STEP 4: Generate dossier via LLM ---
        if not settings.openai_api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key not configured")

        pdf_context = ""
        if profile_data.get("linkedin_pdf_raw_text"):
            pdf_text = profile_data["linkedin_pdf_raw_text"][:10000]
            pdf_context = (
                f"\n\n## USER-SUPPLIED LINKEDIN PDF (artifact evidence)\n"
                f"Tag claims from this source as [VERIFIED-PDF].\n\n{pdf_text}"
            )

        # Compute adaptive evidence threshold for the LLM prompt
        total_web_results = sum(len(v) for v in search_results.values())
        if total_web_results >= 10:
            evidence_threshold = 85
        elif total_web_results >= 5:
            evidence_threshold = 70
        else:
            evidence_threshold = 60

        result = generate_deep_profile(
            name=p_name,
            title=p_title,
            company=p_company,
            linkedin_url=p_linkedin,
            location=p_location,
            industry=profile_data.get("company_industry", ""),
            company_size=profile_data.get("company_size"),
            interactions_summary=interactions_summary + pdf_context,
            web_research=web_research,
            visibility_research=visibility_research,
            evidence_threshold=evidence_threshold,
            identity_lock_score=entity_lock.score,
        )

        # --- STEP 5: Post-synthesis QA gates ---
        visibility_categories_searched = [
            cat for cat in VISIBILITY_CATEGORIES
            if visibility_results.get(cat) is not None
        ]
        qa_report = generate_dossier_qa_report(
            dossier_text=result,
            disambiguation=entity_lock,
            person_name=p_name,
            visibility_categories=visibility_categories_searched,
            visibility_sweep_executed=visibility_sweep_executed,
        )
        qa_markdown = render_qa_report_markdown(qa_report)

        # --- STEP 6: Post-synthesis fail-closed enforcement ---
        visibility_ledger_count = len(graph.get_visibility_ledger_rows())
        evidence_coverage = qa_report.evidence_coverage.coverage_pct
        total_web_results = sum(len(v) for v in search_results.values())
        should_output, fail_message = enforce_fail_closed_gates(
            dossier_text=result,
            entity_lock_score=entity_lock.score,
            visibility_ledger_count=visibility_ledger_count,
            evidence_coverage_pct=evidence_coverage,
            person_name=p_name,
            has_public_results=has_public_results,
            web_results_count=total_web_results,
        )

        # Apply mode-based prose filtering
        if should_output:
            result = filter_prose_by_mode(result, dossier_mode, entity_lock.score)

        # Build visibility report
        highest_signal = extract_highest_signal_artifacts(graph, max_artifacts=3)
        visibility_report = {
            "sweep_executed": visibility_sweep_executed,
            "categories_searched": visibility_categories_searched,
            "total_results": sum(len(v) for v in visibility_results.values()),
            "results_by_category": {
                cat: len(visibility_results.get(cat, []))
                for cat in VISIBILITY_CATEGORIES
            },
            "highest_signal_artifacts": highest_signal,
            "coverage_confidence": vis_coverage_confidence,
        }

        fail_closed_status = {
            "gates_passed": should_output,
            "dossier_mode": dossier_mode,
            "mode_reason": mode_reason,
            "visibility_ledger_rows": visibility_ledger_count,
            "visibility_confidence": vis_coverage_confidence,
            "evidence_coverage_pct": round(evidence_coverage, 1),
            "entity_lock_score": entity_lock.score,
            "entity_lock_status": entity_lock.lock_status,
            "has_public_results": has_public_results,
            "failure_message": fail_message if not should_output else None,
        }

        # --- Persist ---
        generated_at = datetime.utcnow().isoformat()
        deep_research_status = (
            DossierMode.SUCCEEDED if should_output else DossierMode.FAILED
        )
        profile_data["deep_profile"] = result if should_output else fail_message
        profile_data["dossier_mode_b_markdown"] = (
            result if should_output else fail_message
        )
        profile_data["deep_profile_generated_at"] = generated_at
        profile_data["deep_research_status"] = deep_research_status
        profile_data["entity_lock_score"] = entity_lock.score
        profile_data["entity_lock_report"] = entity_lock_report
        profile_data["qa_genericness_score"] = qa_report.genericness.genericness_score
        profile_data["qa_evidence_coverage_pct"] = round(
            qa_report.evidence_coverage.coverage_pct, 1,
        )
        profile_data["qa_passes_all"] = qa_report.passes_all
        profile_data["qa_report_markdown"] = qa_markdown
        profile_data["search_plan"] = search_plan
        profile_data["visibility_report"] = visibility_report
        profile_data["fail_closed_status"] = fail_closed_status
        profile_data["evidence_graph"] = graph.to_dict()
        profile_data["retrieval_ledger"] = [r.model_dump() for r in graph.ledger]
        profile_data["visibility_ledger"] = [
            r.model_dump() for r in graph.get_visibility_ledger_rows()
        ]
        profile_data["evidence_nodes"] = [
            n.model_dump() for n in graph.nodes.values()
        ]
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok" if should_output else "halted",
            "mode": DossierMode.DEEP_RESEARCH,
            "dossier_mode": dossier_mode,
            "deep_research_status": deep_research_status,
            "deep_profile": result if should_output else fail_message,
            "generated_at": generated_at,
            "entity_lock": entity_lock_report,
            "qa_report": {
                "passes_all": qa_report.passes_all,
                "genericness_score": qa_report.genericness.genericness_score,
                "evidence_coverage_pct": round(
                    qa_report.evidence_coverage.coverage_pct, 1,
                ),
                "person_level_pct": round(qa_report.person_level.person_pct, 1),
                "contradictions": len(qa_report.contradictions),
                "hallucination_risk_flags": qa_report.hallucination_risk_flags,
                "markdown": qa_markdown,
            },
            "search_plan": search_plan,
            "visibility_report": visibility_report,
            "fail_closed_status": fail_closed_status,
            "evidence_graph": graph.to_dict(),
            "retrieval_ledger_count": len(graph.ledger),
            "visibility_ledger_count": len(graph.get_visibility_ledger_rows()),
            "pdl_enriched": pdl_ran or bool(entity.pdl_person_id),
        }
    except HTTPException:
        raise
    except Exception:
        # Mark as FAILED on unexpected errors
        try:
            profile_data = json.loads(entity.domains or "{}")
            profile_data["deep_research_status"] = DossierMode.FAILED
            entity.domains = json.dumps(profile_data)
            session.commit()
        except Exception:
            session.rollback()
        logger.exception("Deep research failed for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Deep research failed")
    finally:
        session.close()


@app.get("/profiles", dependencies=[Depends(verify_api_key)])
def list_profiles():
    """Return all auto-generated contact profiles."""
    return get_all_profiles()


@app.post("/profiles/refresh-photos", dependencies=[Depends(verify_api_key)])
async def refresh_photos():
    """Re-enrich confirmed profiles that are missing photos.

    Targets verified contacts whose photo_url is empty and tries Apollo
    enrichment using their stored linkedin_url for a precise match.
    """
    if not settings.apollo_api_key:
        raise HTTPException(status_code=400, detail="Apollo API key not configured")

    updated = await _re_enrich_confirmed_profiles()
    return {"status": "ok", "profiles_updated": updated}


async def _enrich_confirmed_profile(
    entity: EntityRecord, profile_data: dict, session
) -> bool:
    """Try to enrich a confirmed profile via Apollo using its LinkedIn URL.

    Called after confirm or manual-set to get photo, title, and other data
    that Apollo's search endpoint may not have returned.
    Returns True if new data was written.
    """
    linkedin_url = profile_data.get("linkedin_url", "")
    if not linkedin_url or not settings.apollo_api_key:
        return False

    try:
        client = ApolloClient()
        emails = json.loads(entity.emails or "[]")
        parts = entity.name.split(None, 1) if entity.name else []
        first = parts[0] if parts else None
        last = parts[1] if len(parts) > 1 else None

        person = await client.enrich_person(
            email=emails[0] if emails else None,
            first_name=first,
            last_name=last,
            linkedin_url=linkedin_url,
            organization_name=profile_data.get("company"),
        )
        if not person:
            return False

        enrichment = normalize_enrichment(person)
        if not enrichment:
            return False

        updated = False
        # Fill in missing fields from enrichment — never overwrite user-confirmed data
        for key in (
            "photo_url", "title", "headline", "seniority",
            "company_industry", "company_size", "company_linkedin",
        ):
            if enrichment.get(key) and not profile_data.get(key):
                profile_data[key] = enrichment[key]
                updated = True

        # Location is a composite field
        if not profile_data.get("location"):
            loc = ", ".join(filter(None, [
                person.get("city", ""),
                person.get("state", ""),
            ]))
            if loc:
                profile_data["location"] = loc
                updated = True

        if not profile_data.get("company"):
            org = person.get("organization") or {}
            if org.get("name"):
                profile_data["company"] = org["name"]
                updated = True

        if updated:
            entity.domains = json.dumps(profile_data)
            session.commit()
            logger.info(
                "Enriched confirmed profile %s (id=%d) via Apollo",
                entity.name, entity.id,
            )
        return updated
    except Exception:
        logger.exception("Post-confirm enrichment failed for %s", entity.name)
        return False


class ConfirmLinkedInRequest(BaseModel):
    candidate_index: int = Field(
        ..., ge=-1,
        description="Index into linkedin_candidates, or -1 for 'none of the above'",
    )


@app.get("/profiles/pending-review", dependencies=[Depends(verify_api_key)])
def profiles_pending_review():
    """Return profiles that need LinkedIn disambiguation.

    Includes both pending_review (have candidates) and no_match (need manual search).
    Also includes contacts without a LinkedIn PDF uploaded, since uploading a PDF
    dramatically improves research quality (evidence coverage, entity lock score).
    """
    all_p = get_all_profiles()

    linkedin_pending = []
    needs_pdf = []

    for p in all_p:
        status = p.get("linkedin_status", "")
        has_pdf = bool(p.get("linkedin_pdf_path"))

        # LinkedIn disambiguation contacts
        if status in ("pending_review", "no_match"):
            linkedin_pending.append(p)
        # Contacts that could benefit from a PDF upload:
        # - No PDF uploaded yet, AND
        # - Has at least 1 meeting (real contact, not just an email)
        elif not has_pdf and p.get("meeting_count", 0) >= 1:
            needs_pdf.append(p)

    return {
        "linkedin_pending": linkedin_pending,
        "needs_pdf": needs_pdf,
    }


@app.post("/profiles/{profile_id}/confirm-linkedin", dependencies=[Depends(verify_api_key)])
async def confirm_linkedin(profile_id: int, request: ConfirmLinkedInRequest):
    """Confirm or reject a LinkedIn candidate for a profile.

    candidate_index = -1 means 'none of the above' (mark as no_match).
    After confirmation, tries Apollo enrichment to fill in missing photo/title.
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

        # If the candidate had no photo, try Apollo enrichment using LinkedIn URL
        if not profile_data.get("photo_url") and profile_data.get("linkedin_url"):
            await _enrich_confirmed_profile(entity, profile_data, session)

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
async def set_linkedin(profile_id: int, request: SetLinkedInRequest):
    """Manually set a LinkedIn URL for a profile.

    After saving, tries Apollo enrichment using the LinkedIn URL to pull
    photo, title, and other data.
    """
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

        # Try to enrich with photo/title from Apollo using the LinkedIn URL
        await _enrich_confirmed_profile(entity, profile_data, session)

        return {"status": "ok", "linkedin_status": "confirmed"}
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Failed to set LinkedIn for profile %d", profile_id)
        raise HTTPException(status_code=500, detail="Failed to update profile")
    finally:
        session.close()


@app.post("/profiles/{profile_id}/meeting-prep", dependencies=[Depends(verify_api_key)])
async def generate_meeting_prep(profile_id: int):
    """Generate a Mode A Meeting-Prep Brief for any contact.

    Mode A: fast, always available, no web required.
    Uses only internal evidence (meetings, emails, stored profile data).
    NO SerpAPI, NO visibility sweep, NO fail-closed gating.

    Tags: [VERIFIED-MEETING], [INFERRED-L/M], [UNKNOWN]
    """
    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")
        p_name = entity.name

        # Build evidence graph from internal data only
        graph = EvidenceGraph()
        for interaction in profile_data.get("interactions", [])[:15]:
            graph.add_meeting_node(
                source=interaction.get("title", "meeting"),
                snippet=(
                    interaction.get("summary", "")[:200]
                    or interaction.get("title", "")
                ),
                date=interaction.get("date", "UNKNOWN"),
                ref=interaction.get("type", "meeting"),
            )

        # Generate the meeting-prep brief (no LLM needed)
        brief_markdown = build_meeting_prep_brief(
            person_name=p_name,
            graph=graph,
            profile_data=profile_data,
        )

        # Persist
        generated_at = datetime.utcnow().isoformat()
        profile_data["dossier_mode_a_markdown"] = brief_markdown
        profile_data["dossier_mode_a_generated_at"] = generated_at
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok",
            "mode": DossierMode.MEETING_PREP,
            "brief": brief_markdown,
            "generated_at": generated_at,
            "evidence_nodes": len(graph.nodes),
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Meeting-prep brief failed for profile %d", profile_id)
        raise HTTPException(
            status_code=500, detail="Meeting-prep brief generation failed"
        )
    finally:
        session.close()


@app.post(
    "/profiles/{profile_id}/artifact-dossier",
    dependencies=[Depends(verify_api_key)],
)
async def generate_artifact_dossier_endpoint(profile_id: int):
    """Generate an artifact-first "Public Statements & Positions" dossier.

    Path B1: Uses PDF text + meeting evidence only. No SerpAPI required.
    Works even without a confirmed LinkedIn profile.

    Requires a LinkedIn PDF to have been uploaded first via /ingest-linkedin-pdf.
    If OpenAI API key is configured, uses LLM for enhanced synthesis.
    Otherwise falls back to template-based dossier.
    """
    from app.services.artifact_dossier import run_artifact_dossier_pipeline

    session = get_session()
    try:
        entity = session.query(EntityRecord).get(profile_id)
        if not entity:
            raise HTTPException(status_code=404, detail="Profile not found")

        profile_data = json.loads(entity.domains or "{}")
        p_name = entity.name

        # Check if PDF has been uploaded
        if not profile_data.get("linkedin_pdf_raw_text"):
            raise HTTPException(
                status_code=422,
                detail="No LinkedIn PDF uploaded. Upload a PDF first via "
                       "/profiles/{profile_id}/ingest-linkedin-pdf",
            )

        # Run the artifact-first pipeline
        use_llm = bool(settings.openai_api_key)
        result = run_artifact_dossier_pipeline(
            profile_data=profile_data,
            person_name=p_name,
            use_llm=use_llm,
        )

        # Persist
        profile_data["artifact_dossier_markdown"] = result["dossier_markdown"]
        profile_data["artifact_dossier_generated_at"] = result["generated_at"]
        profile_data["artifact_dossier_mode"] = result["mode"]
        profile_data["artifact_dossier_coverage"] = result["coverage_pct"]
        profile_data["artifact_evidence_graph"] = result["evidence_graph"]
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok",
            "mode": result["mode"],
            "dossier": result["dossier_markdown"],
            "generated_at": result["generated_at"],
            "artifact_count": result["artifact_count"],
            "meeting_count": result["meeting_count"],
            "total_evidence_nodes": result["total_evidence_nodes"],
            "coverage_pct": result["coverage_pct"],
            "passes_coverage": result["passes_coverage"],
            "evidence_graph": result["evidence_graph"],
        }
    except HTTPException:
        raise
    except Exception:
        session.rollback()
        logger.exception("Artifact dossier failed for profile %d", profile_id)
        raise HTTPException(
            status_code=500, detail="Artifact dossier generation failed"
        )
    finally:
        session.close()


@app.post("/profiles/{profile_id}/deep-profile", dependencies=[Depends(verify_api_key)])
async def generate_profile_research(profile_id: int):
    """Generate a Mode B Deep Research Dossier for a verified contact.

    Dual-path pipeline:
    - If LinkedIn PDF uploaded: includes PDF EvidenceNodes (artifact-first enrichment)
    - Always: SerpAPI retrieval → LLM synthesis → QA gates

    Pipeline: Entity Lock → SerpAPI retrieval → LLM synthesis → QA gates.
    Returns the dossier, entity lock report, QA report, and search queries.
    Only works for profiles with linkedin_status == 'confirmed'.
    Fail-closed: halts if visibility sweep or entity lock fails.
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

        p_name = entity.name
        p_company = profile_data.get("company", "")
        p_title = profile_data.get("title", "")
        p_linkedin = profile_data.get("linkedin_url", "")
        p_location = profile_data.get("location", "")

        # --- STEP 0: Generate auditable search plan ---
        search_plan = generate_search_plan(
            name=p_name,
            company=p_company,
            title=p_title,
            linkedin_url=p_linkedin,
            location=p_location,
        )

        interactions_summary = build_interactions_summary(profile_data)

        # --- Initialize Evidence Graph ---
        graph = EvidenceGraph()

        # Add PDF evidence nodes if LinkedIn PDF was uploaded (dual-path)
        pdf_artifact_count = 0
        if profile_data.get("linkedin_pdf_raw_text"):
            from app.services.linkedin_pdf import (
                LinkedInPDFTextResult,
                build_evidence_nodes_from_pdf,
            )
            text_result = LinkedInPDFTextResult(
                raw_text=profile_data.get("linkedin_pdf_raw_text", ""),
                headline=profile_data.get("headline", ""),
                location=profile_data.get("location", ""),
                about=profile_data.get("linkedin_pdf_sections", {}).get("about", ""),
                experience=profile_data.get("linkedin_pdf_experience", []),
                education=profile_data.get("linkedin_pdf_education", []),
                skills=profile_data.get("linkedin_pdf_skills", []),
                sections=profile_data.get("linkedin_pdf_sections", {}),
            )
            pdf_nodes = build_evidence_nodes_from_pdf(text_result, contact_name=p_name)
            for node_data in pdf_nodes:
                graph.add_pdf_node(
                    source=node_data["source"],
                    snippet=node_data["snippet"],
                    date=node_data.get("date", "UNKNOWN"),
                    ref=node_data.get("ref", ""),
                )
            pdf_artifact_count = len(pdf_nodes)
            logger.info(
                "Added %d PDF evidence nodes for '%s' from LinkedIn PDF",
                pdf_artifact_count, p_name,
            )

        # Add meeting evidence nodes from interactions
        for interaction in profile_data.get("interactions", [])[:15]:
            graph.add_meeting_node(
                source=interaction.get("title", "meeting"),
                snippet=interaction.get("summary", "")[:200] or interaction.get("title", ""),
                date=interaction.get("date", "UNKNOWN"),
                ref=interaction.get("type", "meeting"),
            )

        # --- STEP 1: Fetch web data via SerpAPI (with ledger logging) ---
        web_research = ""
        visibility_research = ""
        search_results: dict = {}
        visibility_results: dict = {}
        visibility_sweep_executed = False
        if settings.serpapi_api_key:
            try:
                serp = SerpAPIClient()
                search_results = await serp.search_person_with_ledger(
                    name=p_name,
                    company=p_company,
                    title=p_title,
                    linkedin_url=p_linkedin,
                    graph=graph,
                )
                web_research = format_web_results_for_prompt(search_results)
                logger.info(
                    "Web search for '%s' returned %d total results (ledger: %d rows)",
                    p_name,
                    sum(len(v) for v in search_results.values()),
                    len(graph.ledger),
                )
            except Exception:
                logger.exception("Web search failed for '%s', proceeding without", p_name)

            # --- STEP 1b: Public Visibility Sweep (MANDATORY, fail-closed) ---
            try:
                if not serp.api_key:
                    serp = SerpAPIClient()
                visibility_results = await serp.search_visibility_sweep_with_ledger(
                    name=p_name,
                    company=p_company,
                    graph=graph,
                )
                visibility_research = format_visibility_results_for_prompt(
                    visibility_results
                )
                visibility_sweep_executed = True
                logger.info(
                    "Visibility sweep for '%s' returned %d total results (ledger: %d rows)",
                    p_name,
                    sum(len(v) for v in visibility_results.values()),
                    len(graph.get_visibility_ledger_rows()),
                )
            except Exception:
                logger.exception(
                    "Visibility sweep failed for '%s', proceeding without", p_name
                )

        # --- STEP 2: Entity Lock (disambiguation scoring) ---
        apollo_data = {}
        if profile_data.get("apollo_raw"):
            apollo_data = profile_data["apollo_raw"]
        elif p_title or profile_data.get("photo_url"):
            apollo_data = {"name": p_name, "title": p_title}
            if profile_data.get("photo_url"):
                apollo_data["photo_url"] = profile_data["photo_url"]

        # Build PDF data dict from uploaded LinkedIn PDF
        pdf_exp = profile_data.get("linkedin_pdf_experience", [])
        pdf_co = ""
        pdf_ti = ""
        if pdf_exp:
            first_exp = pdf_exp[0] if pdf_exp else {}
            pdf_co = first_exp.get("company", "")
            pdf_ti = first_exp.get("title", "")

        linkedin_pdf_data_2 = {
            "company": pdf_co,
            "title": pdf_ti,
            "headline": profile_data.get("headline", ""),
            "location": profile_data.get("location", ""),
            "text_usable": profile_data.get("linkedin_pdf_text_usable", False),
        }

        entity_lock = score_disambiguation(
            name=p_name,
            company=p_company,
            title=p_title,
            linkedin_url=p_linkedin,
            location=p_location,
            search_results=search_results,
            apollo_data=apollo_data,
            has_meeting_data=bool(profile_data.get("interactions")),
            pdf_data=linkedin_pdf_data_2,
        )

        entity_lock_report = {
            "canonical_name": p_name,
            "confirmed_employer": p_company or None,
            "confirmed_title": p_title or None,
            "location": p_location or None,
            "linkedin_url": p_linkedin or None,
            "entity_lock_score": entity_lock.score,
            "lock_status": entity_lock.lock_status,
            "is_locked": entity_lock.is_locked,
            "disambiguation_risks": (
                [] if entity_lock.is_locked
                else [f"IDENTITY {entity_lock.lock_status} — review evidence before acting"]
            ),
            "evidence": entity_lock.evidence,
            "signals": {
                "name_match": entity_lock.name_match,
                "company_match": entity_lock.company_match,
                "title_match": entity_lock.title_match,
                "linkedin_url_present": entity_lock.linkedin_url_present,
                "linkedin_confirmed": entity_lock.linkedin_confirmed,
                "linkedin_verified_by_retrieval": entity_lock.linkedin_verified_by_retrieval,
                "location_match": entity_lock.location_match,
                "photo_available": entity_lock.photo_available,
                "multiple_sources_agree": entity_lock.multiple_sources_agree,
                "employer_match": entity_lock.employer_match,
                "meeting_confirmed": entity_lock.meeting_confirmed,
                "secondary_source_match": entity_lock.secondary_source_match,
            },
        }

        # --- PRE-SYNTHESIS GATE: Determine output mode BEFORE calling LLM ---
        has_public_results = any(len(v) > 0 for v in search_results.values())
        vis_coverage_confidence = compute_visibility_coverage_confidence(graph)

        dossier_mode, mode_reason = determine_dossier_mode(
            entity_lock_score=entity_lock.score,
            visibility_executed=visibility_sweep_executed,
            has_public_results=has_public_results,
            person_name=p_name,
        )

        logger.info(
            "Pre-synthesis gate for '%s': mode=%s, entity_lock=%d, "
            "visibility_executed=%s, public_results=%s",
            p_name, dossier_mode, entity_lock.score,
            visibility_sweep_executed, has_public_results,
        )

        # If HALTED, do NOT call LLM — return failure report immediately
        if dossier_mode == DossierMode.HALTED:
            failure_report = build_failure_report(
                mode_reason=mode_reason,
                entity_lock_score=entity_lock.score,
                visibility_confidence=vis_coverage_confidence,
                graph=graph,
                person_name=p_name,
            )

            fail_closed_status = {
                "gates_passed": False,
                "dossier_mode": dossier_mode,
                "mode_reason": mode_reason,
                "visibility_ledger_rows": len(graph.get_visibility_ledger_rows()),
                "visibility_confidence": vis_coverage_confidence,
                "entity_lock_score": entity_lock.score,
                "entity_lock_status": entity_lock.lock_status,
                "has_public_results": has_public_results,
                "failure_message": failure_report,
            }

            generated_at = datetime.utcnow().isoformat()
            profile_data["deep_profile"] = failure_report
            profile_data["dossier_mode_b_markdown"] = failure_report
            profile_data["deep_profile_generated_at"] = generated_at
            profile_data["deep_research_status"] = DossierMode.FAILED
            profile_data["entity_lock_score"] = entity_lock.score
            profile_data["entity_lock_report"] = entity_lock_report
            profile_data["search_plan"] = search_plan
            profile_data["fail_closed_status"] = fail_closed_status
            profile_data["evidence_graph"] = graph.to_dict()
            profile_data["retrieval_ledger"] = [
                r.model_dump() for r in graph.ledger
            ]
            profile_data["visibility_ledger"] = [
                r.model_dump() for r in graph.get_visibility_ledger_rows()
            ]
            entity.domains = json.dumps(profile_data)
            session.commit()

            return {
                "status": "halted",
                "mode": DossierMode.DEEP_RESEARCH,
                "dossier_mode": dossier_mode,
                "deep_research_status": DossierMode.FAILED,
                "deep_profile": failure_report,
                "generated_at": generated_at,
                "entity_lock": entity_lock_report,
                "qa_report": None,
                "search_plan": search_plan,
                "visibility_report": None,
                "fail_closed_status": fail_closed_status,
                "evidence_graph": graph.to_dict(),
            }

        # --- STEP 3: Generate dossier via LLM (ONLY if pre-synthesis gates pass) ---
        # If LinkedIn PDF available, include PDF text in the interactions summary
        pdf_context = ""
        if profile_data.get("linkedin_pdf_raw_text"):
            pdf_text = profile_data["linkedin_pdf_raw_text"][:10000]
            pdf_context = (
                f"\n\n## USER-SUPPLIED LINKEDIN PDF (artifact evidence)\n"
                f"The following was extracted from a LinkedIn profile PDF uploaded by the user. "
                f"Tag claims from this source as [VERIFIED-PDF].\n\n{pdf_text}"
            )

        # Compute adaptive evidence threshold for the LLM prompt
        total_web_results_2 = sum(len(v) for v in search_results.values())
        if total_web_results_2 >= 10:
            evidence_threshold_2 = 85
        elif total_web_results_2 >= 5:
            evidence_threshold_2 = 70
        else:
            evidence_threshold_2 = 60

        result = generate_deep_profile(
            name=p_name,
            title=p_title,
            company=p_company,
            linkedin_url=p_linkedin,
            location=p_location,
            industry=profile_data.get("company_industry", ""),
            company_size=profile_data.get("company_size"),
            interactions_summary=interactions_summary + pdf_context,
            web_research=web_research,
            visibility_research=visibility_research,
            evidence_threshold=evidence_threshold_2,
            identity_lock_score=entity_lock.score,
        )

        # --- STEP 4: Post-synthesis QA gates ---
        visibility_categories_searched = [
            cat for cat in VISIBILITY_CATEGORIES
            if visibility_results.get(cat) is not None
        ]
        qa_report = generate_dossier_qa_report(
            dossier_text=result,
            disambiguation=entity_lock,
            person_name=p_name,
            visibility_categories=visibility_categories_searched,
            visibility_sweep_executed=visibility_sweep_executed,
        )
        qa_markdown = render_qa_report_markdown(qa_report)

        if not qa_report.passes_all:
            logger.warning(
                "QA gates FAILED for profile %d (%s): genericness=%d, coverage=%.0f%%, "
                "contradictions=%d",
                profile_id,
                p_name,
                qa_report.genericness.genericness_score,
                qa_report.evidence_coverage.coverage_pct,
                len(qa_report.contradictions),
            )

        # --- STEP 5: Post-synthesis fail-closed enforcement ---
        visibility_ledger_count = len(graph.get_visibility_ledger_rows())
        evidence_coverage = qa_report.evidence_coverage.coverage_pct
        total_web_results_2 = sum(len(v) for v in search_results.values())
        should_output, fail_message = enforce_fail_closed_gates(
            dossier_text=result,
            entity_lock_score=entity_lock.score,
            visibility_ledger_count=visibility_ledger_count,
            evidence_coverage_pct=evidence_coverage,
            person_name=p_name,
            has_public_results=has_public_results,
            web_results_count=total_web_results_2,
        )

        # --- STEP 6: Apply mode-based prose filtering ---
        if should_output:
            result = filter_prose_by_mode(result, dossier_mode, entity_lock.score)

        # --- Build visibility report summary ---
        highest_signal = extract_highest_signal_artifacts(graph, max_artifacts=3)
        visibility_report = {
            "sweep_executed": visibility_sweep_executed,
            "categories_searched": visibility_categories_searched,
            "total_results": sum(len(v) for v in visibility_results.values()),
            "ted_tedx_found": any(
                len(visibility_results.get(c, [])) > 0 for c in ("ted", "tedx")
            ),
            "podcast_webinar_found": any(
                len(visibility_results.get(c, [])) > 0 for c in ("podcast", "webinar")
            ),
            "conference_keynote_found": any(
                len(visibility_results.get(c, [])) > 0
                for c in ("conference", "keynote", "summit")
            ),
            "results_by_category": {
                cat: len(visibility_results.get(cat, []))
                for cat in VISIBILITY_CATEGORIES
            },
            "highest_signal_artifacts": highest_signal,
            "coverage_confidence": vis_coverage_confidence,
        }

        # --- Evidence Graph + fail-closed status ---
        fail_closed_status = {
            "gates_passed": should_output,
            "dossier_mode": dossier_mode,
            "mode_reason": mode_reason,
            "visibility_ledger_rows": visibility_ledger_count,
            "visibility_confidence": vis_coverage_confidence,
            "evidence_coverage_pct": round(evidence_coverage, 1),
            "entity_lock_score": entity_lock.score,
            "entity_lock_status": entity_lock.lock_status,
            "has_public_results": has_public_results,
            "failure_message": fail_message if not should_output else None,
        }

        # --- Persist ---
        generated_at = datetime.utcnow().isoformat()
        deep_research_status = (
            DossierMode.SUCCEEDED if should_output else DossierMode.FAILED
        )
        profile_data["deep_profile"] = result if should_output else fail_message
        profile_data["dossier_mode_b_markdown"] = (
            result if should_output else fail_message
        )
        profile_data["deep_profile_generated_at"] = generated_at
        profile_data["deep_research_status"] = deep_research_status
        profile_data["entity_lock_score"] = entity_lock.score
        profile_data["entity_lock_report"] = entity_lock_report
        profile_data["qa_genericness_score"] = qa_report.genericness.genericness_score
        profile_data["qa_evidence_coverage_pct"] = round(
            qa_report.evidence_coverage.coverage_pct, 1
        )
        profile_data["qa_passes_all"] = qa_report.passes_all
        profile_data["qa_report_markdown"] = qa_markdown
        profile_data["search_plan"] = search_plan
        profile_data["visibility_report"] = visibility_report
        profile_data["fail_closed_status"] = fail_closed_status
        profile_data["evidence_graph"] = graph.to_dict()
        profile_data["retrieval_ledger"] = [
            r.model_dump() for r in graph.ledger
        ]
        profile_data["visibility_ledger"] = [
            r.model_dump() for r in graph.get_visibility_ledger_rows()
        ]
        profile_data["evidence_nodes"] = [
            n.model_dump() for n in graph.nodes.values()
        ]
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok" if should_output else "halted",
            "mode": DossierMode.DEEP_RESEARCH,
            "dossier_mode": dossier_mode,
            "deep_research_status": deep_research_status,
            "deep_profile": result if should_output else fail_message,
            "generated_at": generated_at,
            "entity_lock": entity_lock_report,
            "qa_report": {
                "passes_all": qa_report.passes_all,
                "genericness_score": qa_report.genericness.genericness_score,
                "evidence_coverage_pct": round(
                    qa_report.evidence_coverage.coverage_pct, 1
                ),
                "person_level_pct": round(qa_report.person_level.person_pct, 1),
                "contradictions": len(qa_report.contradictions),
                "hallucination_risk_flags": qa_report.hallucination_risk_flags,
                "markdown": qa_markdown,
            },
            "search_plan": search_plan,
            "visibility_report": visibility_report,
            "fail_closed_status": fail_closed_status,
            "evidence_graph": graph.to_dict(),
            "artifacts": {
                "pdf_uploaded": bool(profile_data.get("linkedin_pdf_path")),
                "pdf_evidence_nodes": pdf_artifact_count,
                "pdf_ingested_at": profile_data.get("linkedin_pdf_ingested_at"),
            },
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
    try:
        return get_dashboard_stats()
    except Exception:
        logger.exception("Stats endpoint failed")
        return {
            "profiles": 0,
            "transcripts": 0,
            "briefs": 0,
            "last_sync": None,
        }


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
