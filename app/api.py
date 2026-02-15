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
from app.brief.evidence_graph import (
    EvidenceGraph,
    extract_highest_signal_artifacts,
    compute_visibility_coverage_confidence,
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
    # Repair any linkedin_status fields wiped by the previous sync bug
    try:
        repaired = repair_linkedin_status()
        if repaired:
            logger.info("Repaired %d profiles with missing linkedin_status", repaired)
    except Exception:
        logger.exception("LinkedIn status repair failed")

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
    """
    all_p = get_all_profiles()
    return [
        p for p in all_p
        if p.get("linkedin_status") in ("pending_review", "no_match")
    ]


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


@app.post("/profiles/{profile_id}/deep-profile", dependencies=[Depends(verify_api_key)])
async def generate_profile_research(profile_id: int):
    """Generate a decision-grade intelligence dossier for a verified contact.

    Pipeline: Entity Lock → SerpAPI retrieval → LLM synthesis → QA gates.
    Returns the dossier, entity lock report, QA report, and search queries.
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

        entity_lock = score_disambiguation(
            name=p_name,
            company=p_company,
            title=p_title,
            linkedin_url=p_linkedin,
            location=p_location,
            search_results=search_results,
            apollo_data=apollo_data,
            has_meeting_data=bool(profile_data.get("interactions")),
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
                "linkedin_confirmed": entity_lock.linkedin_confirmed,
                "location_match": entity_lock.location_match,
                "photo_available": entity_lock.photo_available,
                "multiple_sources_agree": entity_lock.multiple_sources_agree,
                "employer_match": entity_lock.employer_match,
                "meeting_confirmed": entity_lock.meeting_confirmed,
                "secondary_source_match": entity_lock.secondary_source_match,
            },
        }

        # --- STEP 3: Generate dossier via LLM ---
        result = generate_deep_profile(
            name=p_name,
            title=p_title,
            company=p_company,
            linkedin_url=p_linkedin,
            location=p_location,
            industry=profile_data.get("company_industry", ""),
            company_size=profile_data.get("company_size"),
            interactions_summary=interactions_summary,
            web_research=web_research,
            visibility_research=visibility_research,
        )

        # --- STEP 4: QA gates (with visibility sweep audit) ---
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

        # --- STEP 4b: Fail-closed gate enforcement ---
        visibility_ledger_count = len(graph.get_visibility_ledger_rows())
        evidence_coverage = qa_report.evidence_coverage.coverage_pct
        should_output, fail_message = enforce_fail_closed_gates(
            dossier_text=result,
            entity_lock_score=entity_lock.score,
            visibility_ledger_count=visibility_ledger_count,
            evidence_coverage_pct=evidence_coverage,
            person_name=p_name,
        )

        # --- Build visibility report summary ---
        highest_signal = extract_highest_signal_artifacts(graph, max_artifacts=3)
        vis_coverage_confidence = compute_visibility_coverage_confidence(graph)
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
            "visibility_ledger_rows": visibility_ledger_count,
            "evidence_coverage_pct": round(evidence_coverage, 1),
            "entity_lock_score": entity_lock.score,
            "entity_lock_status": entity_lock.lock_status,
            "failure_message": fail_message if not should_output else None,
        }

        # --- Persist ---
        generated_at = datetime.utcnow().isoformat()
        profile_data["deep_profile"] = result if should_output else fail_message
        profile_data["deep_profile_generated_at"] = generated_at
        profile_data["entity_lock_score"] = entity_lock.score
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
        entity.domains = json.dumps(profile_data)
        session.commit()

        return {
            "status": "ok" if should_output else "halted",
            "deep_profile": result if should_output else fail_message,
            "generated_at": generated_at,
            "entity_lock": entity_lock_report,
            "qa_report": {
                "passes_all": qa_report.passes_all,
                "genericness_score": qa_report.genericness.genericness_score,
                "evidence_coverage_pct": round(qa_report.evidence_coverage.coverage_pct, 1),
                "person_level_pct": round(qa_report.person_level.person_pct, 1),
                "contradictions": len(qa_report.contradictions),
                "hallucination_risk_flags": qa_report.hallucination_risk_flags,
                "markdown": qa_markdown,
            },
            "search_plan": search_plan,
            "visibility_report": visibility_report,
            "fail_closed_status": fail_closed_status,
            "evidence_graph": graph.to_dict(),
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
