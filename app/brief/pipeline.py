"""End-to-end pipeline: input → ingest → resolve → retrieve → generate → output.

This is the main orchestrator that wires all modules together.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from app.brief.generator import generate_brief
from app.brief.renderer import render_markdown
from app.config import settings
from app.ingest.fireflies_ingest import ingest_fireflies_sync
from app.ingest.gmail_ingest import ingest_gmail_for_company, ingest_gmail_for_person
from app.models import BriefOutput
from app.normalize.entity_resolver import resolve_company, resolve_person
from app.retrieve.retriever import retrieve_for_entity
from app.store.database import BriefLog, get_session, init_db

logger = logging.getLogger(__name__)


class PipelineResult:
    """Container for pipeline output."""

    def __init__(
        self,
        brief: BriefOutput,
        markdown: str,
        json_path: Path | None = None,
        md_path: Path | None = None,
    ):
        self.brief = brief
        self.markdown = markdown
        self.json_path = json_path
        self.md_path = md_path


def run_pipeline(
    person: str | None = None,
    company: str | None = None,
    topic: str | None = None,
    meeting_when: str | None = None,
    skip_ingestion: bool = False,
) -> PipelineResult:
    """Run the full Pre-Call Intelligence Briefing pipeline.

    Args:
        person: Person name to brief on
        company: Company name to brief on
        topic: Meeting topic
        meeting_when: Meeting datetime string (ISO format or "YYYY-MM-DD HH:MM")
        skip_ingestion: If True, skip API calls and only use stored data

    Returns:
        PipelineResult with the brief, markdown, and file paths
    """
    init_db()

    # Parse meeting datetime
    meeting_dt = None
    if meeting_when:
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M"):
            try:
                meeting_dt = datetime.strptime(meeting_when, fmt)
                break
            except ValueError:
                continue

    since = datetime.utcnow() - timedelta(days=settings.retrieval_window_days)

    # ── Step 1: Entity Resolution ──
    logger.info("Step 1: Entity resolution")
    person_entity = None
    company_entity = None

    if person:
        person_entity = resolve_person(person)
        logger.info(
            "Resolved person '%s' → entity_id=%s, emails=%s",
            person, person_entity.entity_id, person_entity.emails,
        )

    if company:
        company_entity = resolve_company(company)
        logger.info(
            "Resolved company '%s' → entity_id=%s, domains=%s",
            company, company_entity.entity_id, company_entity.domains,
        )

    # ── Step 2: Ingestion ──
    if not skip_ingestion:
        logger.info("Step 2: Ingestion (fetching from APIs)")

        if person_entity:
            # Fireflies
            try:
                email = person_entity.emails[0] if person_entity.emails else None
                ingest_fireflies_sync(
                    email=email,
                    name=person,
                    since=since,
                    entity_id=person_entity.entity_id,
                )
            except Exception:
                logger.exception("Fireflies ingestion failed – continuing with stored data")

            # Gmail
            try:
                email = person_entity.emails[0] if person_entity.emails else None
                ingest_gmail_for_person(
                    email=email,
                    name=person,
                    since_days=settings.retrieval_window_days,
                    entity_id=person_entity.entity_id,
                )
            except Exception:
                logger.exception("Gmail ingestion failed – continuing with stored data")

        if company_entity:
            try:
                domain = company_entity.domains[0] if company_entity.domains else None
                ingest_gmail_for_company(
                    domain=domain,
                    company_name=company,
                    since_days=settings.retrieval_window_days,
                    entity_id=company_entity.entity_id,
                )
            except Exception:
                logger.exception("Gmail company ingestion failed – continuing with stored data")
    else:
        logger.info("Step 2: Skipping ingestion (using stored data only)")

    # ── Step 3: Retrieval ──
    logger.info("Step 3: Retrieval")
    evidence = retrieve_for_entity(
        entity_id=person_entity.entity_id if person_entity else (
            company_entity.entity_id if company_entity else None
        ),
        person_name=person,
        company_name=company,
        emails=person_entity.emails if person_entity else None,
        aliases=person_entity.aliases if person_entity else None,
        domains=company_entity.domains if company_entity else None,
    )
    logger.info("Retrieved %d source records", evidence.source_count)

    # ── Step 4: Brief Generation ──
    logger.info("Step 4: Brief generation")
    brief = generate_brief(
        person=person,
        company=company,
        topic=topic,
        meeting_datetime=meeting_dt,
        evidence=evidence,
    )

    # ── Step 5: Render Markdown ──
    markdown = render_markdown(brief)

    # ── Step 6: Write output files ──
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    name_slug = (person or company or "unknown").replace(" ", "_")
    date_slug = meeting_dt.strftime("%Y%m%d") if meeting_dt else datetime.utcnow().strftime("%Y%m%d")
    base_name = f"brief_{name_slug}_{date_slug}"

    json_path = output_dir / f"{base_name}.json"
    md_path = output_dir / f"{base_name}.md"

    brief_json = brief.model_dump_json(indent=2)
    json_path.write_text(brief_json)
    md_path.write_text(markdown)

    logger.info("Wrote %s and %s", json_path, md_path)

    # ── Step 7: Audit log ──
    try:
        session = get_session()
        log_entry = BriefLog(
            person=person,
            company=company,
            topic=topic,
            meeting_datetime=meeting_dt,
            brief_json=brief_json,
            brief_markdown=markdown,
            confidence_score=brief.header.confidence_score,
            source_record_ids=json.dumps([r.id for r in evidence.all_source_records]),
        )
        session.add(log_entry)
        session.commit()
        session.close()
    except Exception:
        logger.exception("Failed to write audit log – brief was still produced")

    return PipelineResult(
        brief=brief,
        markdown=markdown,
        json_path=json_path,
        md_path=md_path,
    )
