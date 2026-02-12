"""Ingest Fireflies transcripts: fetch → normalise → store."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from app.clients.fireflies import FirefliesClient
from app.models import NormalizedTranscript, TranscriptSentence
from app.store.database import SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)


def _parse_fireflies_date(raw_date) -> datetime | None:
    """Parse the date field which can be epoch-ms or ISO string."""
    if raw_date is None:
        return None
    try:
        if isinstance(raw_date, (int, float)):
            return datetime.utcfromtimestamp(raw_date / 1000)
        return datetime.fromisoformat(str(raw_date))
    except (ValueError, TypeError):
        return None


def normalize_transcript(raw: dict) -> NormalizedTranscript:
    """Convert a raw Fireflies transcript dict into our normalised model."""
    summary_obj = raw.get("summary") or {}
    sentences_raw = raw.get("sentences") or []

    sentences = [
        TranscriptSentence(
            speaker=s.get("speaker_name"),
            text=s.get("text", ""),
            start_time=s.get("start_time"),
            end_time=s.get("end_time"),
        )
        for s in sentences_raw
    ]

    action_items = summary_obj.get("action_items") or []
    if isinstance(action_items, str):
        action_items = [line.strip("- ").strip() for line in action_items.split("\n") if line.strip()]

    # Build participant list from both participants[] and meeting_attendees[]
    participants = list(raw.get("participants") or [])
    for attendee in raw.get("meeting_attendees") or []:
        email = attendee.get("email")
        display = attendee.get("displayName") or attendee.get("name")
        if email and email not in participants:
            participants.append(email)
        if display and display not in participants:
            participants.append(display)

    # Use the best available summary: overview > short_summary > shorthand_bullet
    summary = (
        summary_obj.get("overview")
        or summary_obj.get("short_summary")
        or summary_obj.get("shorthand_bullet")
    )

    return NormalizedTranscript(
        source_id=raw.get("id", ""),
        title=raw.get("title"),
        date=_parse_fireflies_date(raw.get("date")),
        duration_minutes=(raw.get("duration") or 0) / 60 if raw.get("duration") else None,
        participants=participants,
        summary=summary,
        action_items=action_items,
        sentences=sentences,
        raw_json=raw,
    )


def store_transcript(normalized: NormalizedTranscript, entity_id: int | None = None) -> SourceRecord:
    """Persist a normalised transcript to the database."""
    init_db()
    session = get_session()
    try:
        existing = session.query(SourceRecord).filter_by(source_id=normalized.source_id).first()
        if existing:
            existing.normalized_json = normalized.model_dump_json()
            existing.raw_json = json.dumps(normalized.raw_json) if normalized.raw_json else None
            existing.summary = normalized.summary
            existing.action_items = json.dumps(normalized.action_items)
            existing.date = normalized.date
            existing.title = normalized.title
            existing.participants = json.dumps(normalized.participants)
            if entity_id:
                existing.entity_id = entity_id
            session.commit()
            session.refresh(existing)
            return existing

        # Extract transcript_url for deep-linking in citations
        transcript_url = None
        if normalized.raw_json:
            transcript_url = normalized.raw_json.get("transcript_url")

        record = SourceRecord(
            source_type="fireflies",
            source_id=normalized.source_id,
            entity_id=entity_id,
            title=normalized.title,
            date=normalized.date,
            participants=json.dumps(normalized.participants),
            summary=normalized.summary,
            action_items=json.dumps(normalized.action_items),
            body="\n".join(
                f"{s.speaker or 'Unknown'}: {s.text}" for s in normalized.sentences
            ),
            raw_json=json.dumps(normalized.raw_json) if normalized.raw_json else None,
            normalized_json=normalized.model_dump_json(),
            link=transcript_url,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


async def ingest_fireflies_for_person(
    email: str | None = None,
    name: str | None = None,
    since: datetime | None = None,
    entity_id: int | None = None,
) -> list[SourceRecord]:
    """Fetch and store Fireflies transcripts for a person."""
    client = FirefliesClient()
    raw_transcripts = await client.search_transcripts(
        participant_email=email,
        participant_name=name,
        since=since,
    )
    logger.info("Fireflies: fetched %d transcripts for %s / %s", len(raw_transcripts), email, name)

    records = []
    for raw in raw_transcripts:
        normalized = normalize_transcript(raw)
        record = store_transcript(normalized, entity_id=entity_id)
        records.append(record)
    return records


def ingest_fireflies_sync(
    email: str | None = None,
    name: str | None = None,
    since: datetime | None = None,
    entity_id: int | None = None,
) -> list[SourceRecord]:
    """Synchronous wrapper for CLI usage."""
    return asyncio.run(
        ingest_fireflies_for_person(email=email, name=name, since=since, entity_id=entity_id)
    )
