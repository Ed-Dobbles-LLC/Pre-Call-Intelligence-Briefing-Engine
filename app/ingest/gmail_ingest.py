"""Ingest Gmail emails: fetch → normalise → store."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from email.utils import parsedate_to_datetime

from app.clients.gmail import GmailClient
from app.models import NormalizedEmail
from app.store.database import SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)


def normalize_email(raw: dict) -> NormalizedEmail:
    """Convert a raw Gmail API message into our normalised model."""
    headers = GmailClient.extract_headers(raw)
    body = GmailClient.extract_body(raw)

    date = None
    if headers.get("date"):
        try:
            date = parsedate_to_datetime(headers["date"])
        except (ValueError, TypeError):
            pass

    to_addrs = [a.strip() for a in headers.get("to", "").split(",") if a.strip()]
    cc_addrs = [a.strip() for a in headers.get("cc", "").split(",") if a.strip()]

    return NormalizedEmail(
        source_id=raw.get("id", ""),
        thread_id=raw.get("threadId"),
        subject=headers.get("subject"),
        date=date,
        from_address=headers.get("from"),
        to_addresses=to_addrs,
        cc_addresses=cc_addrs,
        body_plain=body,
        snippet=raw.get("snippet"),
        labels=raw.get("labelIds", []),
        raw_json=raw,
    )


def store_email(normalized: NormalizedEmail, entity_id: int | None = None) -> SourceRecord:
    """Persist a normalised email to the database."""
    init_db()
    session = get_session()
    try:
        existing = session.query(SourceRecord).filter_by(source_id=normalized.source_id).first()
        if existing:
            existing.normalized_json = normalized.model_dump_json()
            existing.summary = normalized.subject
            existing.date = normalized.date
            existing.title = normalized.subject
            all_participants = [normalized.from_address or ""] + normalized.to_addresses
            existing.participants = json.dumps([p for p in all_participants if p])
            existing.body = normalized.body_plain
            if entity_id:
                existing.entity_id = entity_id
            session.commit()
            session.refresh(existing)
            return existing

        all_participants = [normalized.from_address or ""] + normalized.to_addresses
        record = SourceRecord(
            source_type="gmail",
            source_id=normalized.source_id,
            entity_id=entity_id,
            title=normalized.subject,
            date=normalized.date,
            participants=json.dumps([p for p in all_participants if p]),
            summary=normalized.subject,
            body=normalized.body_plain,
            raw_json=json.dumps(normalized.raw_json) if normalized.raw_json else None,
            normalized_json=normalized.model_dump_json(),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
        return record
    finally:
        session.close()


def ingest_gmail_for_person(
    email: str | None = None,
    name: str | None = None,
    since_days: int = 90,
    entity_id: int | None = None,
) -> list[SourceRecord]:
    """Fetch and store Gmail messages for a person."""
    client = GmailClient()
    raw_messages = client.search_by_person(email=email, name=name, since_days=since_days)
    logger.info("Gmail: fetched %d messages for %s / %s", len(raw_messages), email, name)

    records = []
    for raw in raw_messages:
        normalized = normalize_email(raw)
        record = store_email(normalized, entity_id=entity_id)
        records.append(record)
    return records


def ingest_gmail_for_company(
    domain: str | None = None,
    company_name: str | None = None,
    since_days: int = 90,
    entity_id: int | None = None,
) -> list[SourceRecord]:
    """Fetch and store Gmail messages for a company."""
    client = GmailClient()
    raw_messages = client.search_by_company(
        domain=domain, company_name=company_name, since_days=since_days
    )
    logger.info("Gmail: fetched %d messages for company %s / %s", len(raw_messages), domain, company_name)

    records = []
    for raw in raw_messages:
        normalized = normalize_email(raw)
        record = store_email(normalized, entity_id=entity_id)
        records.append(record)
    return records
