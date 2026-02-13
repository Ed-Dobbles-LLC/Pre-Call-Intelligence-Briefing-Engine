"""Automatic Fireflies transcript sync and profile builder.

Fetches ALL transcripts from Fireflies, stores them, and builds/updates
contact profiles by extracting participants from each transcript.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta

from app.clients.fireflies import FirefliesClient
from app.config import settings
from app.ingest.fireflies_ingest import normalize_transcript, store_transcript
from app.store.database import EntityRecord, SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)

# Global state for tracking sync status
_last_sync: datetime | None = None
_sync_lock = threading.Lock()
_sync_running = False


def get_last_sync() -> datetime | None:
    return _last_sync


def is_sync_running() -> bool:
    return _sync_running


def _extract_participants_from_transcript(raw: dict) -> list[dict]:
    """Extract structured participant info from a raw Fireflies transcript."""
    participants = []
    seen_emails = set()
    seen_names = set()

    # From meeting_attendees (most structured)
    for attendee in raw.get("meeting_attendees") or []:
        email = (attendee.get("email") or "").strip().lower()
        name = attendee.get("displayName") or attendee.get("name") or ""
        name = name.strip()

        if not name and not email:
            continue

        key = email or name.lower()
        if key in seen_emails or key in seen_names:
            continue

        if email:
            seen_emails.add(email)
        if name:
            seen_names.add(name.lower())

        participants.append({
            "name": name,
            "email": email,
        })

    # From speaker names in sentences (discover speakers not in attendees)
    for sentence in raw.get("sentences") or []:
        speaker = (sentence.get("speaker_name") or "").strip()
        if not speaker or speaker.lower() in seen_names:
            continue
        seen_names.add(speaker.lower())
        participants.append({
            "name": speaker,
            "email": "",
        })

    return participants


def _infer_company_from_email(email: str) -> str | None:
    """Try to extract company name from email domain."""
    if not email or "@" not in email:
        return None
    domain = email.split("@")[1].lower()
    # Skip common free email providers
    free_providers = {
        "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
        "aol.com", "icloud.com", "mail.com", "protonmail.com",
        "live.com", "msn.com", "ymail.com",
    }
    if domain in free_providers:
        return None
    # Extract company name from domain (e.g., "acme.com" -> "Acme")
    company = domain.split(".")[0]
    return company.title()


def _determine_relationship_health(
    meeting_count: int,
    last_interaction: datetime | None,
) -> str:
    """Infer relationship health from meeting frequency and recency."""
    if not last_interaction:
        return "unknown"

    days_since = (datetime.utcnow() - last_interaction).days

    if days_since <= 14 and meeting_count >= 2:
        return "active"
    if days_since <= 30:
        return "warm"
    if days_since <= 60:
        return "neutral"
    return "cold"


def sync_fireflies_transcripts() -> dict:
    """Fetch all Fireflies transcripts and build/update profiles.

    Returns a summary dict with counts.
    """
    global _last_sync, _sync_running

    if not settings.fireflies_api_key:
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": "Fireflies API key not configured",
        }

    with _sync_lock:
        if _sync_running:
            return {
                "transcripts_synced": 0,
                "profiles_updated": 0,
                "error": "Sync already in progress",
            }
        _sync_running = True

    try:
        init_db()
        client = FirefliesClient()

        # Fetch transcripts from the last retrieval_window_days
        since = datetime.utcnow() - timedelta(days=settings.retrieval_window_days)

        loop = asyncio.new_event_loop()
        try:
            raw_transcripts = loop.run_until_complete(
                client.search_transcripts(since=since, limit=200)
            )
        finally:
            loop.close()

        logger.info("Auto-sync: fetched %d transcripts from Fireflies", len(raw_transcripts))

        transcripts_synced = 0
        all_participants: dict[str, dict] = {}  # key -> participant info

        for raw in raw_transcripts:
            # Normalize and store transcript
            normalized = normalize_transcript(raw)
            store_transcript(normalized)
            transcripts_synced += 1

            # Extract participants
            transcript_date = normalized.date
            transcript_title = normalized.title or "Meeting"
            summary = normalized.summary or ""
            action_items = normalized.action_items or []

            for p in _extract_participants_from_transcript(raw):
                key = p["email"] or p["name"].lower()
                if not key:
                    continue

                if key not in all_participants:
                    all_participants[key] = {
                        "name": p["name"],
                        "email": p["email"],
                        "company": _infer_company_from_email(p["email"]),
                        "meeting_count": 0,
                        "last_interaction": None,
                        "interactions": [],
                        "action_items": [],
                    }

                entry = all_participants[key]

                # Update name if we have a better one
                if p["name"] and (not entry["name"] or len(p["name"]) > len(entry["name"])):
                    entry["name"] = p["name"]
                if p["email"] and not entry["email"]:
                    entry["email"] = p["email"]
                if not entry["company"]:
                    entry["company"] = _infer_company_from_email(p["email"])

                entry["meeting_count"] += 1

                if transcript_date:
                    if not entry["last_interaction"] or transcript_date > entry["last_interaction"]:
                        entry["last_interaction"] = transcript_date

                entry["interactions"].append({
                    "title": transcript_title,
                    "date": transcript_date.isoformat() if transcript_date else None,
                    "summary": summary[:300] if summary else None,
                    "participants": normalized.participants[:5],
                })

                # Collect action items
                for item in action_items:
                    if item and item not in entry["action_items"]:
                        entry["action_items"].append(item)

        # Build profiles from participants
        profiles_updated = _update_profiles(all_participants)

        _last_sync = datetime.utcnow()

        return {
            "transcripts_synced": transcripts_synced,
            "profiles_updated": profiles_updated,
        }

    except Exception:
        logger.exception("Auto-sync failed")
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": "Sync failed - check server logs",
        }
    finally:
        _sync_running = False


def _update_profiles(all_participants: dict[str, dict]) -> int:
    """Update entity records and store profile data from participant info."""
    session = get_session()
    updated = 0

    try:
        for key, pdata in all_participants.items():
            name = pdata["name"]
            if not name:
                continue

            # Find or create entity
            entity = session.query(EntityRecord).filter(
                EntityRecord.entity_type == "person",
                EntityRecord.name.ilike(f"%{name}%"),
            ).first()

            if not entity and pdata["email"]:
                # Try email match
                for e in session.query(EntityRecord).filter(
                    EntityRecord.entity_type == "person"
                ).all():
                    if pdata["email"].lower() in [
                        x.lower() for x in json.loads(e.emails or "[]")
                    ]:
                        entity = e
                        break

            if not entity:
                entity = EntityRecord(
                    name=name,
                    entity_type="person",
                )
                session.add(entity)
                session.flush()

            # Update entity fields
            emails = json.loads(entity.emails or "[]")
            if pdata["email"] and pdata["email"] not in [e.lower() for e in emails]:
                emails.append(pdata["email"])
                entity.emails = json.dumps(emails)

            aliases = json.loads(entity.aliases or "[]")
            if name.lower() not in [a.lower() for a in aliases]:
                aliases.append(name.lower())
                entity.aliases = json.dumps(aliases)

            # Store profile metadata in domains field (repurposed for persons)
            # We serialize the full profile data as JSON here
            health = _determine_relationship_health(
                pdata["meeting_count"],
                pdata["last_interaction"],
            )

            profile_data = {
                "meeting_count": pdata["meeting_count"],
                "last_interaction": pdata["last_interaction"].isoformat() if pdata["last_interaction"] else None,
                "company": pdata["company"],
                "relationship_health": health,
                "interactions": sorted(
                    pdata["interactions"],
                    key=lambda x: x.get("date") or "",
                    reverse=True,
                )[:20],
                "action_items": pdata["action_items"][:10],
                "action_items_count": len(pdata["action_items"]),
                "email_count": 0,
                "updated_at": datetime.utcnow().isoformat(),
            }

            entity.domains = json.dumps(profile_data)
            entity.updated_at = datetime.utcnow()
            updated += 1

        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Failed to update profiles")
        raise
    finally:
        session.close()

    return updated


def get_all_profiles() -> list[dict]:
    """Retrieve all person profiles from the database."""
    init_db()
    session = get_session()
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person"
        ).all()

        profiles = []
        for entity in entities:
            # Parse profile data from domains field
            profile_data = {}
            if entity.domains:
                try:
                    profile_data = json.loads(entity.domains)
                except (json.JSONDecodeError, TypeError):
                    pass

            # Skip entities without profile data (no meetings)
            if not profile_data.get("meeting_count"):
                continue

            emails = json.loads(entity.emails or "[]")
            profile = {
                "id": entity.id,
                "name": entity.name,
                "email": emails[0] if emails else None,
                "company": profile_data.get("company"),
                "meeting_count": profile_data.get("meeting_count", 0),
                "email_count": profile_data.get("email_count", 0),
                "last_interaction": profile_data.get("last_interaction"),
                "relationship_health": profile_data.get("relationship_health", "unknown"),
                "interactions": profile_data.get("interactions", []),
                "action_items": profile_data.get("action_items", []),
                "action_items_count": profile_data.get("action_items_count", 0),
            }
            profiles.append(profile)

        # Sort by last interaction (most recent first)
        profiles.sort(
            key=lambda p: p.get("last_interaction") or "",
            reverse=True,
        )
        return profiles
    finally:
        session.close()


def get_dashboard_stats() -> dict:
    """Get summary stats for the dashboard."""
    init_db()
    session = get_session()
    try:
        profiles_count = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
            EntityRecord.domains.isnot(None),
            EntityRecord.domains != "[]",
        ).count()

        transcripts_count = session.query(SourceRecord).filter(
            SourceRecord.source_type == "fireflies"
        ).count()

        from app.store.database import BriefLog
        briefs_count = session.query(BriefLog).count()

        return {
            "profiles": profiles_count,
            "transcripts": transcripts_count,
            "briefs": briefs_count,
            "last_sync": _last_sync.isoformat() if _last_sync else None,
        }
    finally:
        session.close()


def start_background_sync(interval_minutes: int = 30):
    """Start a background thread that periodically syncs Fireflies transcripts."""
    def _sync_loop():
        while True:
            try:
                if settings.fireflies_api_key:
                    logger.info("Background sync: starting...")
                    result = sync_fireflies_transcripts()
                    logger.info("Background sync: %s", result)
            except Exception:
                logger.exception("Background sync error")
            time.sleep(interval_minutes * 60)

    thread = threading.Thread(target=_sync_loop, daemon=True, name="fireflies-sync")
    thread.start()
    logger.info("Background sync thread started (interval: %d min)", interval_minutes)
