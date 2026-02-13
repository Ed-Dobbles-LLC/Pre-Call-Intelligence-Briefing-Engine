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

from app.clients.apollo import ApolloClient, normalize_enrichment
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


async def _fetch_transcripts(client: FirefliesClient, since: datetime) -> list[dict]:
    """Fetch transcripts from Fireflies (async)."""
    return await client.search_transcripts(since=since, limit=200)


def _process_transcripts(raw_transcripts: list[dict]) -> dict:
    """Process raw transcripts: store them and extract participant data.

    Returns a summary dict.
    """
    global _last_sync

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


async def async_sync_fireflies() -> dict:
    """Async version: fetch transcripts and build profiles.

    Safe to call from within an existing event loop (e.g., FastAPI).
    """
    global _sync_running

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
        since = datetime.utcnow() - timedelta(days=settings.retrieval_window_days)

        raw_transcripts = await _fetch_transcripts(client, since)
        logger.info("Auto-sync: fetched %d transcripts from Fireflies", len(raw_transcripts))

        result = _process_transcripts(raw_transcripts)

        # Enrich profiles with Apollo.io data (photos, LinkedIn, titles)
        enriched = await _enrich_profiles_with_apollo()
        result["profiles_enriched"] = enriched

        return result

    except Exception as exc:
        logger.exception("Auto-sync failed")
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": f"Sync failed: {exc}",
        }
    finally:
        _sync_running = False


def sync_fireflies_transcripts() -> dict:
    """Sync version: for use from background threads (no existing event loop).

    Creates its own event loop. Do NOT call from inside FastAPI handlers.
    """
    global _sync_running

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
        since = datetime.utcnow() - timedelta(days=settings.retrieval_window_days)

        loop = asyncio.new_event_loop()
        try:
            raw_transcripts = loop.run_until_complete(
                _fetch_transcripts(client, since)
            )
        finally:
            loop.close()

        logger.info("Auto-sync: fetched %d transcripts from Fireflies", len(raw_transcripts))

        result = _process_transcripts(raw_transcripts)

        # Enrich profiles with Apollo.io data
        enrich_loop = asyncio.new_event_loop()
        try:
            enriched = enrich_loop.run_until_complete(
                _enrich_profiles_with_apollo()
            )
            result["profiles_enriched"] = enriched
        finally:
            enrich_loop.close()

        return result

    except Exception as exc:
        logger.exception("Auto-sync failed")
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": f"Sync failed: {exc}",
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


async def _enrich_profiles_with_apollo() -> int:
    """Enrich person profiles via Apollo.io (skips already-enriched contacts)."""
    if not settings.apollo_api_key:
        return 0

    session = get_session()
    enriched_count = 0
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
            EntityRecord.domains.isnot(None),
        ).all()

        # Collect entities that need enrichment
        to_enrich: list[tuple[EntityRecord, dict]] = []
        for entity in entities:
            try:
                profile_data = json.loads(entity.domains or "{}")
            except (json.JSONDecodeError, TypeError):
                continue

            # Skip if already enriched (has a non-empty apollo field)
            if profile_data.get("apollo_enriched"):
                continue

            if not profile_data.get("meeting_count"):
                continue

            emails = json.loads(entity.emails or "[]")
            to_enrich.append((entity, {
                "email": emails[0] if emails else None,
                "name": entity.name,
                "company": profile_data.get("company"),
            }))

        if not to_enrich:
            return 0

        logger.info("Apollo enrichment: %d contacts to enrich", len(to_enrich))

        client = ApolloClient()

        # Build bulk request details
        bulk_details = []
        for _, info in to_enrich:
            detail: dict[str, str] = {}
            if info["email"]:
                detail["email"] = info["email"]
            elif info["name"]:
                parts = info["name"].split(None, 1)
                detail["first_name"] = parts[0]
                if len(parts) > 1:
                    detail["last_name"] = parts[1]
                if info["company"]:
                    detail["organization_name"] = info["company"]
            bulk_details.append(detail)

        results = await client.enrich_bulk(bulk_details)

        for (entity, _), person in zip(to_enrich, results):
            enrichment = normalize_enrichment(person)
            if not enrichment:
                # Mark as attempted so we don't retry every sync
                try:
                    profile_data = json.loads(entity.domains or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue
                profile_data["apollo_enriched"] = True
                entity.domains = json.dumps(profile_data)
                continue

            try:
                profile_data = json.loads(entity.domains or "{}")
            except (json.JSONDecodeError, TypeError):
                continue

            profile_data["photo_url"] = enrichment.get("photo_url", "")
            profile_data["linkedin_url"] = enrichment.get("linkedin_url", "")
            profile_data["title"] = enrichment.get("title", "")
            profile_data["headline"] = enrichment.get("headline", "")
            profile_data["seniority"] = enrichment.get("seniority", "")
            profile_data["location"] = ", ".join(
                filter(None, [enrichment.get("city"), enrichment.get("state")])
            )

            # Use Apollo company name if we only had a domain guess
            if enrichment.get("company_name") and not profile_data.get("company"):
                profile_data["company"] = enrichment["company_name"]
            elif enrichment.get("company_name"):
                profile_data["company_full"] = enrichment["company_name"]

            profile_data["company_industry"] = enrichment.get("company_industry", "")
            profile_data["company_size"] = enrichment.get("company_size")
            profile_data["company_linkedin"] = enrichment.get("company_linkedin", "")
            profile_data["apollo_enriched"] = True

            entity.domains = json.dumps(profile_data)
            enriched_count += 1

        session.commit()
        logger.info("Apollo enrichment: enriched %d contacts", enriched_count)
    except Exception:
        session.rollback()
        logger.exception("Apollo enrichment failed")
    finally:
        session.close()

    return enriched_count


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
                # Apollo enrichment fields
                "photo_url": profile_data.get("photo_url", ""),
                "linkedin_url": profile_data.get("linkedin_url", ""),
                "title": profile_data.get("title", ""),
                "headline": profile_data.get("headline", ""),
                "seniority": profile_data.get("seniority", ""),
                "location": profile_data.get("location", ""),
                "company_full": profile_data.get("company_full", ""),
                "company_industry": profile_data.get("company_industry", ""),
                "company_size": profile_data.get("company_size"),
                "company_linkedin": profile_data.get("company_linkedin", ""),
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
    """Start a background thread that periodically syncs Fireflies transcripts.

    Uses the sync version (own event loop) since this runs in a separate thread.
    """
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
