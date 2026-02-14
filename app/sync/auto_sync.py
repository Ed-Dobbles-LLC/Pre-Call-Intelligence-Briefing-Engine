"""Automatic Fireflies transcript sync, Gmail email sync, and profile builder.

Fetches ALL transcripts from Fireflies, syncs Gmail emails, stores them,
and builds/updates contact profiles with enrichment from Apollo.io.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from datetime import datetime, timedelta

from app.clients.apollo import ApolloClient, normalize_candidate, normalize_enrichment
from app.clients.fireflies import FirefliesClient
from app.config import settings
from app.ingest.fireflies_ingest import normalize_transcript, store_transcript
from app.ingest.gmail_ingest import normalize_email, store_email
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


_FREE_EMAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com",
    "aol.com", "icloud.com", "mail.com", "protonmail.com",
    "live.com", "msn.com", "ymail.com", "me.com",
}


def _infer_company_from_email(email: str) -> str | None:
    """Try to extract company name from email domain."""
    if not email or "@" not in email:
        return None
    domain = email.split("@")[1].lower()
    if domain in _FREE_EMAIL_PROVIDERS:
        return None
    company = domain.split(".")[0]
    return company.title()


def _infer_company_from_meeting(title: str, participants: list[dict]) -> str | None:
    """Infer company from meeting title and co-participant email domains.

    Examples:
      "Interview with AnswerRocket" → "AnswerRocket"
      "Ed Dobbles - VP, Global Analytics & Data Science" + participant @aristocrat.com → "Aristocrat"
    """
    title_lower = (title or "").lower()

    # Strategy 1: Extract company from "Interview with <Company>" pattern
    patterns = [
        r"interview\s+with\s+(.+?)(?:\s*[-–(]|$)",
        r"(?:call|meeting|chat)\s+with\s+(.+?)(?:\s*[-–(]|$)",
        r"^(.+?)\s+(?:video\s+)?call\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, title_lower)
        if match:
            candidate = match.group(1).strip()
            # Filter out person names (companies usually don't have first+last pattern)
            if candidate and " " not in candidate:
                return candidate.title()
            # If it's a known company keyword, use it
            words = candidate.split()
            if len(words) == 1:
                return words[0].title()

    # Strategy 2: Use co-participant email domains (most common non-free domain)
    domain_counts: dict[str, int] = {}
    for p in participants:
        email = (p.get("email") or "").lower()
        if email and "@" in email:
            domain = email.split("@")[1]
            if domain not in _FREE_EMAIL_PROVIDERS and "greenhouse.io" not in domain and "metaview" not in domain:
                company = domain.split(".")[0].title()
                domain_counts[company] = domain_counts.get(company, 0) + 1

    if domain_counts:
        # Return the most common company domain
        return max(domain_counts, key=domain_counts.get)

    return None


# Common nickname → formal name variants for Apollo enrichment
_NAME_VARIANTS = {
    "ben": ["benjamin", "benedict"],
    "bill": ["william"],
    "bob": ["robert"],
    "charlie": ["charles"],
    "chris": ["christopher"],
    "dan": ["daniel"],
    "dave": ["david"],
    "dick": ["richard"],
    "ed": ["edward", "edmund"],
    "eli": ["elijah", "elisabeth", "elizabeth"],
    "jake": ["jacob"],
    "jim": ["james"],
    "joe": ["joseph"],
    "jon": ["jonathan"],
    "kate": ["katherine", "catherine"],
    "ken": ["kenneth"],
    "liz": ["elizabeth", "elisabeth"],
    "matt": ["matthew"],
    "mike": ["michael"],
    "nick": ["nicholas"],
    "pat": ["patrick", "patricia"],
    "pete": ["peter"],
    "rob": ["robert"],
    "sam": ["samuel", "samantha"],
    "steve": ["steven", "stephen"],
    "sue": ["susan", "suzanne"],
    "ted": ["theodore", "edward"],
    "tim": ["timothy"],
    "tom": ["thomas"],
    "tony": ["anthony"],
    "will": ["william"],
}


def _is_non_person(name: str, email: str = "") -> bool:
    """Filter out meeting rooms, system accounts, and non-person entries."""
    name_lower = name.lower().strip()
    email_lower = (email or "").lower()

    # System/bot email patterns
    if email_lower and any(
        pat in email_lower
        for pat in ["@metaview.ai", "schedule@", "noreply@", "calendar@", "call-"]
    ):
        return True

    # Meeting room / location names
    room_patterns = [
        "floor", "room", "suite", "building", " - ", "conf ",
        "summerlin", "hq-", "nyc-", "usa-", "boardroom", "lobby",
    ]
    if any(pat in name_lower for pat in room_patterns):
        return True

    # Names that start with a location/room code pattern (e.g. "USA-NYC-HQ-36-19")
    if re.match(r"^[A-Z]{2,4}[-\s]", name):
        return True

    # Names that are clearly not people (just an email, etc.)
    if "@" in name and "." in name:
        return True

    return False


def _normalize_name(name: str) -> str:
    """Normalize a name for dedup comparison: 'Fox, Una' -> 'una fox', 'TRAWINSKI' -> 'trawinski'."""
    name = name.strip()
    # Handle "Last, First" format
    if "," in name:
        parts = [p.strip() for p in name.split(",", 1)]
        if len(parts) == 2:
            name = f"{parts[1]} {parts[0]}"
    return name.lower()


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

        transcript_participants = _extract_participants_from_transcript(raw)

        # Infer company from meeting title + participant emails
        meeting_company = _infer_company_from_meeting(
            transcript_title, transcript_participants
        )

        for p in transcript_participants:
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
                # Try email domain first, then meeting context
                entry["company"] = (
                    _infer_company_from_email(p["email"])
                    or meeting_company
                )

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
    Holds _sync_lock during all database-mutating operations so that
    concurrent repair or profile reads see a consistent state.
    """
    global _sync_running

    if not settings.fireflies_api_key:
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": "Fireflies API key not configured",
        }

    acquired = _sync_lock.acquire(blocking=False)
    if not acquired:
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

        # Sync Gmail emails for contacts and discover email-only profiles
        gmail_result = _sync_gmail_emails()
        result["emails_synced"] = gmail_result["emails_synced"]
        result["email_profiles_created"] = gmail_result["email_profiles_created"]

        # Enrich profiles with Apollo.io data (photos, LinkedIn, titles)
        enriched = await _enrich_profiles_with_apollo()
        result["profiles_enriched"] = enriched

        # Re-enrich confirmed profiles that are missing photos
        re_enriched = await _re_enrich_confirmed_profiles()
        result["photos_re_enriched"] = re_enriched

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
        _sync_lock.release()


def sync_fireflies_transcripts() -> dict:
    """Sync version: for use from background threads (no existing event loop).

    Creates its own event loop. Do NOT call from inside FastAPI handlers.
    Holds _sync_lock during all database-mutating operations.
    """
    global _sync_running

    if not settings.fireflies_api_key:
        return {
            "transcripts_synced": 0,
            "profiles_updated": 0,
            "error": "Fireflies API key not configured",
        }

    acquired = _sync_lock.acquire(blocking=False)
    if not acquired:
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

        # Sync Gmail
        gmail_result = _sync_gmail_emails()
        result["emails_synced"] = gmail_result["emails_synced"]
        result["email_profiles_created"] = gmail_result["email_profiles_created"]

        # Enrich profiles with Apollo.io data
        enrich_loop = asyncio.new_event_loop()
        try:
            enriched = enrich_loop.run_until_complete(
                _enrich_profiles_with_apollo()
            )
            result["profiles_enriched"] = enriched

            # Re-enrich confirmed profiles that are missing photos
            re_enriched = enrich_loop.run_until_complete(
                _re_enrich_confirmed_profiles()
            )
            result["photos_re_enriched"] = re_enriched
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
        _sync_lock.release()


def _update_profiles(all_participants: dict[str, dict]) -> int:
    """Update entity records and store profile data from participant info."""
    session = get_session()
    updated = 0

    try:
        for key, pdata in all_participants.items():
            name = pdata["name"]
            if not name:
                continue

            # Skip non-person entries (meeting rooms, system accounts, etc.)
            if _is_non_person(name, pdata.get("email", "")):
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
            # Preserve existing Apollo enrichment data if present
            existing_profile = {}
            if entity.domains:
                try:
                    existing_profile = json.loads(entity.domains)
                except (json.JSONDecodeError, TypeError):
                    pass

            health = _determine_relationship_health(
                pdata["meeting_count"],
                pdata["last_interaction"],
            )

            profile_data = {
                "meeting_count": pdata["meeting_count"],
                "last_interaction": pdata["last_interaction"].isoformat() if pdata["last_interaction"] else None,
                "company": pdata["company"] or existing_profile.get("company"),
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

            # Carry forward Apollo enrichment, LinkedIn verification, and deep profile
            for key in (
                "apollo_enriched", "photo_url", "linkedin_url", "title",
                "headline", "seniority", "location", "company_full",
                "company_industry", "company_size", "company_linkedin",
                "linkedin_status", "linkedin_candidates",
                "deep_profile", "deep_profile_generated_at",
            ):
                if key in existing_profile:
                    profile_data[key] = existing_profile[key]

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


def _parse_email_address(header_value: str) -> tuple[str, str]:
    """Extract (name, email) from a Gmail header like 'Name <email>' or just 'email'."""
    match = re.match(r'"?([^"<]*)"?\s*<([^>]+)>', header_value.strip())
    if match:
        return match.group(1).strip(), match.group(2).strip().lower()
    # Plain email
    email = header_value.strip().lower()
    if "@" in email:
        return "", email
    return "", ""


def _sync_gmail_emails() -> dict:
    """Sync Gmail emails for known contacts, discover email-only contacts."""
    from app.clients.gmail import GmailClient

    client = GmailClient()
    if not client.service:
        logger.info("Gmail service not available, skipping email sync")
        return {"emails_synced": 0, "email_profiles_created": 0}

    session = get_session()
    emails_synced = 0
    email_profiles_created = 0

    try:
        # Build a set of known contact emails → entity
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
        ).all()

        known_emails: dict[str, EntityRecord] = {}  # email -> entity
        own_emails: set[str] = set()  # user's own emails (skip these)
        for entity in entities:
            for email in json.loads(entity.emails or "[]"):
                known_emails[email.lower()] = entity

        # Fetch recent personal emails (exclude noise categories)
        since = datetime.utcnow() - timedelta(days=min(settings.retrieval_window_days, 60))
        date_str = since.strftime("%Y/%m/%d")
        query = f"after:{date_str} -category:promotions -category:social -category:updates -category:forums"
        raw_messages = client.search_messages(query, max_results=200)
        logger.info("Gmail sync: fetched %d messages", len(raw_messages))

        # Track all correspondents for email-only profile creation
        correspondents: dict[str, dict] = {}  # email -> info

        for raw_msg in raw_messages:
            normalized = normalize_email(raw_msg)

            # Parse sender and recipients
            from_name, from_email = _parse_email_address(normalized.from_address or "")
            to_addresses = []
            for addr in normalized.to_addresses:
                name, email = _parse_email_address(addr)
                if email:
                    to_addresses.append((name, email))

            # Determine who this email is with (the other party, not us)
            all_parties = [(from_name, from_email)] + to_addresses

            # Figure out which entity this email belongs to
            matched_entity = None
            for _, party_email in all_parties:
                if party_email in known_emails:
                    matched_entity = known_emails[party_email]
                    break

            # Store the email
            record = store_email(
                normalized,
                entity_id=matched_entity.id if matched_entity else None,
            )
            emails_synced += 1

            # Track unknown correspondents for email-only profiles
            for corr_name, corr_email in all_parties:
                if not corr_email or corr_email in known_emails:
                    continue
                if _is_non_person(corr_name or corr_email, corr_email):
                    continue

                if corr_email not in correspondents:
                    correspondents[corr_email] = {
                        "name": corr_name,
                        "email": corr_email,
                        "email_count": 0,
                        "last_email": None,
                        "interactions": [],
                        "has_scheduling": False,
                    }
                c = correspondents[corr_email]
                c["email_count"] += 1
                if corr_name and (not c["name"] or len(corr_name) > len(c["name"])):
                    c["name"] = corr_name
                if normalized.date:
                    if not c["last_email"] or normalized.date > c["last_email"]:
                        c["last_email"] = normalized.date
                c["interactions"].append({
                    "title": normalized.subject or "Email",
                    "date": normalized.date.isoformat() if normalized.date else None,
                    "summary": normalized.snippet or normalized.subject,
                    "type": "email",
                })

                # Detect scheduling patterns
                subj = (normalized.subject or "").lower()
                body_start = (normalized.body_plain or "")[:500].lower()
                sched_keywords = ["scheduled", "calendar", "invitation:", "meeting confirmed",
                                  "interview", "let's meet", "call scheduled", "booked"]
                if any(kw in subj or kw in body_start for kw in sched_keywords):
                    c["has_scheduling"] = True

        # Update email counts for known entities
        for entity in entities:
            email_count = session.query(SourceRecord).filter(
                SourceRecord.source_type == "gmail",
                SourceRecord.entity_id == entity.id,
            ).count()
            if email_count:
                profile_data = {}
                try:
                    profile_data = json.loads(entity.domains or "{}")
                except (json.JSONDecodeError, TypeError):
                    pass
                profile_data["email_count"] = email_count

                # Add email interactions to profile
                email_records = session.query(SourceRecord).filter(
                    SourceRecord.source_type == "gmail",
                    SourceRecord.entity_id == entity.id,
                ).order_by(SourceRecord.date.desc()).limit(10).all()

                email_interactions = []
                for rec in email_records:
                    email_interactions.append({
                        "title": rec.title or "Email",
                        "date": rec.date.isoformat() if rec.date else None,
                        "summary": rec.summary,
                        "type": "email",
                    })

                # Merge with existing interactions
                existing = profile_data.get("interactions", [])
                # Add email interactions that aren't already there
                existing_dates = {i.get("date") for i in existing}
                for ei in email_interactions:
                    if ei["date"] not in existing_dates:
                        existing.append(ei)

                profile_data["interactions"] = sorted(
                    existing,
                    key=lambda x: x.get("date") or "",
                    reverse=True,
                )[:20]

                # Update last_interaction if email is more recent
                if email_records and email_records[0].date:
                    last_meeting = profile_data.get("last_interaction")
                    last_email = email_records[0].date.isoformat()
                    if not last_meeting or last_email > last_meeting:
                        profile_data["last_interaction"] = last_email

                entity.domains = json.dumps(profile_data)

        # Create email-only profiles for frequent correspondents (2+ emails)
        for corr_email, corr_data in correspondents.items():
            if corr_data["email_count"] < 2:
                continue

            # Check if entity already exists (might have been created by email matching)
            exists = session.query(EntityRecord).filter(
                EntityRecord.entity_type == "person",
            ).all()
            already_exists = False
            for e in exists:
                entity_emails = [x.lower() for x in json.loads(e.emails or "[]")]
                if corr_email in entity_emails:
                    already_exists = True
                    break
            if already_exists:
                continue

            name = corr_data["name"] or corr_email.split("@")[0].replace(".", " ").title()
            entity = EntityRecord(
                name=name,
                entity_type="person",
                emails=json.dumps([corr_email]),
            )
            session.add(entity)
            session.flush()

            health = _determine_relationship_health(0, corr_data["last_email"])
            profile_data = {
                "meeting_count": 0,
                "email_count": corr_data["email_count"],
                "last_interaction": corr_data["last_email"].isoformat() if corr_data["last_email"] else None,
                "company": _infer_company_from_email(corr_email),
                "relationship_health": health,
                "interactions": sorted(
                    corr_data["interactions"],
                    key=lambda x: x.get("date") or "",
                    reverse=True,
                )[:20],
                "action_items": [],
                "action_items_count": 0,
                "updated_at": datetime.utcnow().isoformat(),
            }
            entity.domains = json.dumps(profile_data)
            email_profiles_created += 1
            logger.info("Created email-only profile: %s (%s)", name, corr_email)

        session.commit()
        logger.info("Gmail sync: %d emails synced, %d email-only profiles created",
                     emails_synced, email_profiles_created)

    except Exception:
        session.rollback()
        logger.exception("Gmail sync failed")
    finally:
        session.close()

    return {"emails_synced": emails_synced, "email_profiles_created": email_profiles_created}


def _extract_next_steps() -> list[dict]:
    """Extract next steps from recent scheduling emails and profile data."""
    session = get_session()
    next_steps = []

    try:
        # Look at recent emails for scheduling patterns
        recent = datetime.utcnow() - timedelta(days=14)
        email_records = session.query(SourceRecord).filter(
            SourceRecord.source_type == "gmail",
            SourceRecord.date >= recent,
        ).order_by(SourceRecord.date.desc()).limit(50).all()

        sched_keywords = ["scheduled", "invitation:", "interview", "calendar",
                          "meeting confirmed", "call scheduled", "booked"]

        for rec in email_records:
            subj = (rec.title or "").lower()
            if any(kw in subj for kw in sched_keywords):
                # Extract who it's with
                participants = json.loads(rec.participants or "[]")
                person_name = None
                for p in participants[:3]:
                    _, email = _parse_email_address(p)
                    if email:
                        entity = None
                        for e in session.query(EntityRecord).filter(
                            EntityRecord.entity_type == "person",
                        ).all():
                            if email.lower() in [x.lower() for x in json.loads(e.emails or "[]")]:
                                entity = e
                                break
                        if entity:
                            person_name = entity.name
                            break

                next_steps.append({
                    "type": "scheduled_meeting",
                    "title": rec.title,
                    "date": rec.date.isoformat() if rec.date else None,
                    "person": person_name,
                    "summary": f"Meeting scheduled: {rec.title}",
                })

        # Deduplicate by title
        seen = set()
        unique = []
        for ns in next_steps:
            key = ns["title"]
            if key not in seen:
                seen.add(key)
                unique.append(ns)
        return unique[:10]

    except Exception:
        logger.exception("Failed to extract next steps")
        return []
    finally:
        session.close()


async def _enrich_profiles_with_apollo() -> int:
    """Enrich person profiles via Apollo.io.

    All matches (direct and search) are presented as candidates on the
    Review page for user confirmation.  Nothing is auto-accepted so
    that photos and LinkedIn links are always verified by a human.
    """
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

            # Skip if already confirmed by the user
            if profile_data.get("linkedin_status") == "confirmed":
                continue

            # Skip contacts already pending review (candidates stored)
            if profile_data.get("linkedin_status") == "pending_review":
                continue

            # Skip contacts the user already dismissed
            if profile_data.get("linkedin_status") == "no_match":
                continue

            # Previously auto-matched profiles should be re-reviewed —
            # clear their stale data so the user can pick the right person.
            if profile_data.get("linkedin_status") == "matched":
                profile_data.pop("photo_url", None)
                profile_data.pop("linkedin_url", None)

            # Need at least some interactions
            if not profile_data.get("meeting_count") and not profile_data.get("email_count"):
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

        # Build contact list with all available data for best matching
        contacts = [
            {
                "email": info["email"],
                "name": info["name"],
                "company": info["company"],
            }
            for _, info in to_enrich
        ]

        results = await client.enrich_many(contacts, name_variants=_NAME_VARIANTS)

        for (entity, info), person in zip(to_enrich, results):
            try:
                profile_data = json.loads(entity.domains or "{}")
            except (json.JSONDecodeError, TypeError):
                continue

            # Build candidate list: start with the direct match (if any),
            # then add search results so the user can pick the right one.
            candidates: list[dict] = []

            direct_match = normalize_enrichment(person)
            if direct_match:
                # Include the direct match as the first (recommended) candidate
                candidate = normalize_candidate(person)
                candidate["recommended"] = True
                candidates.append(candidate)

            # Always search for additional candidates
            await asyncio.sleep(0.3)
            candidates_raw = await client.search_people(
                name=info["name"],
                organization_name=info["company"],
                per_page=5,
            )

            # Deduplicate search results against the direct match
            direct_linkedin = candidates[0].get("linkedin_url") if candidates else None
            for c in candidates_raw:
                if not (c.get("linkedin_url") or c.get("photo_url")):
                    continue
                # Skip if same LinkedIn profile as the direct match
                if direct_linkedin and c.get("linkedin_url") == direct_linkedin:
                    continue
                candidates.append(normalize_candidate(c))

            if candidates and direct_match:
                # Direct email match = high confidence → auto-confirm
                profile_data["photo_url"] = direct_match.get("photo_url", "")
                profile_data["linkedin_url"] = direct_match.get("linkedin_url", "")
                profile_data["title"] = direct_match.get("title", "")
                profile_data["headline"] = direct_match.get("headline", "")
                profile_data["seniority"] = direct_match.get("seniority", "")
                loc = ", ".join(filter(None, [
                    direct_match.get("city", ""),
                    direct_match.get("state", ""),
                ]))
                if loc:
                    profile_data["location"] = loc
                if direct_match.get("company_name") and not profile_data.get("company"):
                    profile_data["company"] = direct_match["company_name"]
                profile_data["company_industry"] = direct_match.get("company_industry", "")
                profile_data["company_size"] = direct_match.get("company_size")
                profile_data["company_linkedin"] = direct_match.get("company_linkedin", "")
                profile_data["linkedin_status"] = "confirmed"
                profile_data["linkedin_candidates"] = []
                logger.info(
                    "Apollo: auto-confirmed %s (direct email match, %d candidates)",
                    entity.name,
                    len(candidates),
                )
                enriched_count += 1
            elif candidates:
                # No direct match but have search candidates → needs review
                profile_data["linkedin_status"] = "pending_review"
                profile_data["linkedin_candidates"] = candidates
                logger.info(
                    "Apollo: %d candidates for %s – pending review",
                    len(candidates),
                    entity.name,
                )
                enriched_count += 1
            else:
                profile_data["linkedin_status"] = "no_match"
                profile_data["linkedin_candidates"] = []

            profile_data["apollo_enriched"] = True
            entity.domains = json.dumps(profile_data)

        session.commit()
        logger.info("Apollo enrichment: %d contacts processed", enriched_count)
    except Exception:
        session.rollback()
        logger.exception("Apollo enrichment failed")
    finally:
        session.close()

    return enriched_count


async def _re_enrich_confirmed_profiles() -> int:
    """Re-enrich confirmed profiles that are missing photo_url.

    Targets confirmed profiles without a working photo. Tries Apollo
    enrichment using linkedin_url when available, otherwise falls back
    to email + name lookup.
    """
    if not settings.apollo_api_key:
        return 0

    session = get_session()
    updated_count = 0
    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
            EntityRecord.domains.isnot(None),
        ).all()

        to_enrich: list[tuple[EntityRecord, dict]] = []
        for entity in entities:
            try:
                profile_data = json.loads(entity.domains or "{}")
            except (json.JSONDecodeError, TypeError):
                continue

            # Only target confirmed profiles missing photos
            if profile_data.get("linkedin_status") != "confirmed":
                continue
            if profile_data.get("photo_url"):
                continue  # already has a photo

            # Need at least a LinkedIn URL or email to look up
            emails = json.loads(entity.emails or "[]")
            if not profile_data.get("linkedin_url") and not emails:
                continue

            to_enrich.append((entity, profile_data))

        if not to_enrich:
            return 0

        logger.info(
            "Photo re-enrichment: %d confirmed profiles missing photos",
            len(to_enrich),
        )

        client = ApolloClient()

        for entity, profile_data in to_enrich:
            try:
                emails = json.loads(entity.emails or "[]")
                parts = entity.name.split(None, 1) if entity.name else []
                first = parts[0] if parts else None
                last = parts[1] if len(parts) > 1 else None

                person = await client.enrich_person(
                    email=emails[0] if emails else None,
                    first_name=first,
                    last_name=last,
                    linkedin_url=profile_data.get("linkedin_url"),
                    organization_name=profile_data.get("company"),
                )

                # If direct enrichment found no photo, try search as fallback
                if not person or not person.get("photo_url"):
                    await asyncio.sleep(0.3)
                    search_results = await client.search_people(
                        name=entity.name,
                        organization_name=profile_data.get("company"),
                        per_page=3,
                    )
                    for sr in search_results:
                        if sr.get("photo_url"):
                            if not person:
                                person = sr
                            else:
                                person["photo_url"] = sr["photo_url"]
                            break

                if not person:
                    await asyncio.sleep(0.3)
                    continue

                changed = False
                # Fill missing fields only
                for key in (
                    "photo_url", "title", "headline", "seniority",
                    "company_industry", "company_size", "company_linkedin",
                ):
                    value = person.get(key) or (
                        (person.get("organization") or {}).get(key)
                        if key.startswith("company_") else None
                    )
                    if value and not profile_data.get(key):
                        profile_data[key] = value
                        changed = True

                # Photo from top-level
                if person.get("photo_url") and not profile_data.get("photo_url"):
                    profile_data["photo_url"] = person["photo_url"]
                    changed = True

                # Location
                if not profile_data.get("location"):
                    loc = ", ".join(filter(None, [
                        person.get("city", ""),
                        person.get("state", ""),
                    ]))
                    if loc:
                        profile_data["location"] = loc
                        changed = True

                if changed:
                    entity.domains = json.dumps(profile_data)
                    updated_count += 1
                    logger.info(
                        "Photo re-enrichment: updated %s (id=%d)%s",
                        entity.name, entity.id,
                        " with photo" if profile_data.get("photo_url") else "",
                    )

                await asyncio.sleep(0.5)
            except Exception:
                logger.exception("Photo re-enrichment failed for %s", entity.name)
                continue

        if updated_count:
            session.commit()
            logger.info("Photo re-enrichment: updated %d profiles", updated_count)
    except Exception:
        session.rollback()
        logger.exception("Photo re-enrichment failed")
    finally:
        session.close()

    return updated_count


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

            # Skip entities without any interactions
            if not profile_data.get("meeting_count") and not profile_data.get("email_count"):
                continue

            # Filter out non-person entries (meeting rooms, system accounts)
            emails = json.loads(entity.emails or "[]")
            if _is_non_person(entity.name, emails[0] if emails else ""):
                continue
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
                # LinkedIn disambiguation
                "linkedin_status": profile_data.get("linkedin_status", ""),
                "linkedin_candidates": profile_data.get("linkedin_candidates", []),
                # Deep profile
                "deep_profile": profile_data.get("deep_profile", ""),
                "deep_profile_generated_at": profile_data.get(
                    "deep_profile_generated_at", ""
                ),
            }
            profiles.append(profile)

        # Deduplicate: merge profiles with the same normalized name
        deduped: dict[str, dict] = {}
        for profile in profiles:
            norm = _normalize_name(profile["name"])
            if norm in deduped:
                existing = deduped[norm]
                # Keep the one with more data (prefer enriched, more meetings)
                existing_score = (
                    (1 if existing.get("photo_url") else 0)
                    + (1 if existing.get("linkedin_url") else 0)
                    + existing.get("meeting_count", 0)
                )
                new_score = (
                    (1 if profile.get("photo_url") else 0)
                    + (1 if profile.get("linkedin_url") else 0)
                    + profile.get("meeting_count", 0)
                )
                if new_score > existing_score:
                    # Merge counts from existing into new
                    profile["meeting_count"] = max(
                        profile.get("meeting_count", 0),
                        existing.get("meeting_count", 0),
                    )
                    profile["email_count"] = max(
                        profile.get("email_count", 0),
                        existing.get("email_count", 0),
                    )
                    deduped[norm] = profile
                else:
                    # Merge counts from new into existing
                    existing["meeting_count"] = max(
                        existing.get("meeting_count", 0),
                        profile.get("meeting_count", 0),
                    )
                    existing["email_count"] = max(
                        existing.get("email_count", 0),
                        profile.get("email_count", 0),
                    )
            else:
                deduped[norm] = profile

        # Sort by last interaction (most recent first)
        result = list(deduped.values())
        result.sort(
            key=lambda p: p.get("last_interaction") or "",
            reverse=True,
        )
        return result
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


def repair_linkedin_status():
    """One-time repair: restore linkedin_status for profiles that were confirmed
    but had their status wiped by a sync bug.

    Detection heuristic (to avoid false positives from old auto-matches):
    - Profiles with linkedin_url + photo_url + title → were user-confirmed
      (auto-match alone would not populate all three).
    - Profiles with linkedin_url + only title OR only photo → likely auto-matched;
      queue for user review instead of auto-confirming.
    - Profiles with linkedin_url but no enrichment → queue for review.

    Acquires _sync_lock to prevent concurrent modification with background sync.
    """
    with _sync_lock:
        session = get_session()
        repaired = 0
        try:
            entities = session.query(EntityRecord).filter(
                EntityRecord.entity_type == "person",
                EntityRecord.domains.isnot(None),
            ).all()

            for entity in entities:
                try:
                    profile_data = json.loads(entity.domains or "{}")
                except (json.JSONDecodeError, TypeError):
                    continue

                # Already has a status — nothing to repair
                if profile_data.get("linkedin_status"):
                    continue

                has_linkedin = bool(profile_data.get("linkedin_url"))
                if not has_linkedin:
                    continue

                has_photo = bool(profile_data.get("photo_url"))
                has_title = bool(profile_data.get("title"))

                # Only auto-confirm if we have strong evidence of prior user
                # confirmation: both photo AND title present alongside linkedin_url.
                # A single field could come from an old auto-match.
                if has_photo and has_title:
                    profile_data["linkedin_status"] = "confirmed"
                    entity.domains = json.dumps(profile_data)
                    repaired += 1
                    logger.info(
                        "Repaired linkedin_status for %s (id=%d)",
                        entity.name, entity.id,
                    )
                else:
                    # Incomplete enrichment — queue for user review
                    profile_data["linkedin_status"] = "pending_review"
                    if not profile_data.get("linkedin_candidates"):
                        profile_data["linkedin_candidates"] = [
                            normalize_candidate_stub(
                                linkedin_url=profile_data["linkedin_url"],
                                name=entity.name,
                                title=profile_data.get("title", ""),
                                photo_url=profile_data.get("photo_url", ""),
                            ),
                        ]
                    entity.domains = json.dumps(profile_data)
                    repaired += 1
                    logger.info(
                        "Queued for review: %s (id=%d, incomplete enrichment)",
                        entity.name, entity.id,
                    )

            if repaired:
                session.commit()
                logger.info("LinkedIn status repair: fixed %d profiles", repaired)
        except Exception:
            session.rollback()
            logger.exception("LinkedIn status repair failed")
        finally:
            session.close()

    return repaired


def normalize_candidate_stub(
    linkedin_url: str,
    name: str,
    title: str = "",
    photo_url: str = "",
) -> dict:
    """Build a minimal candidate dict for the review UI from existing data."""
    return {
        "linkedin_url": linkedin_url,
        "name": name,
        "photo_url": photo_url,
        "title": title,
        "headline": "",
        "company_name": "",
        "seniority": "",
        "city": "",
        "state": "",
        "country": "",
        "company_industry": "",
        "company_size": None,
        "company_domain": "",
        "company_linkedin": "",
    }


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
