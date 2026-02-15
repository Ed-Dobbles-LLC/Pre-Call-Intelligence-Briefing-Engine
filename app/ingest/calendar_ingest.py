"""Google Calendar ingest job.

Fetches upcoming meetings (next 7 days) and:
1. Normalizes each event
2. Matches attendees to existing contacts (by email, alias, or name+domain)
3. Creates contact stubs for unknown attendees
4. Stores meeting records and contact-meeting associations
5. Tracks match/miss reasons for debugging

Produces structured meeting data stored in EntityRecord.domains JSON.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

from app.clients.calendar import CalendarClient, normalize_event_for_storage
from app.store.database import EntityRecord, get_session

logger = logging.getLogger(__name__)


class CalendarIngestResult:
    """Result of a calendar ingest run."""

    def __init__(self) -> None:
        self.events_fetched: int = 0
        self.matched_contacts: int = 0
        self.created_stubs: int = 0
        self.unmatched_attendees: list[dict] = []
        self.meetings: list[dict] = []
        self.errors: list[str] = []

    def to_dict(self) -> dict:
        return {
            "events_fetched": self.events_fetched,
            "matched_contacts": self.matched_contacts,
            "created_stubs": self.created_stubs,
            "unmatched_attendees": self.unmatched_attendees,
            "meetings_stored": len(self.meetings),
            "errors": self.errors,
        }


def _build_email_index(session) -> dict[str, EntityRecord]:
    """Build a lowercase email → EntityRecord lookup index."""
    index: dict[str, EntityRecord] = {}
    entities = session.query(EntityRecord).filter(
        EntityRecord.entity_type == "person"
    ).all()

    for entity in entities:
        emails = entity.get_emails()
        for email in emails:
            index[email.lower().strip()] = entity

        # Also index aliases
        aliases = entity.get_aliases()
        for alias in aliases:
            if "@" in alias:
                index[alias.lower().strip()] = entity

    return index


def _match_attendee(
    attendee: dict,
    email_index: dict[str, EntityRecord],
) -> tuple[EntityRecord | None, str]:
    """Try to match an attendee to an existing contact.

    Returns (entity, match_reason) or (None, miss_reason).
    """
    email = attendee.get("email", "").lower().strip()
    if not email:
        return None, "no_email"

    # Primary: exact email match
    if email in email_index:
        return email_index[email], "email_match"

    # Secondary: alias match (already in email_index via _build_email_index)
    # Tertiary: name + domain match
    name = attendee.get("name", "").lower()
    if name and "@" in email:
        domain = email.split("@")[1]
        for entity_email, entity in email_index.items():
            if entity.name.lower() == name:
                return entity, "name_match"
            # Check if same domain
            if "@" in entity_email and entity_email.split("@")[1] == domain:
                if _fuzzy_name_match(name, entity.name.lower()):
                    return entity, "name_domain_match"

    return None, "no_match"


def _fuzzy_name_match(name_a: str, name_b: str) -> bool:
    """Basic name matching: first+last or last name match."""
    parts_a = name_a.strip().split()
    parts_b = name_b.strip().split()
    if not parts_a or not parts_b:
        return False
    # Exact match
    if name_a == name_b:
        return True
    # Last name match + first initial
    if len(parts_a) >= 2 and len(parts_b) >= 2:
        if parts_a[-1] == parts_b[-1] and parts_a[0][0] == parts_b[0][0]:
            return True
    return False


def _create_contact_stub(
    attendee: dict,
    session,
) -> EntityRecord:
    """Create a minimal contact stub for an unknown attendee."""
    email = attendee.get("email", "")
    name = attendee.get("name", "") or email.split("@")[0].replace(".", " ").title()

    entity = EntityRecord(name=name, entity_type="person")
    entity.set_emails([email])

    profile_data = {
        "source": "calendar_ingest",
        "created_from": "calendar_attendee",
        "research_status": "QUEUED",
        "linkedin_status": "not_searched",
    }
    entity.domains = json.dumps(profile_data)

    session.add(entity)
    session.flush()  # Get the ID without committing
    return entity


def _attach_meeting_to_contact(
    entity: EntityRecord,
    meeting: dict,
    match_reason: str,
) -> None:
    """Attach a meeting record to a contact's profile data."""
    profile_data = json.loads(entity.domains or "{}")

    upcoming_meetings = profile_data.get("upcoming_meetings", [])

    # Deduplicate by calendar_event_id
    existing_ids = {m.get("calendar_event_id") for m in upcoming_meetings}
    if meeting.get("calendar_event_id") in existing_ids:
        return

    meeting_record = {
        **meeting,
        "match_reason": match_reason,
        "attached_at": datetime.utcnow().isoformat(),
    }
    upcoming_meetings.append(meeting_record)

    profile_data["upcoming_meetings"] = upcoming_meetings
    profile_data["has_upcoming_meeting"] = True
    profile_data["next_meeting_title"] = upcoming_meetings[0].get("title", "")
    profile_data["next_meeting_time"] = upcoming_meetings[0].get("start_time", "")

    entity.domains = json.dumps(profile_data)


def run_calendar_ingest(days: int = 7) -> CalendarIngestResult:
    """Main calendar ingest job.

    1. Fetch events from Google Calendar (next N days)
    2. Match attendees to existing contacts
    3. Create stubs for unknown attendees
    4. Store meeting associations
    """
    result = CalendarIngestResult()

    try:
        client = CalendarClient()
        events = client.fetch_upcoming_events(days=days)
    except RuntimeError as e:
        result.errors.append(str(e))
        logger.warning("Calendar ingest skipped: %s", e)
        return result
    except Exception:
        logger.exception("Calendar fetch failed")
        result.errors.append("Calendar API fetch failed")
        return result

    result.events_fetched = len(events)
    if not events:
        logger.info("No upcoming calendar events found")
        return result

    session = get_session()
    try:
        email_index = _build_email_index(session)
        logger.info(
            "Calendar ingest: %d events, %d known contacts in index",
            len(events), len(email_index),
        )

        for event in events:
            meeting = normalize_event_for_storage(event)
            result.meetings.append(meeting)

            for attendee in event.get("attendees", []):
                # Skip self (the calendar owner)
                if attendee.get("self"):
                    continue

                entity, reason = _match_attendee(attendee, email_index)

                if entity:
                    result.matched_contacts += 1
                    _attach_meeting_to_contact(entity, meeting, reason)
                    logger.debug(
                        "Matched %s to contact %s (%s)",
                        attendee.get("email"), entity.name, reason,
                    )
                else:
                    # Create stub for unknown attendee
                    stub = _create_contact_stub(attendee, session)
                    _attach_meeting_to_contact(stub, meeting, "new_stub")
                    result.created_stubs += 1
                    # Add to index for dedup within this run
                    email_index[attendee["email"].lower()] = stub

                    result.unmatched_attendees.append({
                        "email": attendee.get("email", ""),
                        "name": attendee.get("name", ""),
                        "event_title": event.get("title", ""),
                        "reason": reason,
                    })

                    logger.info(
                        "Created stub for unknown attendee: %s (%s)",
                        attendee.get("email"), attendee.get("name"),
                    )

        session.commit()
        logger.info(
            "Calendar ingest complete: %d events, %d matched, %d stubs created",
            result.events_fetched, result.matched_contacts, result.created_stubs,
        )

    except Exception:
        session.rollback()
        logger.exception("Calendar ingest failed during processing")
        result.errors.append("Processing failed — rolled back")
    finally:
        session.close()

    return result
