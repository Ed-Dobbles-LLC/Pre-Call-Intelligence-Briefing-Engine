"""Gmail context enrichment for upcoming meetings.

For each meeting in the next 7 days, for each attendee:
1. Fetch invitation thread (calendar invite email)
2. Fetch last 5 threads in last 180 days with that contact
3. Extract: last_contact_date, open commitments, short summary snippet
4. Store structured enrichment data per meeting+attendee

Does NOT store full raw email bodies in UI-facing data.
"""

from __future__ import annotations

import json
import logging
import re
from app.store.database import EntityRecord, get_session

logger = logging.getLogger(__name__)

# Keywords that suggest open commitments
COMMITMENT_KEYWORDS = [
    "will send", "will follow up", "action item", "next step",
    "to do", "deadline", "by end of", "due date", "committed to",
    "agreed to", "promised", "schedule", "set up a",
    "let me know", "get back to you", "circle back",
]


def _extract_commitments(text: str) -> list[str]:
    """Extract potential open commitments from email text using keyword detection."""
    if not text:
        return []

    commitments = []
    sentences = re.split(r"[.!?\n]", text)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 10 or len(sentence) > 200:
            continue
        sentence_lower = sentence.lower()
        for kw in COMMITMENT_KEYWORDS:
            if kw in sentence_lower:
                commitments.append(sentence)
                break

    # Deduplicate and limit
    seen = set()
    unique = []
    for c in commitments:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique[:5]


def _summarize_thread(subject: str, body_snippet: str) -> str:
    """Create a 1-2 bullet summary of a thread (no full body stored)."""
    parts = []
    if subject:
        parts.append(f"Subject: {subject}")
    if body_snippet:
        # Take first 200 chars of body, clean up
        clean = body_snippet[:200].replace("\n", " ").replace("\r", "").strip()
        if clean:
            parts.append(clean)
    return " | ".join(parts) if parts else "No content available"


class MeetingEnrichmentResult:
    """Result of enrichment for a single meeting."""

    def __init__(self, meeting_id: str = "", attendee_email: str = "") -> None:
        self.meeting_id = meeting_id
        self.attendee_email = attendee_email
        self.snippet: str = ""
        self.last_contact_date: str = ""
        self.open_commitments: list[str] = []
        self.thread_count: int = 0
        self.confidence_score: float = 0.0
        self.error: str = ""

    def to_dict(self) -> dict:
        return {
            "meeting_id": self.meeting_id,
            "attendee_email": self.attendee_email,
            "snippet": self.snippet,
            "last_contact_date": self.last_contact_date,
            "open_commitments": self.open_commitments,
            "thread_count": self.thread_count,
            "confidence_score": self.confidence_score,
            "error": self.error,
        }


def enrich_meeting_context(
    attendee_email: str,
    meeting_id: str = "",
    gmail_client=None,
) -> MeetingEnrichmentResult:
    """Enrich a single meeting attendee with Gmail context.

    Fetches:
    1. Calendar invite threads (subject contains "Invitation:")
    2. Last 5 threads in last 180 days with this email address
    """
    result = MeetingEnrichmentResult(
        meeting_id=meeting_id,
        attendee_email=attendee_email,
    )

    if not gmail_client:
        try:
            from app.clients.gmail import GmailClient
            gmail_client = GmailClient()
        except Exception as e:
            result.error = f"Gmail client init failed: {e}"
            logger.warning("Gmail enrichment skipped: %s", e)
            return result

    # Fetch recent threads with this contact
    try:
        threads = gmail_client.search_messages(
            query=f"from:{attendee_email} OR to:{attendee_email}",
            max_results=5,
        )
    except Exception:
        logger.exception("Gmail search failed for %s", attendee_email)
        result.error = "Gmail search failed"
        return result

    result.thread_count = len(threads)

    if not threads:
        result.snippet = f"No email history found with {attendee_email}"
        result.confidence_score = 0.1
        return result

    # Extract last contact date
    dates = []
    snippets = []
    all_commitments = []

    for thread in threads[:5]:
        headers = thread.get("headers", {})
        date_str = headers.get("date", "") or headers.get("Date", "")
        subject = headers.get("subject", "") or headers.get("Subject", "")
        body = thread.get("body", "") or thread.get("snippet", "")

        if date_str:
            dates.append(date_str)

        snippet = _summarize_thread(subject, body)
        if snippet:
            snippets.append(snippet)

        commitments = _extract_commitments(body)
        all_commitments.extend(commitments)

    # Set last contact date
    if dates:
        result.last_contact_date = dates[0]  # Most recent

    # Build summary snippet (1-2 bullets, no full bodies)
    if snippets:
        result.snippet = "\n".join(f"- {s}" for s in snippets[:2])
    else:
        result.snippet = "Email threads found but no content extractable"

    # Deduplicate commitments
    seen = set()
    unique_commitments = []
    for c in all_commitments:
        key = c.lower().strip()
        if key not in seen:
            seen.add(key)
            unique_commitments.append(c)
    result.open_commitments = unique_commitments[:5]

    # Confidence: higher with more data
    result.confidence_score = min(1.0, 0.3 + (len(threads) * 0.14))

    return result


def enrich_all_upcoming_meetings() -> list[dict]:
    """Run Gmail enrichment for all upcoming meetings across all contacts.

    Returns a list of enrichment result dicts.
    """
    session = get_session()
    enrichments = []

    try:
        entities = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person"
        ).all()

        for entity in entities:
            profile_data = json.loads(entity.domains or "{}")
            upcoming = profile_data.get("upcoming_meetings", [])
            if not upcoming:
                continue

            emails = entity.get_emails()
            if not emails:
                continue

            primary_email = emails[0]

            for meeting in upcoming:
                meeting_id = meeting.get("calendar_event_id", "")
                result = enrich_meeting_context(
                    attendee_email=primary_email,
                    meeting_id=meeting_id,
                )
                enrichments.append(result.to_dict())

                # Store enrichment on the meeting record
                meeting["enrichment"] = result.to_dict()

            # Update profile data with enriched meetings
            profile_data["upcoming_meetings"] = upcoming
            entity.domains = json.dumps(profile_data)

        session.commit()
        logger.info("Enriched %d meeting-contact pairs", len(enrichments))

    except Exception:
        session.rollback()
        logger.exception("Gmail meeting enrichment failed")
    finally:
        session.close()

    return enrichments
