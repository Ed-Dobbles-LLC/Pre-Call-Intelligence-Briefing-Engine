"""Tests for Google Calendar ingest and Gmail meeting enrichment."""

from __future__ import annotations

import json
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

from unittest.mock import MagicMock, patch

from app.clients.calendar import CalendarClient, normalize_event_for_storage
from app.ingest.calendar_ingest import (
    CalendarIngestResult,
    _attach_meeting_to_contact,
    _create_contact_stub,
    _fuzzy_name_match,
    _match_attendee,
    run_calendar_ingest,
)
from app.ingest.gmail_meeting_enrichment import (
    MeetingEnrichmentResult,
    _extract_commitments,
    _summarize_thread,
    enrich_meeting_context,
)
from app.store.database import EntityRecord, get_session, init_db

# Ensure test DB exists
init_db("sqlite:///./test_briefing_engine.db")


class TestCalendarEventNormalization:
    def test_normalize_basic_event(self):
        raw = {
            "id": "evt123",
            "summary": "Q1 Review",
            "start": {"dateTime": "2026-02-16T10:00:00Z"},
            "end": {"dateTime": "2026-02-16T11:00:00Z"},
            "attendees": [
                {"email": "andy@acme.com", "displayName": "Andy Sweet"},
                {"email": "me@company.com", "self": True},
            ],
            "organizer": {"email": "me@company.com", "displayName": "Me"},
            "status": "confirmed",
        }
        client = CalendarClient()
        normalized = client._normalize_event(raw)
        assert normalized["id"] == "evt123"
        assert normalized["title"] == "Q1 Review"
        assert len(normalized["attendees"]) == 2
        assert normalized["attendees"][0]["email"] == "andy@acme.com"

    def test_normalize_for_storage(self):
        event = {
            "id": "evt123",
            "title": "Q1 Review",
            "start_time": "2026-02-16T10:00:00Z",
            "end_time": "2026-02-16T11:00:00Z",
            "attendees": [
                {"email": "andy@acme.com", "name": "Andy Sweet"},
            ],
            "organizer_email": "me@company.com",
            "conference_link": "https://meet.google.com/xyz",
        }
        stored = normalize_event_for_storage(event)
        assert stored["calendar_event_id"] == "evt123"
        assert stored["status"] == "UPCOMING"
        assert "andy@acme.com" in stored["attendee_emails"]


class TestAttendeeMatching:
    def test_email_exact_match(self):
        entity = EntityRecord(name="Andy Sweet", entity_type="person")
        entity.set_emails(["andy@acme.com"])
        index = {"andy@acme.com": entity}

        matched, reason = _match_attendee(
            {"email": "andy@acme.com", "name": "Andy Sweet"},
            index,
        )
        assert matched is not None
        assert matched.name == "Andy Sweet"
        assert reason == "email_match"

    def test_no_match_returns_none(self):
        index = {}
        matched, reason = _match_attendee(
            {"email": "unknown@other.com", "name": "Unknown"},
            index,
        )
        assert matched is None
        assert reason == "no_match"

    def test_alias_email_match(self):
        """Alias emails should match too."""
        entity = EntityRecord(name="Andy Sweet", entity_type="person")
        entity.set_emails(["andy@acme.com"])
        entity.set_aliases(["andrew@acme.com"])
        index = {
            "andy@acme.com": entity,
            "andrew@acme.com": entity,
        }

        matched, reason = _match_attendee(
            {"email": "andrew@acme.com", "name": "Andrew Sweet"},
            index,
        )
        assert matched is not None
        assert reason == "email_match"

    def test_name_match(self):
        """Same name should match even with different email."""
        entity = EntityRecord(name="Andy Sweet", entity_type="person")
        entity.set_emails(["a.sweet@acme.com"])
        index = {"a.sweet@acme.com": entity}

        matched, reason = _match_attendee(
            {"email": "andy.sweet@acme.com", "name": "andy sweet"},
            index,
        )
        assert matched is not None
        assert reason == "name_match"

    def test_name_domain_match(self):
        """Similar name + same domain should match."""
        entity = EntityRecord(name="Andrew Sweet", entity_type="person")
        entity.set_emails(["a.sweet@acme.com"])
        index = {"a.sweet@acme.com": entity}

        matched, reason = _match_attendee(
            {"email": "andy.sweet@acme.com", "name": "andy sweet"},
            index,
        )
        assert matched is not None
        assert reason == "name_domain_match"

    def test_no_email_returns_no_email(self):
        matched, reason = _match_attendee({"email": "", "name": "Test"}, {})
        assert matched is None
        assert reason == "no_email"


class TestFuzzyNameMatch:
    def test_exact_match(self):
        assert _fuzzy_name_match("andy sweet", "andy sweet")

    def test_last_name_first_initial(self):
        assert _fuzzy_name_match("andrew sweet", "andy sweet")

    def test_different_names(self):
        assert not _fuzzy_name_match("john doe", "jane smith")

    def test_empty_names(self):
        assert not _fuzzy_name_match("", "")


class TestContactStubCreation:
    def test_creates_stub_with_email(self):
        session = get_session("sqlite:///./test_briefing_engine.db")
        try:
            stub = _create_contact_stub(
                {"email": "new@acme.com", "name": "New Person"},
                session,
            )
            assert stub.name == "New Person"
            assert "new@acme.com" in stub.get_emails()
            profile = json.loads(stub.domains)
            assert profile["source"] == "calendar_ingest"
            assert profile["research_status"] == "QUEUED"
        finally:
            session.rollback()
            session.close()

    def test_creates_stub_without_name(self):
        """When no display name, derives from email."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        try:
            stub = _create_contact_stub(
                {"email": "john.doe@acme.com", "name": ""},
                session,
            )
            assert stub.name == "John Doe"
        finally:
            session.rollback()
            session.close()


class TestMeetingAttachment:
    def test_attaches_meeting_to_contact(self):
        entity = EntityRecord(name="Test", entity_type="person")
        entity.domains = json.dumps({})

        meeting = {
            "calendar_event_id": "evt123",
            "title": "Q1 Review",
            "start_time": "2026-02-16T10:00:00Z",
        }
        _attach_meeting_to_contact(entity, meeting, "email_match")

        profile = json.loads(entity.domains)
        assert profile["has_upcoming_meeting"] is True
        assert profile["next_meeting_title"] == "Q1 Review"
        assert len(profile["upcoming_meetings"]) == 1
        assert profile["upcoming_meetings"][0]["match_reason"] == "email_match"

    def test_deduplicates_by_event_id(self):
        entity = EntityRecord(name="Test", entity_type="person")
        entity.domains = json.dumps({})

        meeting = {"calendar_event_id": "evt123", "title": "Q1 Review"}
        _attach_meeting_to_contact(entity, meeting, "email_match")
        _attach_meeting_to_contact(entity, meeting, "email_match")

        profile = json.loads(entity.domains)
        assert len(profile["upcoming_meetings"]) == 1

    def test_recurring_event_attaches(self):
        """Recurring events with different IDs should both attach."""
        entity = EntityRecord(name="Test", entity_type="person")
        entity.domains = json.dumps({})

        meeting1 = {"calendar_event_id": "evt_recur_1", "title": "Weekly Sync"}
        meeting2 = {"calendar_event_id": "evt_recur_2", "title": "Weekly Sync"}
        _attach_meeting_to_contact(entity, meeting1, "email_match")
        _attach_meeting_to_contact(entity, meeting2, "email_match")

        profile = json.loads(entity.domains)
        assert len(profile["upcoming_meetings"]) == 2


class TestCalendarIngestResult:
    def test_to_dict(self):
        result = CalendarIngestResult()
        result.events_fetched = 5
        result.matched_contacts = 3
        result.created_stubs = 2
        d = result.to_dict()
        assert d["events_fetched"] == 5
        assert d["matched_contacts"] == 3
        assert d["created_stubs"] == 2


class TestCalendarIngestNoCredentials:
    def test_graceful_failure_without_credentials(self):
        """Should return error, not crash, when Calendar not configured."""
        with patch(
            "app.ingest.calendar_ingest.CalendarClient",
            side_effect=RuntimeError("Not configured"),
        ):
            result = run_calendar_ingest()
            assert len(result.errors) > 0
            assert result.events_fetched == 0


# ---------------------------------------------------------------------------
# Gmail Meeting Enrichment Tests
# ---------------------------------------------------------------------------


class TestCommitmentExtraction:
    def test_extracts_will_send(self):
        text = "I will send you the proposal by Friday."
        commitments = _extract_commitments(text)
        assert len(commitments) >= 1
        assert any("will send" in c.lower() for c in commitments)

    def test_extracts_action_item(self):
        text = "Action item: review the deck before Monday."
        commitments = _extract_commitments(text)
        assert len(commitments) >= 1

    def test_empty_text(self):
        assert _extract_commitments("") == []

    def test_no_commitments(self):
        text = "Thanks for the nice weather today."
        assert _extract_commitments(text) == []

    def test_deduplicates(self):
        text = "I will send the report. I will send the report."
        commitments = _extract_commitments(text)
        assert len(commitments) == 1

    def test_limits_to_5(self):
        text = ". ".join(
            [f"Will follow up on item {i}" for i in range(10)]
        )
        commitments = _extract_commitments(text)
        assert len(commitments) <= 5


class TestThreadSummary:
    def test_basic_summary(self):
        summary = _summarize_thread("Re: Proposal", "Thanks for the quick turnaround")
        assert "Proposal" in summary
        assert "Thanks" in summary

    def test_empty_inputs(self):
        summary = _summarize_thread("", "")
        assert "No content" in summary


class TestMeetingEnrichmentResult:
    def test_to_dict(self):
        result = MeetingEnrichmentResult(meeting_id="m1", attendee_email="a@b.com")
        result.snippet = "test snippet"
        result.confidence_score = 0.5
        d = result.to_dict()
        assert d["meeting_id"] == "m1"
        assert d["snippet"] == "test snippet"


class TestEnrichMeetingContext:
    def test_no_gmail_client_returns_error(self):
        """When Gmail client can't be initialized, return error."""
        with patch(
            "app.clients.gmail.GmailClient",
            side_effect=RuntimeError("No Gmail"),
        ):
            result = enrich_meeting_context("test@example.com", "m1")
            assert result.error != ""

    def test_with_mock_threads(self):
        """With mock Gmail threads, should extract data."""
        mock_client = MagicMock()
        mock_client.search_messages.return_value = [
            {
                "headers": {"date": "2026-02-10", "subject": "Re: Proposal"},
                "body": "I will send the updated deck by Friday.",
            },
            {
                "headers": {"date": "2026-02-08", "subject": "Meeting notes"},
                "body": "Great discussion on the roadmap.",
            },
        ]
        result = enrich_meeting_context(
            "test@example.com", "m1", gmail_client=mock_client
        )
        assert result.thread_count == 2
        assert result.last_contact_date == "2026-02-10"
        assert result.snippet != ""
        assert result.confidence_score > 0.3

    def test_no_threads_returns_low_confidence(self):
        mock_client = MagicMock()
        mock_client.search_messages.return_value = []
        result = enrich_meeting_context(
            "test@example.com", "m1", gmail_client=mock_client
        )
        assert result.thread_count == 0
        assert result.confidence_score == 0.1
        assert "No email history" in result.snippet
