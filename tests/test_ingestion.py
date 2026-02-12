"""Tests for ingestion parsing (Fireflies + Gmail)."""

from __future__ import annotations

import json
from datetime import datetime

from app.ingest.fireflies_ingest import normalize_transcript, store_transcript
from app.ingest.gmail_ingest import normalize_email, store_email
from app.store.database import SourceRecord, get_session


class TestFirefliesIngestion:
    """Test Fireflies transcript parsing and storage."""

    def test_normalize_transcript_basic_fields(self, sample_fireflies_transcript):
        result = normalize_transcript(sample_fireflies_transcript)

        assert result.source_id == "ff-transcript-001"
        assert result.title == "Q1 Pipeline Review with Jane Doe"
        assert result.date is not None
        assert isinstance(result.date, datetime)
        assert result.duration_minutes == 30.0
        assert "me@mycompany.com" in result.participants
        assert "jane.doe@acmecorp.com" in result.participants

    def test_normalize_transcript_summary(self, sample_fireflies_transcript):
        result = normalize_transcript(sample_fireflies_transcript)

        assert result.summary is not None
        assert "Q1 pipeline" in result.summary

    def test_normalize_transcript_action_items(self, sample_fireflies_transcript):
        result = normalize_transcript(sample_fireflies_transcript)

        assert len(result.action_items) >= 2
        assert any("proposal" in item.lower() for item in result.action_items)
        assert any("Friday" in item for item in result.action_items)

    def test_normalize_transcript_sentences(self, sample_fireflies_transcript):
        result = normalize_transcript(sample_fireflies_transcript)

        assert len(result.sentences) == 4
        assert result.sentences[0].speaker == "Me"
        assert result.sentences[1].speaker == "Jane Doe"
        assert "concern" in result.sentences[1].text.lower()

    def test_normalize_transcript_date_epoch_ms(self):
        raw = {
            "id": "test-epoch",
            "date": 1707500000000,  # epoch milliseconds
            "summary": {},
            "sentences": [],
        }
        result = normalize_transcript(raw)
        assert result.date is not None
        assert result.date.year >= 2024

    def test_normalize_transcript_missing_fields(self):
        raw = {"id": "test-minimal"}
        result = normalize_transcript(raw)
        assert result.source_id == "test-minimal"
        assert result.title is None
        assert result.participants == []
        assert result.sentences == []

    def test_store_transcript(self, sample_fireflies_transcript):
        normalized = normalize_transcript(sample_fireflies_transcript)
        record = store_transcript(normalized)

        assert record.id is not None
        assert record.source_type == "fireflies"
        assert record.source_id == "ff-transcript-001"
        assert record.summary is not None

    def test_store_transcript_idempotent(self, sample_fireflies_transcript):
        """Storing the same transcript twice should update, not duplicate."""
        normalized = normalize_transcript(sample_fireflies_transcript)
        record1 = store_transcript(normalized)
        record2 = store_transcript(normalized)

        assert record1.source_id == record2.source_id
        session = get_session("sqlite:///./test_briefing_engine.db")
        count = session.query(SourceRecord).filter_by(source_id="ff-transcript-001").count()
        assert count == 1
        session.close()


class TestGmailIngestion:
    """Test Gmail message parsing and storage."""

    def test_normalize_email_basic_fields(self, sample_gmail_message):
        result = normalize_email(sample_gmail_message)

        assert result.source_id == "gmail-msg-001"
        assert result.thread_id == "gmail-thread-001"
        assert result.subject == "Re: Phase 2 Proposal - Updated Timeline"
        assert result.from_address == "me@mycompany.com"
        assert "jane.doe@acmecorp.com" in result.to_addresses

    def test_normalize_email_body(self, sample_gmail_message):
        result = normalize_email(sample_gmail_message)

        assert result.body_plain is not None
        assert "updated proposal" in result.body_plain.lower()
        assert "Phase 2" in result.body_plain

    def test_normalize_email_date(self, sample_gmail_message):
        result = normalize_email(sample_gmail_message)

        assert result.date is not None
        assert result.date.year == 2026
        assert result.date.month == 2

    def test_normalize_email_missing_body(self):
        raw = {
            "id": "no-body",
            "payload": {
                "headers": [
                    {"name": "Subject", "value": "Test"},
                    {"name": "From", "value": "test@test.com"},
                ],
                "body": {},
            },
        }
        result = normalize_email(raw)
        assert result.source_id == "no-body"
        assert result.body_plain == ""

    def test_store_email(self, sample_gmail_message):
        normalized = normalize_email(sample_gmail_message)
        record = store_email(normalized)

        assert record.id is not None
        assert record.source_type == "gmail"
        assert record.source_id == "gmail-msg-001"

    def test_store_email_idempotent(self, sample_gmail_message):
        """Storing the same email twice should update, not duplicate."""
        normalized = normalize_email(sample_gmail_message)
        store_email(normalized)
        store_email(normalized)

        session = get_session("sqlite:///./test_briefing_engine.db")
        count = session.query(SourceRecord).filter_by(source_id="gmail-msg-001").count()
        assert count == 1
        session.close()
