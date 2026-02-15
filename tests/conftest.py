"""Shared test fixtures."""

from __future__ import annotations

import os
from datetime import datetime, timedelta

import pytest

# Force SQLite in-memory for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""  # disable auth for tests

from app.store.database import Base, EntityRecord, get_engine, get_session


@pytest.fixture(autouse=True)
def setup_test_db():
    """Create a fresh in-memory database for each test."""
    engine = get_engine("sqlite:///./test_briefing_engine.db")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield
    Base.metadata.drop_all(engine)


@pytest.fixture
def db_session():
    """Provide a database session for tests."""
    session = get_session("sqlite:///./test_briefing_engine.db")
    yield session
    session.close()


@pytest.fixture
def sample_fireflies_transcript() -> dict:
    """A realistic Fireflies transcript payload."""
    return {
        "id": "ff-transcript-001",
        "title": "Q1 Pipeline Review with Jane Doe",
        "date": int((datetime.utcnow() - timedelta(days=5)).timestamp() * 1000),
        "duration": 1800,  # 30 minutes in seconds
        "organizer_email": "me@mycompany.com",
        "participants": ["me@mycompany.com", "jane.doe@acmecorp.com"],
        "summary": {
            "overview": "Discussed Q1 pipeline targets. Jane expressed concern about timeline for Phase 2. Agreed to send updated proposal by Friday.",
            "shorthand_bullet": "- Q1 pipeline review\n- Phase 2 timeline concern\n- Proposal due Friday",
            "action_items": "- Send updated proposal by Friday\n- Schedule follow-up for next Tuesday\n- Jane to confirm budget approval internally",
        },
        "sentences": [
            {
                "speaker_name": "Me",
                "text": "Let's review where we stand on the Q1 pipeline.",
                "start_time": 0,
                "end_time": 3.5,
            },
            {
                "speaker_name": "Jane Doe",
                "text": "Sure. I have some concerns about the Phase 2 timeline.",
                "start_time": 4.0,
                "end_time": 7.2,
            },
            {
                "speaker_name": "Jane Doe",
                "text": "The budget approval process on our end is taking longer than expected.",
                "start_time": 7.5,
                "end_time": 11.0,
            },
            {
                "speaker_name": "Me",
                "text": "I understand. Let me send you an updated proposal by Friday that addresses the timeline.",
                "start_time": 11.5,
                "end_time": 15.0,
            },
        ],
    }


@pytest.fixture
def sample_gmail_message() -> dict:
    """A realistic Gmail API message payload."""
    import base64

    body_text = (
        "Hi Jane,\n\n"
        "Following up on our call yesterday. As discussed, I'm attaching the updated proposal "
        "for Phase 2. Please review and let me know if the revised timeline works for your team.\n\n"
        "Key changes:\n"
        "- Extended Phase 2 deadline by 2 weeks\n"
        "- Added budget breakdown per milestone\n\n"
        "Looking forward to your feedback.\n\n"
        "Best,\nMe"
    )

    return {
        "id": "gmail-msg-001",
        "threadId": "gmail-thread-001",
        "labelIds": ["SENT", "IMPORTANT"],
        "snippet": "Following up on our call yesterday. As discussed...",
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "me@mycompany.com"},
                {"name": "To", "value": "jane.doe@acmecorp.com"},
                {"name": "Cc", "value": ""},
                {"name": "Subject", "value": "Re: Phase 2 Proposal - Updated Timeline"},
                {"name": "Date", "value": "Mon, 10 Feb 2026 09:30:00 -0500"},
            ],
            "body": {
                "data": base64.urlsafe_b64encode(body_text.encode()).decode(),
            },
        },
    }


@pytest.fixture
def populated_db(db_session, sample_fireflies_transcript, sample_gmail_message):
    """Populate DB with sample data and return the entity."""
    from app.ingest.fireflies_ingest import normalize_transcript, store_transcript
    from app.ingest.gmail_ingest import normalize_email, store_email

    # Create entity
    entity = EntityRecord(name="Jane Doe", entity_type="person")
    entity.set_emails(["jane.doe@acmecorp.com"])
    entity.set_aliases(["jane doe", "jane"])
    db_session.add(entity)
    db_session.commit()
    db_session.refresh(entity)

    # Store transcript
    norm_t = normalize_transcript(sample_fireflies_transcript)
    store_transcript(norm_t, entity_id=entity.id)

    # Store email
    norm_e = normalize_email(sample_gmail_message)
    store_email(norm_e, entity_id=entity.id)

    return entity
