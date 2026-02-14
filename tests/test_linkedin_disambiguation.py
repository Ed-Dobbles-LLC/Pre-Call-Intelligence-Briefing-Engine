"""Tests for LinkedIn profile disambiguation endpoints and repair logic."""

from __future__ import annotations

import json
import os

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""  # disable auth for tests
os.environ["APOLLO_API_KEY"] = ""  # disable Apollo for unit tests

from fastapi.testclient import TestClient

from app.api import app
from app.store.database import EntityRecord, get_session

client = TestClient(app)


def _create_profile(
    name: str = "Jane Doe",
    email: str = "jane@acme.com",
    linkedin_status: str = "",
    linkedin_url: str = "",
    photo_url: str = "",
    title: str = "",
    linkedin_candidates: list | None = None,
    meeting_count: int = 3,
) -> int:
    """Helper: create a person entity with profile data and return its id."""
    session = get_session("sqlite:///./test_briefing_engine.db")
    entity = EntityRecord(name=name, entity_type="person")
    entity.set_emails([email])

    profile_data = {
        "meeting_count": meeting_count,
        "email_count": 1,
        "company": "Acme Corp",
        "relationship_health": "active",
        "interactions": [],
        "action_items": [],
        "action_items_count": 0,
    }
    if linkedin_status:
        profile_data["linkedin_status"] = linkedin_status
    if linkedin_url:
        profile_data["linkedin_url"] = linkedin_url
    if photo_url:
        profile_data["photo_url"] = photo_url
    if title:
        profile_data["title"] = title
    if linkedin_candidates is not None:
        profile_data["linkedin_candidates"] = linkedin_candidates

    entity.domains = json.dumps(profile_data)
    session.add(entity)
    session.commit()
    entity_id = entity.id
    session.close()
    return entity_id


SAMPLE_CANDIDATES = [
    {
        "name": "Jane Doe",
        "photo_url": "https://example.com/photo1.jpg",
        "linkedin_url": "https://linkedin.com/in/janedoe",
        "title": "VP Engineering",
        "headline": "VP Engineering at Acme",
        "company_name": "Acme Corp",
        "seniority": "vp",
        "city": "San Francisco",
        "state": "CA",
        "country": "US",
        "company_industry": "Technology",
        "company_size": 500,
        "company_domain": "acme.com",
        "company_linkedin": "https://linkedin.com/company/acme",
        "recommended": True,
    },
    {
        "name": "Jane Doe",
        "photo_url": "",
        "linkedin_url": "https://linkedin.com/in/janedoe2",
        "title": "Product Manager",
        "headline": "PM at Other Co",
        "company_name": "Other Co",
        "seniority": "manager",
        "city": "New York",
        "state": "NY",
        "country": "US",
        "company_industry": "Finance",
        "company_size": 200,
        "company_domain": "otherco.com",
        "company_linkedin": "",
    },
]


class TestPendingReview:
    def test_no_pending_profiles(self):
        response = client.get("/profiles/pending-review")
        assert response.status_code == 200
        assert response.json() == []

    def test_pending_review_returned(self):
        _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        response = client.get("/profiles/pending-review")
        assert response.status_code == 200
        data = response.json()
        assert len(data) >= 1
        assert data[0]["linkedin_status"] == "pending_review"

    def test_no_match_included_in_pending(self):
        _create_profile(
            name="Bob Smith",
            email="bob@test.com",
            linkedin_status="no_match",
        )
        response = client.get("/profiles/pending-review")
        assert response.status_code == 200
        data = response.json()
        names = [p["name"] for p in data]
        assert "Bob Smith" in names

    def test_confirmed_not_in_pending(self):
        _create_profile(
            name="Alice Confirmed",
            email="alice@done.com",
            linkedin_status="confirmed",
            linkedin_url="https://linkedin.com/in/alice",
        )
        response = client.get("/profiles/pending-review")
        assert response.status_code == 200
        data = response.json()
        names = [p["name"] for p in data]
        assert "Alice Confirmed" not in names


class TestConfirmLinkedIn:
    def test_confirm_valid_candidate(self):
        pid = _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        response = client.post(
            f"/profiles/{pid}/confirm-linkedin",
            json={"candidate_index": 0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linkedin_status"] == "confirmed"

        # Verify profile was updated
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, pid)
        profile = json.loads(entity.domains)
        assert profile["linkedin_status"] == "confirmed"
        assert profile["linkedin_url"] == "https://linkedin.com/in/janedoe"
        assert profile["photo_url"] == "https://example.com/photo1.jpg"
        assert profile["title"] == "VP Engineering"
        assert profile["linkedin_candidates"] == []
        session.close()

    def test_confirm_candidate_without_photo(self):
        """Choosing a candidate with no photo should still confirm."""
        pid = _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        response = client.post(
            f"/profiles/{pid}/confirm-linkedin",
            json={"candidate_index": 1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linkedin_status"] == "confirmed"

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, pid)
        profile = json.loads(entity.domains)
        assert profile["linkedin_url"] == "https://linkedin.com/in/janedoe2"
        assert profile["title"] == "Product Manager"
        session.close()

    def test_reject_all_candidates(self):
        pid = _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        response = client.post(
            f"/profiles/{pid}/confirm-linkedin",
            json={"candidate_index": -1},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linkedin_status"] == "no_match"

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, pid)
        profile = json.loads(entity.domains)
        assert profile["linkedin_status"] == "no_match"
        assert profile["linkedin_candidates"] == []
        session.close()

    def test_invalid_candidate_index(self):
        pid = _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        response = client.post(
            f"/profiles/{pid}/confirm-linkedin",
            json={"candidate_index": 99},
        )
        assert response.status_code == 422

    def test_confirm_nonexistent_profile(self):
        response = client.post(
            "/profiles/99999/confirm-linkedin",
            json={"candidate_index": 0},
        )
        assert response.status_code == 404

    def test_confirm_copies_location(self):
        pid = _create_profile(
            linkedin_status="pending_review",
            linkedin_candidates=SAMPLE_CANDIDATES,
        )
        client.post(
            f"/profiles/{pid}/confirm-linkedin",
            json={"candidate_index": 0},
        )
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, pid)
        profile = json.loads(entity.domains)
        assert "San Francisco" in profile.get("location", "")
        session.close()


class TestSetLinkedIn:
    def test_set_manual_url(self):
        pid = _create_profile(linkedin_status="no_match")
        response = client.post(
            f"/profiles/{pid}/set-linkedin",
            json={"linkedin_url": "https://linkedin.com/in/janedoe-manual"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["linkedin_status"] == "confirmed"

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, pid)
        profile = json.loads(entity.domains)
        assert profile["linkedin_url"] == "https://linkedin.com/in/janedoe-manual"
        assert profile["linkedin_status"] == "confirmed"
        assert profile["linkedin_candidates"] == []
        session.close()

    def test_set_linkedin_nonexistent_profile(self):
        response = client.post(
            "/profiles/99999/set-linkedin",
            json={"linkedin_url": "https://linkedin.com/in/nobody"},
        )
        assert response.status_code == 404

    def test_set_linkedin_empty_url_rejected(self):
        pid = _create_profile(linkedin_status="no_match")
        response = client.post(
            f"/profiles/{pid}/set-linkedin",
            json={"linkedin_url": ""},
        )
        assert response.status_code == 422


class TestSearchLinkedIn:
    def test_search_requires_apollo_key(self):
        """Without Apollo API key, search should fail gracefully."""
        pid = _create_profile(linkedin_status="no_match")
        response = client.post(
            f"/profiles/{pid}/search-linkedin",
            json={"query": "Jane Doe @ Acme"},
        )
        assert response.status_code == 400
        assert "Apollo" in response.json()["detail"]

    def test_search_nonexistent_profile(self):
        response = client.post(
            "/profiles/99999/search-linkedin",
            json={"query": "Nobody"},
        )
        assert response.status_code == 404

    def test_search_empty_query_rejected(self):
        pid = _create_profile(linkedin_status="no_match")
        response = client.post(
            f"/profiles/{pid}/search-linkedin",
            json={"query": ""},
        )
        assert response.status_code == 422


class TestRepairLinkedInStatus:
    def test_repair_confirmed_with_full_enrichment(self):
        """Profile with linkedin_url + photo + title → confirmed."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="Full Enriched",
            email="full@test.com",
            linkedin_url="https://linkedin.com/in/full",
            photo_url="https://example.com/photo.jpg",
            title="CTO",
            linkedin_status="",  # wiped by bug
        )
        repaired = repair_linkedin_status()
        assert repaired >= 1

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.query(EntityRecord).filter(
            EntityRecord.name == "Full Enriched"
        ).first()
        profile = json.loads(entity.domains)
        assert profile["linkedin_status"] == "confirmed"
        session.close()

    def test_repair_pending_with_partial_enrichment(self):
        """Profile with linkedin_url + title only → pending_review (not auto-confirmed)."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="Partial Enriched",
            email="partial@test.com",
            linkedin_url="https://linkedin.com/in/partial",
            title="Manager",
            linkedin_status="",  # wiped
        )
        repaired = repair_linkedin_status()
        assert repaired >= 1

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.query(EntityRecord).filter(
            EntityRecord.name == "Partial Enriched"
        ).first()
        profile = json.loads(entity.domains)
        assert profile["linkedin_status"] == "pending_review"
        # Should create a candidate stub for the review UI
        assert len(profile.get("linkedin_candidates", [])) >= 1
        stub = profile["linkedin_candidates"][0]
        assert stub["linkedin_url"] == "https://linkedin.com/in/partial"
        assert stub["name"] == "Partial Enriched"
        assert stub["title"] == "Manager"
        session.close()

    def test_repair_pending_with_linkedin_only(self):
        """Profile with just linkedin_url (no title/photo) → pending_review."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="Bare LinkedIn",
            email="bare@test.com",
            linkedin_url="https://linkedin.com/in/bare",
            linkedin_status="",
        )
        repaired = repair_linkedin_status()
        assert repaired >= 1

        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.query(EntityRecord).filter(
            EntityRecord.name == "Bare LinkedIn"
        ).first()
        profile = json.loads(entity.domains)
        assert profile["linkedin_status"] == "pending_review"
        session.close()

    def test_repair_skips_already_confirmed(self):
        """Profiles with existing linkedin_status are left alone."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="Already Done",
            email="done@test.com",
            linkedin_url="https://linkedin.com/in/done",
            photo_url="https://example.com/p.jpg",
            title="CEO",
            linkedin_status="confirmed",
        )
        repaired = repair_linkedin_status()
        assert repaired == 0

    def test_repair_skips_no_linkedin(self):
        """Profiles without linkedin_url are not touched."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="No LinkedIn",
            email="noli@test.com",
            linkedin_status="",
        )
        repaired = repair_linkedin_status()
        assert repaired == 0

    def test_repair_idempotent(self):
        """Running repair twice doesn't double-repair."""
        from app.sync.auto_sync import repair_linkedin_status

        _create_profile(
            name="Idem Profile",
            email="idem@test.com",
            linkedin_url="https://linkedin.com/in/idem",
            photo_url="https://example.com/idem.jpg",
            title="Director",
            linkedin_status="",
        )
        first = repair_linkedin_status()
        assert first >= 1
        second = repair_linkedin_status()
        assert second == 0


class TestNormalizeCandidateStub:
    def test_stub_has_all_fields(self):
        from app.sync.auto_sync import normalize_candidate_stub

        stub = normalize_candidate_stub(
            linkedin_url="https://linkedin.com/in/test",
            name="Test Person",
            title="Engineer",
            photo_url="https://example.com/photo.jpg",
        )
        assert stub["linkedin_url"] == "https://linkedin.com/in/test"
        assert stub["name"] == "Test Person"
        assert stub["title"] == "Engineer"
        assert stub["photo_url"] == "https://example.com/photo.jpg"
        # All normalize_candidate fields should be present
        for key in ("headline", "company_name", "seniority", "city", "state",
                     "country", "company_industry", "company_domain", "company_linkedin"):
            assert key in stub

    def test_stub_defaults_empty(self):
        from app.sync.auto_sync import normalize_candidate_stub

        stub = normalize_candidate_stub(
            linkedin_url="https://linkedin.com/in/min",
            name="Min Person",
        )
        assert stub["title"] == ""
        assert stub["photo_url"] == ""
        assert stub["company_size"] is None
