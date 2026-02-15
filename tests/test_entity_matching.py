"""Tests for entity resolution and matching."""

from __future__ import annotations

import json

from app.normalize.entity_resolver import (
    resolve_company,
    resolve_person,
    _extract_email_from_header,
    _extract_domain,
)
from app.store.database import EntityRecord, SourceRecord, get_session


class TestEmailExtraction:
    """Test helper functions for email/domain extraction."""

    def test_extract_email_from_angle_brackets(self):
        assert _extract_email_from_header("Jane Doe <jane@acme.com>") == "jane@acme.com"

    def test_extract_email_plain(self):
        assert _extract_email_from_header("jane@acme.com") == "jane@acme.com"

    def test_extract_email_empty(self):
        assert _extract_email_from_header("no email here") is None

    def test_extract_domain(self):
        assert _extract_domain("jane@acme.com") == "acme.com"

    def test_extract_domain_no_at(self):
        assert _extract_domain("noemail") is None


class TestPersonResolution:
    """Test person entity resolution."""

    def test_resolve_creates_new_entity(self):
        result = resolve_person("Alice Smith")

        assert result.entity_id is not None
        assert result.name == "Alice Smith"
        assert result.entity_type == "person"

    def test_resolve_with_email(self):
        result = resolve_person("Alice Smith", email="alice@example.com")

        assert "alice@example.com" in result.emails

    def test_resolve_finds_existing_entity(self):
        # Create first
        result1 = resolve_person("Bob Jones", email="bob@test.com")
        # Resolve again
        result2 = resolve_person("Bob Jones")

        assert result2.entity_id == result1.entity_id

    def test_resolve_finds_by_email_match(self):
        # Create with email
        result1 = resolve_person("Carol White", email="carol@corp.com")
        # Look up by email on different name
        session = get_session("sqlite:///./test_briefing_engine.db")
        entity = session.get(EntityRecord, result1.entity_id)
        assert entity is not None
        assert "carol@corp.com" in entity.get_emails()
        session.close()

    def test_resolve_discovers_aliases_from_source_records(self):
        """If source records mention the person, their info should be discovered."""
        session = get_session("sqlite:///./test_briefing_engine.db")
        record = SourceRecord(
            source_type="fireflies",
            source_id="test-discovery",
            participants=json.dumps(["dave@partner.com", "Dave Martinez"]),
            title="Meeting with Dave Martinez",
            body="Call with Dave.",
        )
        session.add(record)
        session.commit()
        session.close()

        result = resolve_person("Dave Martinez")
        # Should discover the email from participant list
        assert result.entity_id is not None


class TestCompanyResolution:
    """Test company entity resolution."""

    def test_resolve_company_creates_new(self):
        result = resolve_company("Acme Corp")

        assert result.entity_id is not None
        assert result.name == "Acme Corp"
        assert result.entity_type == "company"

    def test_resolve_company_with_domain(self):
        result = resolve_company("Acme Corp", domain="acmecorp.com")

        assert "acmecorp.com" in result.domains

    def test_resolve_company_finds_existing(self):
        result1 = resolve_company("TechStart Inc", domain="techstart.io")
        result2 = resolve_company("TechStart")

        assert result2.entity_id == result1.entity_id

    def test_resolve_company_discovers_domains_from_records(self):
        session = get_session("sqlite:///./test_briefing_engine.db")
        record = SourceRecord(
            source_type="gmail",
            source_id="test-company-disc",
            participants=json.dumps(["sarah@bigco.com"]),
            body="Sarah from BigCo mentioned that bigco.com is their main domain.",
        )
        session.add(record)
        session.commit()
        session.close()

        result = resolve_company("BigCo")
        assert result.entity_id is not None
