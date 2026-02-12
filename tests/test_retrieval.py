"""Tests for the retrieval module."""

from __future__ import annotations

import json

from app.retrieve.retriever import RetrievedEvidence, retrieve_for_entity
from app.store.database import EntityRecord, SourceRecord, get_session


class TestRetrieval:
    """Test evidence retrieval for brief generation."""

    def test_retrieve_by_entity_id(self, populated_db):
        evidence = retrieve_for_entity(entity_id=populated_db.id)

        assert evidence.has_data
        assert evidence.source_count >= 2  # transcript + email

    def test_retrieve_by_person_name(self, populated_db):
        evidence = retrieve_for_entity(person_name="Jane Doe")

        assert evidence.has_data

    def test_retrieve_by_email(self, populated_db):
        evidence = retrieve_for_entity(emails=["jane.doe@acmecorp.com"])

        assert evidence.has_data

    def test_retrieve_last_interaction(self, populated_db):
        evidence = retrieve_for_entity(entity_id=populated_db.id)

        assert evidence.last_interaction is not None
        assert evidence.last_interaction["source_id"] in (
            "ff-transcript-001",
            "gmail-msg-001",
        )

    def test_retrieve_action_items(self, populated_db):
        evidence = retrieve_for_entity(entity_id=populated_db.id)

        # The fireflies transcript has action items
        assert len(evidence.action_items) > 0

    def test_retrieve_concern_snippets(self, populated_db):
        evidence = retrieve_for_entity(entity_id=populated_db.id)

        # The transcript body mentions "concern"
        assert len(evidence.concern_snippets) > 0

    def test_retrieve_empty_for_unknown_entity(self):
        evidence = retrieve_for_entity(entity_id=99999)

        assert not evidence.has_data
        assert evidence.source_count == 0

    def test_retrieve_interactions_sorted_by_date(self, populated_db):
        evidence = retrieve_for_entity(entity_id=populated_db.id)

        if len(evidence.interactions) >= 2:
            dates = [ix.get("date") for ix in evidence.interactions if ix.get("date")]
            # Most recent first
            for i in range(len(dates) - 1):
                assert dates[i] >= dates[i + 1]

    def test_retrieved_evidence_has_data_property(self):
        empty = RetrievedEvidence()
        assert not empty.has_data
        assert empty.source_count == 0

        empty.interactions = [{"test": True}]
        assert empty.has_data
