"""Tests for the FastAPI web API."""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""

from fastapi.testclient import TestClient

from app.api import app


client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

    def test_health_shows_config_status(self):
        response = client.get("/health")
        data = response.json()
        assert "fireflies_configured" in data
        assert "openai_configured" in data
        assert "database" in data


class TestBriefEndpoint:
    def test_brief_requires_person_or_company(self):
        response = client.post("/brief", json={})
        assert response.status_code == 422

    def test_brief_with_person_no_data(self):
        """With no stored data and no API keys, should return a zero-confidence brief."""
        response = client.post("/brief", json={
            "person": "Ghost Person",
            "skip_ingestion": True,
        })
        # Should succeed (graceful degradation) or fail if OpenAI not configured
        # With skip_ingestion and no data, pipeline produces a no-evidence brief
        assert response.status_code == 200
        data = response.json()
        assert data["confidence_score"] == 0.0
        assert "brief" in data
        assert "markdown" in data

    def test_brief_markdown_endpoint(self):
        response = client.post("/brief/markdown", json={
            "person": "Nobody",
            "skip_ingestion": True,
        })
        assert response.status_code == 200
        assert "Pre-Call Intelligence Brief" in response.text

    def test_brief_json_endpoint(self):
        response = client.post("/brief/json", json={
            "person": "Nobody",
            "skip_ingestion": True,
        })
        assert response.status_code == 200
        data = response.json()
        assert "header" in data
        assert "open_loops" in data
