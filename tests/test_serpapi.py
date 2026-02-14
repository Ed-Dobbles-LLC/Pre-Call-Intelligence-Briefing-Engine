"""Tests for the SerpAPI web search client."""

from __future__ import annotations

import os

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""
os.environ["SERPAPI_API_KEY"] = ""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.clients.serpapi import (
    SerpAPIClient,
    _normalize_result,
    format_web_results_for_prompt,
)


SAMPLE_ORGANIC_RESULTS = [
    {
        "position": 1,
        "title": "Jane Doe - VP Engineering at Acme Corp | LinkedIn",
        "link": "https://www.linkedin.com/in/janedoe",
        "snippet": "Jane Doe is VP Engineering at Acme Corp, leading a team of 50+...",
        "source": "LinkedIn",
        "date": "2026-01-15",
    },
    {
        "position": 2,
        "title": "Jane Doe on the Future of AI in Enterprise - TechCrunch",
        "link": "https://techcrunch.com/2025/11/jane-doe-ai-enterprise",
        "snippet": "In a keynote at CloudConf 2025, Jane Doe argued that...",
        "source": "TechCrunch",
    },
    {
        "position": 3,
        "title": "Acme Corp Raises $50M Series C",
        "link": "https://news.example.com/acme-series-c",
        "snippet": "Acme Corp announced a $50M round led by...",
        "source": "Business Wire",
        "date": "2025-09-01",
    },
]


class TestNormalizeResult:
    def test_extracts_all_fields(self):
        result = _normalize_result(SAMPLE_ORGANIC_RESULTS[0])
        assert result["title"] == "Jane Doe - VP Engineering at Acme Corp | LinkedIn"
        assert result["link"] == "https://www.linkedin.com/in/janedoe"
        assert "VP Engineering" in result["snippet"]
        assert result["source"] == "LinkedIn"
        assert result["date"] == "2026-01-15"

    def test_handles_missing_fields(self):
        result = _normalize_result({"title": "Test", "link": "https://example.com"})
        assert result["title"] == "Test"
        assert result["link"] == "https://example.com"
        assert result["snippet"] == ""
        assert result["source"] == ""
        assert result["date"] == ""

    def test_handles_empty_dict(self):
        result = _normalize_result({})
        assert result["title"] == ""
        assert result["link"] == ""


class TestFormatWebResults:
    def test_empty_results(self):
        result = format_web_results_for_prompt({
            "general": [], "linkedin": [], "news": [], "talks": []
        })
        assert result == ""

    def test_formats_general_results(self):
        results = {
            "general": [_normalize_result(r) for r in SAMPLE_ORGANIC_RESULTS[:2]],
            "linkedin": [],
            "news": [],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "General Search Results" in formatted
        assert "Jane Doe - VP Engineering" in formatted
        assert "TechCrunch" in formatted
        assert "URL:" in formatted

    def test_includes_urls(self):
        results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [],
            "news": [],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "https://www.linkedin.com/in/janedoe" in formatted

    def test_includes_snippets(self):
        results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [],
            "news": [],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "VP Engineering at Acme Corp" in formatted

    def test_multiple_categories(self):
        results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "news": [_normalize_result(SAMPLE_ORGANIC_RESULTS[1])],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "General Search Results" in formatted
        assert "LinkedIn Results" in formatted
        assert "News & Articles" in formatted
        assert "Conference Talks" not in formatted  # empty

    def test_date_shown_when_present(self):
        results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [],
            "news": [],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "2026-01-15" in formatted

    def test_instructions_included(self):
        results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [],
            "news": [],
            "talks": [],
        }
        formatted = format_web_results_for_prompt(results)
        assert "primary sources" in formatted


class TestSerpAPIClient:
    def test_no_api_key_returns_empty(self):
        client = SerpAPIClient(api_key="")
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.search("test query")
        )
        assert result == []

    def test_no_api_key_search_person_returns_empty(self):
        client = SerpAPIClient(api_key="")
        import asyncio
        result = asyncio.get_event_loop().run_until_complete(
            client.search_person(name="Test Person")
        )
        assert result == {"general": [], "linkedin": [], "news": [], "talks": []}

    @pytest.mark.asyncio
    async def test_search_calls_serpapi(self):
        """Verify search makes the right HTTP call."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "organic_results": SAMPLE_ORGANIC_RESULTS
        }

        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            client = SerpAPIClient(api_key="test-key-123")
            results = await client.search("Jane Doe Acme Corp")

            assert len(results) == 3
            mock_client.get.assert_called_once()
            call_kwargs = mock_client.get.call_args
            assert call_kwargs[1]["params"]["q"] == "Jane Doe Acme Corp"
            assert call_kwargs[1]["params"]["api_key"] == "test-key-123"

    @pytest.mark.asyncio
    async def test_search_handles_403(self):
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Forbidden"

        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            client = SerpAPIClient(api_key="bad-key")
            results = await client.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_429(self):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.text = "Rate limited"

        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            client = SerpAPIClient(api_key="test-key")
            results = await client.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_handles_exception(self):
        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.side_effect = Exception("Network error")
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            client = SerpAPIClient(api_key="test-key")
            results = await client.search("test")
            assert results == []

    @pytest.mark.asyncio
    async def test_search_person_runs_multiple_queries(self):
        """search_person should run 4 categorised searches."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"organic_results": SAMPLE_ORGANIC_RESULTS[:1]}

        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            with patch("app.clients.serpapi.asyncio.sleep", new_callable=AsyncMock):
                client = SerpAPIClient(api_key="test-key")
                results = await client.search_person(
                    name="Jane Doe", company="Acme Corp"
                )

            assert "general" in results
            assert "linkedin" in results
            assert "news" in results
            assert "talks" in results
            # 4 categories = 4 searches
            assert mock_client.get.call_count == 4

    @pytest.mark.asyncio
    async def test_search_person_query_includes_company(self):
        """When company is provided, searches should include it."""
        captured_queries = []

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"organic_results": []}

        async def capture_get(url, params=None):
            captured_queries.append(params["q"])
            return mock_response

        with patch("app.clients.serpapi.httpx.AsyncClient") as MockClient:
            mock_client = AsyncMock()
            mock_client.get = capture_get
            mock_client.__aenter__ = AsyncMock(return_value=mock_client)
            mock_client.__aexit__ = AsyncMock(return_value=False)
            MockClient.return_value = mock_client

            with patch("app.clients.serpapi.asyncio.sleep", new_callable=AsyncMock):
                client = SerpAPIClient(api_key="test-key")
                await client.search_person(name="Jane Doe", company="Acme Corp")

            # General query should have both name and company
            assert '"Jane Doe"' in captured_queries[0]
            assert '"Acme Corp"' in captured_queries[0]
            # LinkedIn query should be site-scoped
            assert "site:linkedin.com" in captured_queries[1]


class TestProfilerWebIntegration:
    """Test that the profiler correctly passes web research to the LLM."""

    @patch("app.brief.profiler.LLMClient")
    def test_web_research_included_in_prompt(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Deep Profile"
        MockLLM.return_value = mock_instance

        from app.brief.profiler import generate_deep_profile

        web_data = "**General Search Results:**\n1. Jane Doe at Acme Corp..."
        generate_deep_profile(
            name="Jane Doe",
            company="Acme Corp",
            web_research=web_data,
        )

        user_prompt = mock_instance.chat.call_args[0][1]
        assert "Jane Doe at Acme Corp" in user_prompt
        assert "WEB RESEARCH" in user_prompt

    @patch("app.brief.profiler.LLMClient")
    def test_no_web_research_uses_fallback(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Deep Profile"
        MockLLM.return_value = mock_instance

        from app.brief.profiler import generate_deep_profile

        generate_deep_profile(name="No Web Person")

        user_prompt = mock_instance.chat.call_args[0][1]
        assert "No web search results available" in user_prompt
        assert "training data" in user_prompt

    @patch("app.brief.profiler.LLMClient")
    def test_web_research_with_full_format(self, MockLLM):
        """End-to-end: format search results then pass to profiler."""
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Profile with Web Data"
        MockLLM.return_value = mock_instance

        from app.brief.profiler import generate_deep_profile

        search_results = {
            "general": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "linkedin": [_normalize_result(SAMPLE_ORGANIC_RESULTS[0])],
            "news": [_normalize_result(SAMPLE_ORGANIC_RESULTS[1])],
            "talks": [],
        }
        web_text = format_web_results_for_prompt(search_results)

        result = generate_deep_profile(
            name="Jane Doe",
            company="Acme Corp",
            web_research=web_text,
        )

        assert result == "# Profile with Web Data"
        user_prompt = mock_instance.chat.call_args[0][1]
        assert "linkedin.com/in/janedoe" in user_prompt
        assert "TechCrunch" in user_prompt
