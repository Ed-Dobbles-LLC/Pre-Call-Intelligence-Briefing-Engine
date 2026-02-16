"""Tests for the decision-grade intelligence dossier module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

os.environ["DATABASE_URL"] = "sqlite:///./test_briefing_engine.db"
os.environ["OPENAI_API_KEY"] = ""
os.environ["FIREFLIES_API_KEY"] = ""
os.environ["BRIEFING_API_KEY"] = ""

import pytest

from app.brief.profiler import (
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    build_interactions_summary,
    generate_deep_profile,
)


class TestBuildInteractionsSummary:
    def test_empty_profile(self):
        result = build_interactions_summary({})
        assert "No internal" in result

    def test_with_interactions(self):
        profile = {
            "interactions": [
                {
                    "type": "meeting",
                    "title": "Q1 Review",
                    "date": "2026-01-15",
                    "summary": "Discussed pipeline.",
                },
                {
                    "type": "email",
                    "title": "Follow up",
                    "date": "2026-01-20",
                },
            ],
        }
        result = build_interactions_summary(profile)
        assert "2 recorded interactions" in result
        assert "Q1 Review" in result
        assert "Follow up" in result
        assert "Discussed pipeline" in result

    def test_with_action_items(self):
        profile = {
            "action_items": [
                "Send proposal by Friday",
                "Schedule follow-up",
            ],
        }
        result = build_interactions_summary(profile)
        assert "Open action items (2)" in result
        assert "Send proposal" in result

    def test_with_both(self):
        profile = {
            "interactions": [
                {"type": "meeting", "title": "Kickoff", "date": "2026-02-01"},
            ],
            "action_items": ["Review contract"],
        }
        result = build_interactions_summary(profile)
        assert "1 recorded interactions" in result
        assert "Open action items" in result

    def test_limits_interactions_to_15(self):
        interactions = [
            {"type": "meeting", "title": f"Meeting {i}", "date": f"2026-01-{i:02d}"}
            for i in range(1, 25)
        ]
        profile = {"interactions": interactions}
        result = build_interactions_summary(profile)
        # Should only include first 15
        assert "Meeting 15" in result
        assert "Meeting 16" not in result

    def test_limits_action_items_to_10(self):
        items = [f"Action item {i}" for i in range(15)]
        profile = {"action_items": items}
        result = build_interactions_summary(profile)
        assert "Action item 9" in result
        assert "Action item 10" not in result

    def test_interaction_without_summary(self):
        profile = {
            "interactions": [
                {"type": "meeting", "title": "Quick Chat", "date": "2026-02-10"},
            ],
        }
        result = build_interactions_summary(profile)
        assert "Quick Chat" in result
        assert "Summary:" not in result

    def test_interaction_type_uppercased(self):
        profile = {
            "interactions": [
                {"type": "email", "title": "Re: Proposal", "date": "2026-02-10"},
            ],
        }
        result = build_interactions_summary(profile)
        assert "[EMAIL]" in result

    def test_includes_participants(self):
        profile = {
            "interactions": [
                {
                    "type": "meeting",
                    "title": "Team Call",
                    "date": "2026-02-10",
                    "participants": ["alice@test.com", "bob@test.com"],
                },
            ],
        }
        result = build_interactions_summary(profile)
        assert "alice@test.com" in result
        assert "bob@test.com" in result

    def test_includes_key_points(self):
        profile = {
            "interactions": [
                {
                    "type": "meeting",
                    "title": "Strategy Session",
                    "date": "2026-02-10",
                    "key_points": "Discussed Q1 targets and hiring plan",
                },
            ],
        }
        result = build_interactions_summary(profile)
        assert "Q1 targets" in result


class TestGenerateDeepProfile:
    def test_raises_without_openai(self):
        """Should raise RuntimeError when OpenAI client is not available."""
        with pytest.raises(RuntimeError, match="OpenAI client"):
            generate_deep_profile(name="Test Person")

    @patch("app.brief.profiler.LLMClient")
    def test_calls_llm_with_correct_prompts(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Deep Profile for Test Person"
        MockLLM.return_value = mock_instance

        result = generate_deep_profile(
            name="Test Person",
            title="CTO",
            company="TestCo",
            linkedin_url="https://linkedin.com/in/testperson",
            location="New York, NY",
            industry="Technology",
            company_size=100,
            interactions_summary="We had 3 meetings.",
        )

        assert result == "# Deep Profile for Test Person"
        mock_instance.chat.assert_called_once()
        call_args = mock_instance.chat.call_args

        # Verify system prompt
        assert call_args[0][0] == SYSTEM_PROMPT

        # Verify user prompt contains all provided data
        user_prompt = call_args[0][1]
        assert "Test Person" in user_prompt
        assert "CTO" in user_prompt
        assert "TestCo" in user_prompt
        assert "linkedin.com/in/testperson" in user_prompt
        assert "New York, NY" in user_prompt
        assert "Technology" in user_prompt
        assert "100 employees" in user_prompt
        assert "We had 3 meetings" in user_prompt

        # Verify temperature
        assert call_args[1]["temperature"] == 0.3

    @patch("app.brief.profiler.LLMClient")
    def test_defaults_for_missing_fields(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Profile"
        MockLLM.return_value = mock_instance

        generate_deep_profile(name="Minimal Person")

        user_prompt = mock_instance.chat.call_args[0][1]
        assert "Minimal Person" in user_prompt
        assert "Unknown" in user_prompt  # defaults for missing fields

    @patch("app.brief.profiler.LLMClient")
    def test_no_interactions_uses_default_message(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Profile"
        MockLLM.return_value = mock_instance

        generate_deep_profile(name="No History Person")

        user_prompt = mock_instance.chat.call_args[0][1]
        assert "No internal meeting or email history" in user_prompt

    @patch("app.brief.profiler.LLMClient")
    def test_returns_string(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "## Executive Summary\n- Career arc..."
        MockLLM.return_value = mock_instance

        result = generate_deep_profile(name="Return Test")
        assert isinstance(result, str)
        assert "Executive Summary" in result

    @patch("app.brief.profiler.LLMClient")
    def test_evidence_threshold_passed_to_prompt(self, MockLLM):
        mock_instance = MagicMock()
        mock_instance.chat.return_value = "# Profile"
        MockLLM.return_value = mock_instance

        generate_deep_profile(name="Threshold Test", evidence_threshold=60)

        user_prompt = mock_instance.chat.call_args[0][1]
        assert "60%" in user_prompt


class TestPromptTemplates:
    def test_system_prompt_has_rules(self):
        assert "ABSOLUTE RULES" in SYSTEM_PROMPT
        assert "Pre-Call Intelligence Analyst" in SYSTEM_PROMPT

    def test_system_prompt_requires_evidence_tagging(self):
        assert "VERIFIED" in SYSTEM_PROMPT
        assert "MEETING" in SYSTEM_PROMPT
        assert "PUBLIC" in SYSTEM_PROMPT
        assert "INFERRED" in SYSTEM_PROMPT
        assert "[UNKNOWN]" in SYSTEM_PROMPT

    def test_system_prompt_requires_gap_flagging(self):
        assert "No evidence available" in SYSTEM_PROMPT

    def test_system_prompt_bans_fluff(self):
        assert "BANNED phrases" in SYSTEM_PROMPT
        assert "strategic leader" in SYSTEM_PROMPT

    def test_system_prompt_requires_disambiguation(self):
        assert "Disambiguation" in SYSTEM_PROMPT

    def test_system_prompt_supports_web_citations(self):
        assert "URL" in SYSTEM_PROMPT

    def test_system_prompt_requires_recency(self):
        assert "24 months" in SYSTEM_PROMPT

    def test_user_prompt_template_has_all_sections(self):
        assert "SUBJECT IDENTIFIERS" in USER_PROMPT_TEMPLATE
        assert "INTERNAL CONTEXT" in USER_PROMPT_TEMPLATE
        assert "Executive Summary" in USER_PROMPT_TEMPLATE
        assert "Identity & Disambiguation" in USER_PROMPT_TEMPLATE
        assert "Career Timeline" in USER_PROMPT_TEMPLATE
        assert "Public Statements & Positions" in USER_PROMPT_TEMPLATE
        assert "Public Visibility" in USER_PROMPT_TEMPLATE
        assert "Quantified Claims Inventory" in USER_PROMPT_TEMPLATE
        assert "Rhetorical & Decision Patterns" in USER_PROMPT_TEMPLATE
        assert "Structural Pressure Model" in USER_PROMPT_TEMPLATE
        assert "Interview Strategy Recommendations" in USER_PROMPT_TEMPLATE
        assert "Primary Source Index" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_career_timeline(self):
        assert "Chronological list of roles" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_interview_strategy(self):
        assert "What to lead with" in USER_PROMPT_TEMPLATE
        assert "Landmines" in USER_PROMPT_TEMPLATE
        assert "Questions that will earn respect" in USER_PROMPT_TEMPLATE

    def test_user_prompt_template_format_fields(self):
        """Ensure all format placeholders can be filled."""
        result = USER_PROMPT_TEMPLATE.format(
            name="Test",
            title="Test",
            company="Test",
            linkedin_url="Test",
            location="Test",
            industry="Test",
            company_size="Test",
            internal_context="Test",
            web_research="Test",
            visibility_research="Test",
            evidence_threshold=85,
        )
        assert "{" not in result  # no unfilled placeholders

    def test_user_prompt_has_self_check(self):
        assert "SELF-CHECK" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_web_research_section(self):
        assert "WEB RESEARCH" in USER_PROMPT_TEMPLATE
        assert "{web_research}" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_pressure_model(self):
        """Pressure model maps mandate and pressures to decision drivers."""
        assert "Current mandate" in USER_PROMPT_TEMPLATE
        assert "Key pressures" in USER_PROMPT_TEMPLATE

    def test_user_prompt_has_quantified_claims(self):
        """Quantified claims section inventories specific numbers."""
        assert "Quantified Claims" in USER_PROMPT_TEMPLATE
        assert "personally owned" in USER_PROMPT_TEMPLATE
