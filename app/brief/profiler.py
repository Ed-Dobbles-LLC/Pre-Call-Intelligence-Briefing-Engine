"""Deep profile generation: executive intelligence dossier for verified contacts.

Generates a structured intelligence profile using LLM analysis of public
information and internal evidence (meetings, emails). Output is a markdown
report covering career patterns, strategic thinking, conversation strategy,
and risk signals.
"""

from __future__ import annotations

import logging

from app.clients.openai_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior executive intelligence analyst. Your job is to produce
a deep, structured profile of a business contact to help the user prepare
for a high-stakes meeting or call.

RULES:
1. Use only publicly available information from your training data.
2. Separate facts from interpretation. Label inferences clearly.
3. Cite source types when possible (LinkedIn post, podcast, press release, etc.).
4. Prioritize depth over breadth. Avoid generic executive clichés.
5. If public data is limited, identify information gaps clearly and infer
   cautiously using career patterns. Flag inference confidence level.
6. Be specific and actionable. This profile will be used to prepare for
   a real conversation."""

USER_PROMPT_TEMPLATE = """\
## PROFILE REQUEST
- **Name**: {name}
- **Current Title**: {title}
- **Company**: {company}
- **LinkedIn**: {linkedin_url}
- **Location**: {location}
- **Industry**: {industry}
- **Company Size**: {company_size}

## INTERNAL CONTEXT
{internal_context}

## RESEARCH SCOPE
Use publicly available information: LinkedIn posts/articles/comments, company \
blogs, podcasts, interviews, webinars, conference talks, press releases, \
Medium/Substack/guest articles, public filings, panel/keynote summaries, \
and news coverage.

## REQUIRED DELIVERABLES

### 1. Executive Snapshot (1 page max)
- Career arc narrative
- Core identity (operator? strategist? technologist? evangelist?)
- Repeated career themes
- Pattern of value creation
- Risk appetite profile

### 2. Intellectual Profile
Based on public writing and speaking:
- How do they define value in their domain?
- Do they lean technical, commercial, or organizational?
- How do they talk about ROI and outcomes?
- Enterprise vs startup mindset indicators
- Signals of long-term thinking vs short-term execution bias
- Language analysis (buzzword-heavy vs pragmatic)
- Innovation posture (incremental vs transformational)
- Decision-making style signals
Include direct quotes when possible.

### 3. Strategic Pattern Analysis
Identify:
- Recurring strategic moves
- Organizational scaling philosophy
- Build vs buy tendencies
- Talent philosophy
- Consulting vs product orientation balance
- Platform vs services bias

### 4. Meeting Behavior Forecast
Predict:
- What they likely screen for in conversations
- Red flags they may react strongly to
- Signals that resonate with them
- Language patterns to mirror
- Topics likely to energize them
- Topics likely to disengage them
Provide supporting evidence from public content.

### 5. Power Map Context
Analyze:
- Their role in the company's growth strategy
- Likely influence over decisions
- Whether they are positioned as operator, visionary, or integration leader

### 6. Strategic Conversation Playbook
Provide:
- 5 high-leverage questions to ask them
- 3 positioning angles that align with their thinking
- 3 positioning mistakes to avoid
- 2 contrarian but intelligent discussion hooks

### 7. Risk & Opportunity Signals
Identify:
- Potential blind spots
- Strategic tensions in their background
- Signals of future direction

## CONSTRAINTS
- No speculation without labeling it clearly as inference.
- Separate facts from interpretation.
- Cite source type (LinkedIn post, podcast, press release, etc.).
- Prioritize depth over breadth.
- Avoid generic executive fluff.

## OUTPUT FORMAT
Use markdown with clear section headers (## 1. Executive Snapshot, etc.).
Use bullet points, bold for key terms, and > blockquotes for direct quotes.

## QUALITY BAR
The profile is successful if:
- It surfaces non-obvious patterns
- It identifies consistent cognitive frameworks
- It distinguishes signal from résumé summary
- It provides actionable conversation strategy
- It avoids generic executive clichés

Before finalizing, self-check:
- Are claims supported?
- Are interpretations separated from facts?
- Are meeting insights actionable?
- Is there pattern recognition, not just biography?
"""


def generate_deep_profile(
    name: str,
    title: str = "",
    company: str = "",
    linkedin_url: str = "",
    location: str = "",
    industry: str = "",
    company_size: int | None = None,
    interactions_summary: str = "",
) -> str:
    """Generate a deep intelligence profile for a verified contact.

    Returns the profile as a markdown string.
    Raises RuntimeError if the LLM client is not available.
    """
    # Build internal context from our data
    if interactions_summary:
        internal_context = interactions_summary
    else:
        internal_context = "No internal meeting or email history available."

    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=name,
        title=title or "Unknown",
        company=company or "Unknown",
        linkedin_url=linkedin_url or "Not available",
        location=location or "Unknown",
        industry=industry or "Unknown",
        company_size=f"{company_size:,} employees" if company_size else "Unknown",
        internal_context=internal_context,
    )

    llm = LLMClient()
    return llm.chat(SYSTEM_PROMPT, user_prompt, temperature=0.3)


def build_interactions_summary(profile_data: dict) -> str:
    """Build a text summary of internal interactions for the LLM prompt."""
    parts = []

    interactions = profile_data.get("interactions", [])
    if interactions:
        parts.append(f"We have {len(interactions)} recorded interactions:")
        for ix in interactions[:15]:
            ix_type = ix.get("type", "meeting").upper()
            ix_title = ix.get("title", "Untitled")
            ix_date = ix.get("date", "Unknown date")
            ix_summary = ix.get("summary", "")
            parts.append(f"- [{ix_type}] {ix_date}: {ix_title}")
            if ix_summary:
                parts.append(f"  Summary: {ix_summary}")

    action_items = profile_data.get("action_items", [])
    if action_items:
        parts.append(f"\nOpen action items ({len(action_items)}):")
        for item in action_items[:10]:
            parts.append(f"- {item}")

    if not parts:
        return "No internal meeting or email history available."

    return "\n".join(parts)
