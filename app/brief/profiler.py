"""Decision-grade intelligence dossier generation.

Replaces generic profile summaries with a Strategic Operating Model.
Every claim is evidence-tagged. Every section is structured for
executive decision-making: negotiation, pressure-testing, and
incentive alignment.

Three inputs are integrated:
1. Meeting notes / transcripts (internal)
2. Public statements & positions (web search)
3. System inferences (LLM analytical layer)
"""

from __future__ import annotations

import logging

from app.clients.openai_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a Strategic Intelligence Analyst. You produce decision-grade \
executive dossiers, not summaries.

## ABSOLUTE RULES

1. **Evidence tagging** — every non-trivial claim must carry ONE tag:
   - [VERIFIED-MEETING] — explicitly stated in meeting transcript/email
   - [VERIFIED-PUBLIC] — explicitly documented in a cited public source (URL required)
   - [INFERRED-HIGH] — high-confidence inference from multiple converging signals
   - [INFERRED-MEDIUM] — medium-confidence inference from limited signals
   - [INFERRED-LOW] — low-confidence inference from weak or single signal
   - [UNKNOWN] — no supporting evidence. State it. Do not guess.

2. **Citation format** — every [VERIFIED-PUBLIC] claim must include:
   (Source: [publisher/platform], URL: [url], Date: [date if known])
   Every [VERIFIED-MEETING] claim must reference the meeting context.

3. **No hallucination** — if you have no evidence for something, write \
"**No evidence available.**" An explicit gap is worth more than plausible fiction.

4. **No generic filler** — before writing any sentence, apply this test: \
"Could this sentence describe 50% of executives in this industry?" \
If yes, DELETE IT. Specific or nothing.

   BANNED phrases (unless directly quoting the subject with citation):
   - "strategic leader", "visionary", "thought leader"
   - "data-driven", "results-driven", "outcome-driven"
   - "passionate about", "deeply committed to"
   - "transformative", "game-changing", "cutting-edge"
   - "proven track record", "extensive experience"
   - "empowers teams", "bridges the gap"
   - "at the intersection of"
   - "holistic approach", "synergies"

5. **Use the subject's own words** — when you have quotes, use them. \
Verbatim quotes in > blockquotes with source attribution. \
Language patterns are evidence. Generic descriptions are not.

6. **Distinguish source tiers** — primary sources (company site, \
registry, direct talks) outweigh secondary sources (news articles, \
conference pages). Low-quality sources (forums, SEO content) should \
be flagged and never used for key claims.

7. **Recency matters** — prefer evidence from the last 24 months for \
career/bio claims. Older evidence should be labeled with its date.

8. **Disambiguation** — if the name is common, note what identifiers \
lock the identity (company, LinkedIn URL, location, photo) and flag \
any ambiguity risk."""

USER_PROMPT_TEMPLATE = """\
## SUBJECT IDENTIFIERS
- **Name**: {name}
- **Current Title**: {title}
- **Company**: {company}
- **LinkedIn**: {linkedin_url}
- **Location**: {location}
- **Industry**: {industry}
- **Company Size**: {company_size}

## INTERNAL CONTEXT (from our meetings and emails)
{internal_context}

## WEB RESEARCH (real-time search results with source tiers)
{web_research}

---

## REQUIRED OUTPUT: DECISION-GRADE INTELLIGENCE DOSSIER

Produce ALL sections below. Follow evidence rules strictly. \
If a section has no evidence, include the header with an explicit \
gap statement. Do NOT pad with generic language.

---

### 1. Strategic Snapshot (5-7 bullets max)

Answer ONLY these questions with evidence-backed statements:
- What game is this person playing? (career trajectory, company stage)
- What incentives are visible? (revenue targets, growth mandates, board pressure)
- What constraints are visible? (budget, headcount, organizational politics)
- What outcomes are they measured on? (KPIs, deliverables, quotas)
- What risks are they managing? (delivery, reputation, competition)

No adjectives. Only structural observations with evidence tags.

---

### 2. Verified Facts Table

| # | Fact | Tag | Source | URL / Reference | Confidence |
|---|------|-----|--------|-----------------|------------|

Include: name, title, company, location, education, notable affiliations, \
quantified outcomes, projects, clients (if stated), timeline events. \
Every row must have a source. If a fact is self-reported and unverified, note that.

---

### 3. Power & Influence Map

For each dimension, state what is known and tag it:
- **Formal authority**: Title, reporting line, org chart position
- **Informal influence**: Network, reputation, expertise leverage
- **Revenue control**: P&L ownership, budget authority, sales targets
- **Decision gate ownership**: What they can approve/veto
- **Who they need to impress**: Board, CEO, investors, customers
- **Who can veto them**: Stakeholders with override power

If unknown, write [UNKNOWN] explicitly. Do not infer org charts you haven't seen.

---

### 4. Incentive & Scorecard Hypothesis

Based on meeting + public positioning:
- **Short-term incentives** (0-3 months): What they need to deliver NOW
- **Medium-term incentives** (3-12 months): What success looks like this year
- **Career incentives**: Where they want to be in 2-3 years
- **Risk exposure**: What could go wrong for them personally
- **Where they personally win**: Outcomes that advance their career
- **Where they personally lose**: Outcomes that damage their position

Each must be tagged with evidence and confidence level. \
If you're inferring, state the upstream signals explicitly.

---

### 5. Strategic Tensions & Fault Lines

Identify live tensions from the evidence. Examples:
- Consulting revenue vs product margin
- Growth targets vs delivery capacity
- Regional expansion vs core market focus
- AI hype in positioning vs implementation realism
- Public optimism vs private caution

Each tension MUST cite specific evidence that reveals it. \
If you infer a tension, state what signals led you there.

---

### 6. Cognitive & Rhetorical Patterns (Evidence-Based Only)

Extract from meeting transcripts AND public content:
- **Repeated language**: Phrases they use multiple times (quote them)
- **Framing devices**: How they structure arguments
- **Growth vs control bias**: Do they lean toward expansion or risk management?
- **Product vs services bias**: Where do they see value creation?
- **Optimism vs realism**: How do they handle uncertainty?
- **Abstraction level**: Do they operate at strategy, tactics, or execution?
- **Comfort with numbers**: Do they cite metrics or stay qualitative?
- **Comfort with ambiguity**: Do they need certainty or embrace uncertainty?

Use direct quotes when available. \
No generic personality typing. No MBTI. No "likely analytical."

---

### 7. Behavioral Forecast (Scenario-Based)

Generate specific predictions with reasoning:

**If [specific scenario] → Likely reaction:**
- Prediction
- Reasoning (cite evidence)

Cover at minimum:
- If challenged on a key claim → How they defend
- If their revenue/delivery target slips → What they likely do
- If asked for a commitment → How they respond
- If presented with competing priorities → How they choose

Each forecast must cite the evidence behind the prediction.

---

### 8. Conversation & Negotiation Playbook

Produce:
- **3 leverage angles** — mapped to their specific incentive structure
- **2 stress tests** — pressure points that reveal real position
- **2 credibility builders** — what earns trust with THIS person specifically
- **1 contrarian wedge** — intelligent challenge that earns respect
- **1 high-upside collaboration vector** — best partnership angle

Each must reference the incentive or pattern that makes it effective.

---

### 9. Delta: Public Persona vs Meeting Persona

Compare what their public positioning says vs what meeting signals reveal:
- **Alignments**: Where public and private messages match
- **Divergences**: Where they differ, and what the gap implies
- **Implications**: What the delta tells us about their real priorities

If only one source is available, state that and note what the other \
source would add.

---

### 10. Unknowns That Matter

ONLY include gaps that materially change strategy. Examples:
- P&L ownership level → affects budget authority assumptions
- Compensation structure → affects motivation analysis
- Board reporting line → affects decision-making speed
- Equity exposure → affects risk tolerance
- Internal political dynamics → affects what they can commit to

Do NOT include generic gaps like "education history unknown" unless \
it specifically affects the meeting strategy.

---

### 11. Engine Improvement Recommendations

After completing the dossier, specify:
- **Missing signals**: What data was absent that would improve accuracy
- **Recommended data sources**: Specific sources to fetch next time
- **Capture fields**: Structured fields the meeting tool should record \
in future calls (e.g., risk appetite signals, growth pressure markers, \
incentive cues, timeline commitments, tone markers, interruptions, \
deflection patterns)

---

## OUTPUT CONSTRAINTS
- Use markdown with clear ## section headers
- Use tables where specified
- Use > blockquotes for direct quotes (with source)
- Bold key terms and names
- Tag EVERY non-trivial claim
- Flag EVERY gap explicitly
- NO padding. Short and honest beats long and fabricated.
- If the output could describe 50% of enterprise AI leaders, rewrite it.

## QUALITY SELF-CHECK (run before finalizing)
- [ ] Every factual claim has a source citation
- [ ] Every inference states confidence + upstream signals
- [ ] Every section with insufficient evidence says so explicitly
- [ ] Behavioral forecasts cite specific evidence
- [ ] Conversation strategy maps to specific incentives
- [ ] No generic executive cliches survived
- [ ] The dossier distinguishes VERIFIED vs INFERRED vs UNKNOWN
- [ ] Unknowns are strategically material, not generic
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
    web_research: str = "",
) -> str:
    """Generate a decision-grade intelligence dossier for a contact.

    Returns the dossier as a markdown string.
    Raises RuntimeError if the LLM client is not available.
    """
    if interactions_summary:
        internal_context = interactions_summary
    else:
        internal_context = "No internal meeting or email history available."

    if not web_research:
        web_research = (
            "No web search results available. Rely on your training data and "
            "flag all claims with appropriate confidence levels. "
            "Tag every claim sourced from training data as [INFERRED-LOW] "
            "since it cannot be verified against current public sources."
        )

    user_prompt = USER_PROMPT_TEMPLATE.format(
        name=name,
        title=title or "Unknown",
        company=company or "Unknown",
        linkedin_url=linkedin_url or "Not available",
        location=location or "Unknown",
        industry=industry or "Unknown",
        company_size=f"{company_size:,} employees" if company_size else "Unknown",
        internal_context=internal_context,
        web_research=web_research,
    )

    llm = LLMClient()
    return llm.chat(SYSTEM_PROMPT, user_prompt, temperature=0.3)


def build_interactions_summary(profile_data: dict) -> str:
    """Build a text summary of internal interactions for the LLM prompt.

    Includes meeting signals, behavioral observations, and action items.
    """
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
