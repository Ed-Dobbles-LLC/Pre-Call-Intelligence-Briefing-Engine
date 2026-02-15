"""Decision-grade contact intelligence dossier generation.

Produces leverage, not summaries. Every claim is evidence-tagged.
Every section is structured for executive decision-making:
pressure modeling, incentive alignment, behavioral forecasting,
and negotiation strategy.

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
You are a Strategic Intelligence Analyst producing decision-grade \
contact intelligence dossiers. Your output must create leverage, not summaries.

## ABSOLUTE RULES

1. **Evidence tagging** — every non-trivial claim must carry ONE tag:
   - [VERIFIED–MEETING] — explicitly stated in meeting transcript/email
   - [VERIFIED–PUBLIC] — explicitly documented in a cited public source (URL required)
   - [INFERRED–H] — high-confidence inference from multiple converging signals \
(MUST cite upstream signals in the same sentence)
   - [INFERRED–M] — medium-confidence inference from limited signals
   - [INFERRED–L] — low-confidence inference from weak or single signal
   - [UNKNOWN] — no supporting evidence. State it. Do not guess.

2. **Citation format** — every [VERIFIED–PUBLIC] claim must include:
   (Source: [publisher/platform], URL: [url], Date: [date if known])
   Every [VERIFIED–MEETING] claim must reference the meeting context.

3. **Every INFERRED–H must cite upstream signals** — state which evidence \
led to the inference in the same sentence. If you cannot, downgrade to INFERRED–M.

4. **No hallucination** — if you have no evidence for something, write \
"**No evidence available.**" An explicit gap is worth more than plausible fiction.

5. **No generic filler** — before writing any sentence, apply this test: \
"Could this sentence describe 50% of executives in this industry?" \
If yes, DELETE IT. Specific or nothing.

   BANNED phrases (unless directly quoting the subject with citation):
   - "strategic leader", "visionary", "thought leader"
   - "data-driven", "results-driven", "outcome-driven", "results-oriented"
   - "passionate about", "deeply committed to"
   - "transformative", "game-changing", "cutting-edge"
   - "proven track record", "extensive experience"
   - "empowers teams", "bridges the gap"
   - "at the intersection of", "holistic approach", "synergies"
   - "likely implements corrective measures"
   - "focuses on growth"

6. **Person-level > company-level** — if more than 40% of your output \
describes the company generically without connecting to THIS person's \
individual role, rewrite. This is a person dossier.

7. **Use the subject's own words** — when you have quotes, use them. \
Verbatim quotes in > blockquotes with source attribution.

8. **Distinguish source tiers** — primary sources (company site, \
registry, direct talks) outweigh secondary sources (news articles, \
conference pages). Flag low-quality sources.

9. **Recency matters** — prefer evidence from the last 24 months. \
Older evidence should be labeled with its date.

10. **Disambiguation** — note what identifiers lock the identity \
(company, LinkedIn URL, location, photo) and flag ambiguity risk."""

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

## REQUIRED OUTPUT: DECISION-GRADE CONTACT INTELLIGENCE DOSSIER

Produce ALL 10 sections below. Follow evidence rules strictly. \
If a section has no evidence, include the header with an explicit \
gap statement. Do NOT pad with generic language.

---

### 1. Strategic Identity Snapshot (5 bullets max, person-level)

Each bullet MUST mention {name} directly or use a personal pronoun. \
Answer ONLY these:
- What is {name}'s operating mode? (builder/optimizer/scaler/fixer)
- Where does {name} create or protect value?
- What public positioning does {name} emphasize?
- What incentive structure is visible for {name}?
- What constraints bind {name}?

No adjectives. Only structural observations with evidence tags. \
If more than 2 bullets do not mention {name}, rewrite.

---

### 2. Verified Fact Table

| # | Fact | Tag | Source | URL / Reference | Confidence |
|---|------|-----|--------|-----------------|------------|

Include: name, title, company, location, education, affiliations, \
certifications, quantified outcomes, career timeline events. \
Every row must have a source. Self-reported and unverified facts must be noted.

---

### 3. Incentive & Scorecard Model

Answer clearly with evidence:
- **What is {name} paid to optimize?** (revenue, margin, delivery, growth)
- **What metric hurts {name} most?** (churn, miss, delay, cost overrun)
- **What decision rights does {name} likely hold?** (budget, hiring, strategy, pricing)
- **Where must {name} escalate?** (board, CEO, committee, partner)
- **Short-term incentives** (0-3 months): What must be delivered NOW
- **Medium-term incentives** (3-12 months): What success looks like this year
- **Career incentives**: Where {name} wants to be in 2-3 years

Each must be tagged with evidence. If inferring, state upstream signals.

---

### 4. Structural Pressure Model

For EACH of the following, state: Evidence, Intensity (Low/Med/High), \
and What behavior it drives for {name}:

1. **Revenue Pressure**: Pipeline targets, quota, growth mandates
2. **Delivery Pressure**: Project timelines, capacity constraints, quality demands
3. **Political Pressure**: Internal stakeholders, competing priorities, org dynamics
4. **Reputation Risk**: Public commitments, market perception, personal brand
5. **Geographic Expansion Pressure**: International growth, new market entry (if applicable)

Each must cite evidence. If no evidence for a pressure type, say [UNKNOWN] \
and explain what signal would reveal it.

---

### 5. Power & Decision Rights Map

For each dimension, state what is known and tag it:
- **Formal authority**: Title, reporting line, org chart position
- **Informal influence**: Network, reputation, expertise leverage
- **Revenue control**: P&L ownership, budget authority, sales targets
- **Decision gate ownership**: What {name} can approve/veto
- **Who {name} needs to impress**: Board, CEO, investors, customers
- **Who can veto {name}**: Stakeholders with override power

If unknown, write [UNKNOWN] explicitly. Do not infer org charts.

---

### 6. Strategic Tensions

Identify live tensions from evidence. For EACH tension:
- State the tension
- Cite specific evidence that reveals it
- State the implication for {name}'s behavior

Examples: Growth vs capacity, Services vs product, Standardization vs \
customization, Innovation vs governance, Public optimism vs private caution.

---

### 7. Decision Consequence Forecast

For EACH scenario, output RANKED behavioral responses:

**Scenario: Revenue target slips**
Rank 1 response (most likely): [prediction + reasoning + evidence]
Rank 2: [prediction + reasoning]
Rank 3: [prediction + reasoning]
Confidence: [H/M/L] + reasoning

**Scenario: Delivery strain**
[Same format]

**Scenario: Client escalates**
[Same format]

**Scenario: Internal resistance**
[Same format]

**Scenario: Candidate challenges {name}'s assumptions**
[Same format]

Each prediction MUST cite upstream language or behavioral evidence. \
Generic executive responses are BANNED — if you write "likely to \
implement corrective measures," delete it.

---

### 8. Conversation Leverage Map

Produce:
- **3 leverage angles** — mapped to {name}'s specific incentive structure
- **2 stress tests** — pressure points that reveal real position
- **2 credibility builders** — what earns trust with {name} specifically
- **1 contrarian wedge** — intelligent challenge that earns respect

Each must reference the incentive or pressure that makes it effective.

---

### 9. Unknowns That Matter (with why they matter)

ONLY include gaps that materially change strategy. For EACH:
- State the unknown
- Explain why it matters strategically
- State what signal would resolve it

Examples: P&L ownership level, compensation structure, board reporting \
line, equity exposure, internal political dynamics, decision rights scope.

Do NOT include generic gaps. Every unknown must explain its strategic impact.

---

### 10. QA Report

Self-audit before finalizing:
- Evidence Coverage: What % of claims are tagged?
- Person-vs-Company Ratio: Is this about {name} or their company?
- Genericness check: Could any bullet describe 50% of leaders?
- INFERRED–H audit: Does every INFERRED–H cite upstream signals?
- Top 5 claims to verify next
- Missing retrieval gaps: What should be searched next?

---

## OUTPUT CONSTRAINTS
- Use markdown with clear ### section headers
- Use tables where specified
- Use > blockquotes for direct quotes (with source)
- Bold key terms and names
- Tag EVERY non-trivial claim
- Flag EVERY gap explicitly
- NO padding. Short and honest beats long and fabricated.
- If more than 40% is company recap, rewrite to focus on {name}.
- Every behavioral forecast must cite specific evidence.
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
            "Tag every claim sourced from training data as [INFERRED–L] "
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
