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
You are a Strategic Intelligence Analyst producing FAIL-CLOSED, \
evidence-backed contact intelligence dossiers. \
Your output must create LEVERAGE, not summaries. \
Every claim must trace to an EvidenceNode. No exceptions.

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

4. **ZERO hallucination** — if you have no evidence for something, write \
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
   - "focuses on growth", "consultative tempo", "ROI-focused"
   - "human-centered"

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
(company, LinkedIn URL, location, photo) and flag ambiguity risk.

11. **FAIL-CLOSED** — never contradict yourself. If you state the sweep \
was not executed, you cannot later claim results from it. If evidence \
is missing, say so. Do not backfill with plausible fiction.

12. **GENERICNESS LINTER** — reject or rewrite sentences containing \
"data-driven", "strategic leader", "results-oriented", "human-centered", \
"consultative tempo", "ROI-focused" UNLESS they cite a verbatim phrase \
from an EvidenceNode."""

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

{visibility_research}

---

## REQUIRED OUTPUT: CONTACT INTELLIGENCE DOSSIER

Produce the sections below. Follow evidence rules strictly. \
**ADAPTIVE SECTIONS**: Only produce sections where you have evidence. \
Sections 1-3 are always required. Sections 4-11 are OPTIONAL — \
SKIP any section where you have no evidence rather than filling it \
with [UNKNOWN] tags or generic speculation. A 5-section dossier with \
high evidence coverage is far better than an 11-section dossier full \
of gaps and [UNKNOWN] tags that fails QA.

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

### 3. Public Visibility Report

Summarise the public visibility sweep results. For EACH category \
where results were found, provide:
- **Category** (TED/TEDx/Keynote/Conference/Summit/Podcast/Webinar/YouTube/Panel/Interview)
- **Title** of the appearance
- **URL** (cite it)
- **Key takeaway** — what position or message did {name} project?
- **Evidence tag** — [VERIFIED–PUBLIC] with URL

For categories with NO results, state "[UNKNOWN] — no appearances \
found in search sweep for this category."

Summary metrics:
- Total visibility artifacts found
- Which categories returned results
- Public positioning signal (what themes appear across appearances)

---

### 4. Incentive & Pressure Model

**Incentive Model:**
- **Paid to optimize**: (revenue, margin, delivery, growth) — cite evidence
- **Metric that hurts most**: (churn, miss, delay, cost overrun) — cite evidence
- **Decision rights**: (budget, hiring, strategy, pricing) — cite evidence
- **Escalation path**: (board, CEO, committee, partner) — cite evidence
- **Failure mode**: What happens when {name}'s key metric misses?

**Structural Pressure Model:**
For EACH, state: Evidence, Intensity (Low/Med/High), Behavior it drives:
1. **Revenue Pressure**: Pipeline targets, quota, growth mandates
2. **Delivery Pressure**: Project timelines, capacity constraints, quality demands
3. **Political Pressure**: Internal stakeholders, competing priorities, org dynamics
4. **Reputation Risk**: Public commitments, market perception, personal brand
5. **Geographic Expansion Pressure**: International growth, new market entry

Each must cite evidence. If no evidence, say [UNKNOWN] and explain \
what signal would reveal it.

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

### 8. Deal Probability Score

Calculate a 0-100 deal probability score with these weighted factors. \
For EACH factor, state your score, the weight range, and evidence:

**Positive factors:**
- Incentive alignment (0-20): How well does our offering align with {name}'s incentives?
- Authority scope (0-15): Does {name} have authority to act?
- Budget / P&L influence (0-15): Does {name} control relevant budget?
- Pressure alignment (0-15): Do current pressures favour our proposition?
- Public positioning (0-10): Does {name}'s public stance support engagement?

**Negative factors (subtract):**
- Political friction risk (0-15): Internal resistance, competing stakeholders
- Competing priorities (0-10): Other initiatives that drain attention/budget

**Output format:**
| Factor | Score | Range | Evidence |
|--------|-------|-------|----------|
(one row per factor)

**Total: [SUM]/100** | Confidence: [H/M/L]

---

### 9. Influence Strategy Recommendation

Based on the above analysis, produce:
- **Primary leverage**: The single strongest angle to open with
- **Secondary leverage**: Backup approach if primary doesn't land
- **Message framing bias**: How to frame value (growth vs risk vs efficiency)
- **Psychological tempo**: Fast close / slow build / consultative
- **Pressure points** (2-3): Specific pressures we can address
- **Avoidance points** (2-3): Topics or framings to avoid
- **Early warning signs** (2-3): Signals that {name} is disengaging or stalling

Each must reference specific evidence from the dossier.

---

### 10. Unknowns That Matter (with why they matter)

ONLY include gaps that materially change strategy. For EACH:
- State the unknown
- Explain why it matters strategically
- State what signal would resolve it
- State the exact question to ask

Examples: P&L ownership level, compensation structure, board reporting \
line, equity exposure, internal political dynamics, decision rights scope.

Do NOT include generic gaps. Every unknown must explain its strategic impact.

---

### 11. QA Report

Self-audit before finalizing:
- Evidence Coverage: What % of claims are tagged?
- Person-vs-Company Ratio: Is this about {name} or their company?
- Genericness check: Could any bullet describe 50% of leaders?
- INFERRED–H audit: Does every INFERRED–H cite upstream signals?
- Public Visibility Sweep: Were all 10 categories searched?
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

## CRITICAL: EVIDENCE COVERAGE GATE (85%)
Your output will be scanned by an automated QA system. If fewer than 85% \
of substantive sentences contain an evidence tag ([VERIFIED–MEETING], \
[VERIFIED–PUBLIC], [INFERRED–H], [INFERRED–M], [INFERRED–L], [UNKNOWN], \
or [VERIFIED-PDF]), the ENTIRE dossier will be REJECTED and generation halts.

To pass this gate:
1. EVERY factual claim MUST have an evidence tag in square brackets.
2. If a section has no evidence, write "**No evidence available.**" — \
do NOT write untagged prose explaining what evidence would look like.
3. If you cannot cite a claim, either tag it [UNKNOWN] or DELETE IT.
4. Prefer short, tagged sentences over long untagged explanations.
5. When describing gaps, use explicit language: "No evidence available", \
"No data found", "Not available" — these are recognized by the QA system.
6. OMIT entire sections if you have no evidence rather than writing \
untagged structural prose. A section header with "**No evidence available.**" \
is better than five untagged sentences.

SELF-CHECK before finalizing: scan every sentence in your output. \
If it lacks an evidence tag and doesn't acknowledge a gap, either add \
a tag or delete the sentence. An 85% coverage failure means your \
work is wasted.
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
    visibility_research: str = "",
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

    if not visibility_research:
        visibility_research = (
            "## PUBLIC VISIBILITY SWEEP RESULTS\n"
            "No visibility sweep was executed. All 10 categories remain unsearched."
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
        visibility_research=visibility_research,
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
