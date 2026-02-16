"""Decision-grade contact intelligence dossier generation.

Produces practical, grounded intelligence for meeting preparation.
Every claim is evidence-tagged (VERIFIED/INFERRED/UNKNOWN).
Sections are built from observable data: career timeline, public
statements, quantified claims, rhetorical patterns, and actionable
interview strategy.

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
You are a Pre-Call Intelligence Analyst producing evidence-backed \
contact dossiers for meeting preparation. Your output must be \
practical and grounded — every claim traces to a source. \
No filler, no speculation dressed as fact.

## EVIDENCE TAGGING RULES

Every non-trivial claim must carry ONE tag:
- [VERIFIED–MEETING] — explicitly stated in meeting transcript/email
- [VERIFIED–PUBLIC] — documented in a cited public source (URL required)
- [VERIFIED-PDF] — stated in the uploaded LinkedIn PDF or resume
- [INFERRED–H] — high-confidence inference from multiple converging signals \
(MUST cite upstream signals in the same sentence)
- [INFERRED–M] — medium-confidence inference from limited signals
- [INFERRED–L] — low-confidence inference from weak or single signal
- [UNKNOWN] — no supporting evidence. State it. Do not guess.

## CITATION FORMAT

- [VERIFIED–PUBLIC] must include: (Source: [publisher], URL: [url])
- [VERIFIED–MEETING] must reference the meeting context
- [VERIFIED-PDF] must reference the PDF section

## ABSOLUTE RULES

1. **ZERO hallucination** — if you have no evidence, write \
"**No evidence available.**" An explicit gap is worth more than plausible fiction.

2. **No generic filler** — test each sentence: "Could this describe \
50% of executives in this industry?" If yes, DELETE IT. Specific or nothing.

   BANNED phrases (unless directly quoting the subject with citation):
   "strategic leader", "visionary", "thought leader", "data-driven", \
   "results-driven", "passionate about", "deeply committed to", \
   "transformative", "game-changing", "cutting-edge", "proven track record", \
   "empowers teams", "bridges the gap", "at the intersection of", \
   "synergies", "likely implements corrective measures"

3. **Person-level > company-level** — this is a person dossier. \
If more than 40% describes the company without connecting to THIS \
person's role, rewrite.

4. **Use their own words** — when you have quotes, use them. \
Verbatim quotes in > blockquotes with source attribution.

5. **Recency matters** — prefer evidence from the last 24 months. \
Older evidence should be labeled with its date.

6. **Disambiguation** — flag name-collision risks. Note what identifiers \
lock the identity (company, LinkedIn URL, location).

7. **FAIL-CLOSED** — never contradict yourself. If evidence is missing, \
say so. Do not backfill with plausible fiction."""

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

Produce the 10 sections below. Follow evidence rules strictly.

For each section: include what you can verify or reasonably infer. \
If a section has no evidence, write "**No evidence available.**" \
Keep the section header but do NOT pad with generic prose.

---

### 1. Executive Summary

**VERIFIED** (2-4 bullets of confirmed facts):
- Current role, company, location, key identifiers

**INFERRED** (1-2 bullets of decision-useful inference):
- Operating mode, likely evaluation criteria, key priorities

Each bullet MUST mention {name} directly. No adjectives without evidence.

---

### 2. Identity & Disambiguation

**VERIFIED identifiers:**
- Name variants used publicly, location, role, LinkedIn handle

**Disambiguation risks:**
- Note any name collisions found in search results
- State what identifiers lock this as the correct person

---

### 3. Career Timeline

Chronological list of roles, most recent first. For EACH:
- **Company** — Title (Start date–End date)
- Tag each entry [VERIFIED-PDF], [VERIFIED–PUBLIC], or [INFERRED–M]

Include: education, certifications, board seats if found.

---

### 4. Public Statements & Positions

Organize by TOPIC (not by source). For each topic cluster:
- **Topic** (e.g., "AI philosophy", "ROI framing", "Governance stance")
- **VERIFIED**: Direct quotes or attributed language with source
- **INFERRED**: What the pattern implies for how they evaluate/decide

Use > blockquotes for direct quotes with source attribution. \
This section reveals how {name} THINKS — prioritize their own words.

---

### 5. Public Visibility

For EACH visibility category (TED/TEDx, Keynote, Conference, Summit, \
Podcast, Webinar, YouTube, Panel, Interview):
- State results found or "No appearances found in search sweep"

Summary: Total visibility artifacts, dominant themes, positioning signal.

---

### 6. Quantified Claims Inventory

List every number {name} or their profile claims. For EACH:
- The claim (exact number and context)
- Source and tag
- Whether it reads as personally owned P&L or engagement/team outcome

Flag: quantified bullets that read like engagement outcomes rather than \
personally owned results. Note this for interview calibration.

---

### 7. Rhetorical & Decision Patterns

Based on repeated phrasing and stated positions:
- **Language bias**: What frames does {name} default to? \
(e.g., pragmatism, platform-thinking, ROI-first, adoption-obsessed)
- **Decision style**: How do they likely evaluate proposals? \
(e.g., structured → constraints → architecture → measurement)
- **Red flags for them**: What will make them dismiss you?

Tag each observation with evidence source.

---

### 8. Structural Pressure Model

What likely drives {name}'s decisions based on evidence:
- **Current mandate**: What are they paid to deliver right now?
- **Key pressures** (2-4): Revenue, delivery, credibility, adoption — \
cite evidence for each, state intensity (Low/Med/High)
- **Vendor posture**: Single-vendor vs multi-ecosystem? \
Portability preference? Risk appetite?

Each must cite evidence. If no evidence, state [UNKNOWN].

---

### 9. Interview Strategy Recommendations

Based on the above analysis:

**What to lead with** (high leverage):
- 2-3 specific angles that align with their stated priorities

**What they will likely probe**:
- 2-3 topics they'll push on based on their background

**Landmines** (what NOT to say):
- 2-3 framings or claims that will lose credibility

**Questions that will earn respect**:
- 3-4 specific questions that demonstrate you understand their world

Each must reference specific evidence from the dossier.

---

### 10. Primary Source Index

List all sources used with URLs:
- Provided artifacts (PDF, uploaded documents)
- LinkedIn profile
- Web search results used
- Third-party references (note reliability)

---

## OUTPUT CONSTRAINTS
- Use markdown with clear ### section headers
- Use tables where specified (Career Timeline, Quantified Claims)
- Use > blockquotes for direct quotes (with source)
- Bold key terms and names
- Tag EVERY non-trivial claim
- Flag EVERY gap explicitly
- NO padding. Short and honest beats long and fabricated.
- Every inference must cite what evidence drove it.

## CRITICAL: EVIDENCE COVERAGE GATE ({evidence_threshold}%)
Your output will be scanned by an automated QA system. If fewer than \
{evidence_threshold}% of substantive sentences contain an evidence tag \
([VERIFIED–MEETING], [VERIFIED–PUBLIC], [INFERRED–H], [INFERRED–M], \
[INFERRED–L], [UNKNOWN], or [VERIFIED-PDF]), the ENTIRE dossier will \
be REJECTED and generation halts.

To pass this gate:
1. EVERY factual claim MUST have an evidence tag in square brackets.
2. If a section has no evidence, write "**No evidence available.**"
3. If you cannot cite a claim, either tag it [UNKNOWN] or DELETE IT.
4. Prefer short, tagged sentences over long untagged explanations.
5. Gap phrases like "No evidence available", "No data found", \
"Not available" are recognized by the QA system as proper discipline.

SELF-CHECK before finalizing: scan every sentence. If it lacks an \
evidence tag and doesn't acknowledge a gap, add a tag or delete it.
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
    evidence_threshold: int = 85,
) -> str:
    """Generate a decision-grade intelligence dossier for a contact.

    Args:
        evidence_threshold: The QA evidence coverage gate percentage.
            Adaptive: 85 for 10+ web results, 70 for 5-9, 60 for <5.

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
        evidence_threshold=evidence_threshold,
    )

    llm = LLMClient()
    return llm.chat(SYSTEM_PROMPT, user_prompt, temperature=0.3)


def build_interactions_summary(profile_data: dict) -> str:
    """Build a text summary of internal interactions for the LLM prompt.

    Includes meeting details, behavioral observations, action items,
    and participant context. Richer detail produces better dossiers.
    """
    parts = []

    interactions = profile_data.get("interactions", [])
    if interactions:
        parts.append(f"We have {len(interactions)} recorded interactions:\n")
        for ix in interactions[:15]:
            ix_type = ix.get("type", "meeting").upper()
            ix_title = ix.get("title", "Untitled")
            ix_date = ix.get("date", "Unknown date")
            ix_summary = ix.get("summary", "")
            participants = ix.get("participants", [])

            parts.append(f"- [{ix_type}] {ix_date}: {ix_title}")
            if participants:
                parts.append(f"  Participants: {', '.join(participants[:8])}")
            if ix_summary:
                parts.append(f"  Summary: {ix_summary}")

            # Include key points / bullet gist if available
            key_points = ix.get("key_points", "")
            if key_points:
                parts.append(f"  Key points: {key_points}")
            bullet_gist = ix.get("bullet_gist", "")
            if bullet_gist:
                parts.append(f"  Details: {bullet_gist}")

            parts.append("")  # blank line between interactions

    action_items = profile_data.get("action_items", [])
    if action_items:
        parts.append(f"Open action items ({len(action_items)}):")
        for item in action_items[:10]:
            parts.append(f"- {item}")

    if not parts:
        return "No internal meeting or email history available."

    return "\n".join(parts)
