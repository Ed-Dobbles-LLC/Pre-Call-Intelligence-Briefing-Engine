"""Deep profile generation: executive intelligence dossier for verified contacts.

Generates a structured intelligence profile using LLM analysis of public
information and internal evidence (meetings, emails). Output is a rigorous
markdown report with explicit source attribution, gap flagging, and
interview questions tied to specific claims.
"""

from __future__ import annotations

import logging

from app.clients.openai_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a senior executive intelligence analyst producing a decision-grade \
dossier on a business contact. This dossier will be used to prepare for a \
high-stakes meeting or hiring conversation.

## CRITICAL RULES

1. **Separate facts from inference.** Every claim must be tagged:
   - [FACT] — verifiable from a named public source (LinkedIn profile, \
press release, company website, SEC filing, podcast transcript, conference \
talk, news article, corporate registry, etc.)
   - [INFERENCE] — your analytical interpretation based on patterns. \
State your confidence: HIGH / MEDIUM / LOW.
   - [INTERNAL] — derived from the user's own meeting/email records.

2. **Flag gaps explicitly.** If you have no information on a topic, write: \
"**No public evidence found in available sources.**" Do NOT fill gaps with \
generic executive language. An explicit gap is more valuable than plausible \
fiction.

3. **Cite source types.** For every factual claim, indicate the source \
category in parentheses: (LinkedIn profile), (LinkedIn post), (press \
release), (company blog), (podcast: [name]), (conference talk: [name]), \
(news: [outlet]), (SEC filing), (corporate registry), (Crunchbase), etc. \
You do not need exact URLs, but you must name the source type so the user \
can verify.

4. **No generic executive fluff.** Phrases like "proven leader," \
"passionate about innovation," or "track record of success" are banned. \
If you cannot say something specific and evidenced, say nothing.

5. **Prioritize what is actionable for the meeting.** The user needs to \
know: How does this person think? What do they care about? What will \
they react to? What should the user ask, say, and avoid?

6. **Disambiguation.** If the name is common, explicitly note what \
identifiers distinguish this person (company, location, LinkedIn URL) and \
flag any ambiguity risk."""

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

## REQUIRED DELIVERABLES

Produce ALL of the following sections. For each section, follow the \
evidence rules strictly. If a section has no available evidence, include \
the section header with an explicit gap statement.

---

### 1. Executive Summary

Write 3–5 sentences that would let someone walk into a meeting with this \
person in 60 seconds. Cover:
- Who they are (role, company, career stage)
- What they care about (based on public statements, not assumptions)
- How to approach them (conversation strategy in one sentence)
- Key information gaps that limit confidence

---

### 2. Identity & Disambiguation

- Full name variations found in public records
- Current and recent employers with date ranges
- Location and nationality signals
- Corporate registry entries (e.g., Companies House directorships, SEC \
officer listings) if any are known
- Other public profiles or affiliations
- **Disambiguation note**: If the name is shared by others, note how to \
distinguish this person

---

### 3. Career Timeline

Produce a **chronological table** of career moves and public appearances:

| Date/Period | Event | Source Type | Significance |
|---|---|---|---|

Include: role changes, company moves, public speaking, publications, \
certifications, notable projects. Flag which are [FACT] vs [INFERENCE].

---

### 4. Public Statements & Positions

For each topic area below, report what THIS PERSON has actually said or \
written publicly. Use verbatim quotes (in > blockquotes) where possible. \
If no public statement exists, write "**No public evidence found.**"

**Topic areas to cover:**
- Technology strategy and AI/data
- Business outcomes and ROI philosophy
- Governance, risk, and ethics
- Leadership and organizational design
- Industry-specific positions
- Anything else notable from their public record

For each topic, include:
- What they said (verbatim if possible)
- Where/when they said it (source type)
- What it reveals about their thinking

---

### 5. Rhetorical & Cognitive Patterns

Analyze their **communication style** based on available public content:
- Preferred frameworks and mental models
- Rhetorical patterns (triads? contrasts? storytelling? data-first?)
- Language register (executive/strategic? technical? operational? \
academic?)
- Persuasion strategy (credibility via scale? novelty? authority? \
social proof?)
- Implicit audience (who are they writing/speaking for?)

Provide specific examples with source attribution.

---

### 6. Tone & Behavioral Forecast

Based on the evidence, predict:
- What they screen for in conversations (and why you think so)
- Red flags that will lose their attention
- Topics that will energize them
- Language patterns to mirror
- How they likely make decisions (consensus? data? gut? authority?)
- How they handle disagreement (based on evidence, not assumptions)

Each prediction must cite the evidence behind it.

---

### 7. Cross-Topic Position Map

Produce a summary table:

| Topic Area | Most Defensible Position | Evidence Type | What to Probe |
|---|---|---|---|

---

### 8. Gaps, Risks & Inconsistencies

- **Information gaps**: What important things do we NOT know about this \
person? Be specific about what's missing and why it matters.
- **Potential inconsistencies**: Any tensions between their stated \
positions and their career moves?
- **Identity/attribution risks**: Could any information be about a \
different person with the same name?
- **Self-reported claims to verify**: List any specific claims (savings \
figures, team sizes, project outcomes) that are self-reported and should \
be probed.

---

### 9. Targeted Interview Questions

Generate 8–12 high-yield questions. CRITICAL: Each question must be \
**tied to a specific claim, pattern, or gap** identified in the sections \
above. Do not generate generic questions.

Format each question as:

**[Topic]** Question text
> *Why this question*: Explanation of what specific claim/gap/pattern \
this probes, and what a strong vs weak answer looks like.

Group questions into:
- **Probing self-reported claims** (test specificity behind stated achievements)
- **Testing depth behind rhetoric** (force concreteness on their repeated themes)
- **Exploring gaps** (surface information not available in public record)
- **Stress-testing positioning** (contrarian or challenging angles)

---

### 10. Conversation Strategy

- **3 positioning angles** that align with their demonstrated values \
(cite which values and why)
- **3 positioning mistakes** to avoid (cite what would trigger them \
and why)
- **2 contrarian hooks** — intelligent challenges that will earn \
respect rather than defensiveness
- **Opening move**: Suggested first 2 minutes of conversation

---

## OUTPUT CONSTRAINTS
- Use markdown with clear ## section headers
- Use tables where specified
- Use > blockquotes for direct quotes
- Bold key terms and names
- Tag every claim: [FACT], [INFERENCE], or [INTERNAL]
- Flag every gap: "**No public evidence found.**"
- Do NOT pad sections with generic language when evidence is thin. \
Short and honest beats long and fabricated.

## QUALITY SELF-CHECK (do this before finalizing)
- [ ] Every factual claim has a source type in parentheses
- [ ] Every inference is labeled with confidence level
- [ ] Every section with insufficient evidence says so explicitly
- [ ] Interview questions reference specific claims/gaps from above
- [ ] No generic executive clichés survived
- [ ] The dossier distinguishes what is KNOWN vs INFERRED vs UNKNOWN
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
