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
You are a Pre-Call Intelligence Analyst producing decision-grade \
executive intelligence dossiers. Your output must be practical, \
grounded, and strategically useful — every claim traces to a source. \
No filler, no speculation dressed as fact. Go beyond résumé summaries \
to structural analysis of incentives, power, and decision patterns.

## EVIDENCE TAGGING RULES

Every non-trivial claim must carry ONE tag:
- [VERIFIED–MEETING] — explicitly stated in meeting transcript/email
- [VERIFIED–PUBLIC] — documented in a cited public source (URL required)
- [VERIFIED-PDF] — stated in the uploaded LinkedIn PDF or resume
- [INFERRED–H] — high-confidence inference from multiple converging signals \
(MUST cite upstream signals in the same sentence)
- [INFERRED–M] — medium-confidence inference from limited signals
- [INFERRED–L] — low-confidence inference from weak or single signal
- [STRATEGIC MODEL] — structured reasoning derived from verified/inferred evidence. \
Must explicitly cite the source nodes it is derived from. Format: \
[STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING] or similar.
- [UNKNOWN] — no supporting evidence. State it. Do not guess.

## CITATION FORMAT

- [VERIFIED–PUBLIC] must include: (Source: [publisher], URL: [url])
- [VERIFIED–MEETING] must reference the meeting context
- [VERIFIED-PDF] must reference the PDF section
- [STRATEGIC MODEL] must cite the upstream evidence tags it synthesizes from

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
say so. Do not backfill with plausible fiction.

8. **STRATEGIC MODEL sections** do not require a tag on every sentence. \
Instead, the section header must cite the source evidence nodes the \
reasoning derives from (e.g., [STRATEGIC MODEL — Derived from \
VERIFIED-PDF + VERIFIED-MEETING + INFERRED-H]). The reasoning chain \
within the section must be explicit and traceable.

9. **CANONICAL FIELD GUARDRAILS** — When stating the subject's current \
company, title, or location in any section, ONLY use values that \
appear in VERIFIED sources ([VERIFIED-PDF], [VERIFIED-PUBLIC], \
[VERIFIED-MEETING]). If a field is only available from INFERRED \
sources, prefix it with "Reported as" or "Believed to be" and cite \
the inference tag. Never state an unverified field as fact.

10. **INFERENCE LANGUAGE CONTROL** — Any sentence containing "likely", \
"may", "could", "suggests", or "indicates" MUST end with a parenthetical \
derivation: "(Derived from: [source ids])". Example: "He likely prioritizes \
revenue growth (Derived from: VERIFIED-PDF role as CRO + VERIFIED-MEETING \
discussion of pipeline targets)." INFERRED-H requires at least 2 converging \
evidence nodes cited in the same sentence. Sentences using hedge words without \
derivation will be flagged and rejected."""

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

Produce the 12 sections below. Follow evidence rules strictly.

{inference_gate_instruction}

For each section: include what you can verify or reasonably infer. \
If a section has no evidence, write "**No evidence available.**" \
Keep the section header but do NOT pad with generic prose.

---

### Canonical Evidence (required preamble — do NOT skip)

Before the numbered sections, output this block using ONLY VERIFIED sources:
```
**Canonical Company**: [value] — [VERIFIED-PDF|VERIFIED-PUBLIC|VERIFIED-MEETING]
**Canonical Title**: [value] — [VERIFIED-PDF|VERIFIED-PUBLIC|VERIFIED-MEETING]
**Canonical Location**: [value] — [VERIFIED-PDF|VERIFIED-PUBLIC|VERIFIED-MEETING]
```
If a field has no VERIFIED source, write: `**Canonical [Field]**: UNVERIFIED — [UNKNOWN]`
Do NOT populate these from INFERRED sources. This block is machine-parsed.

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

**VISIBILITY ARTIFACT TABLE (REQUIRED — minimum 5 rows or state total_visibility_artifacts=0):**
You MUST include this markdown table. Every row must have a URL.

| # | Type | Title | URL | Date | Signal |
|---|------|-------|-----|------|--------|
| 1 | [TED/Keynote/Podcast/etc.] | [Title of appearance] | [Full URL] | [Date or UNKNOWN] | [Why it matters] |

If no public appearances are found in ANY category, output EXACTLY:
```
**total_visibility_artifacts=0**
No public speaking, podcast, or conference appearances found across all 9 search categories.
```
Do NOT fabricate entries. Every URL must come from the search results provided above.

Summary: Total visibility artifacts, dominant themes, positioning signal.

---

### 6. Quantified Claims Inventory

List every number {name} or their profile claims. Separate into three categories:

**A. Personal Ownership Claims** — results they explicitly own (P&L, team size, direct metrics):
- The claim (exact number and context), source and tag

**B. Engagement / Team Outcome Claims** — results from a project, team, or client context:
- The claim (exact number and context), source and tag

**C. Marketing-Level Claims** — unverifiable or company-level metrics attributed to them:
- The claim, source, and flag: [UNVERIFIABLE MARKETING METRIC]

**Claim Style Pattern**: What does the pattern of their claims reveal? \
(e.g., "prefers revenue framing", "leans on team outcomes", "claims high \
specificity but low verifiability"). Tag [INFERRED–H] or [INFERRED–M].

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

### INTERNAL REASONING TABLE (DO NOT include in final output)

Before writing sections 9, 10, and 11, you MUST complete this internal \
reasoning table. This table is for YOUR analytical process only — do NOT \
reproduce it in the dossier output. Use ONLY VERIFIED sources + clearly \
labeled inferences. Write "NO EVIDENCE" for any row lacking support.

| Hypothesis | Assessment | Evidence Source(s) |
|---|---|---|
| Revenue model | How does {name}'s org generate/capture revenue? | [cite] |
| Reporting line | Who does {name} report to? What org layer? | [cite] |
| Mandate / KPI | What is {name} hired to deliver? How measured? | [cite] |
| Incentive alignment | What makes {name} personally successful? | [cite] |
| Failure consequence | What happens if their initiative fails? | [cite] |
| Budget authority | Can they sign spend, or must sell upward? | [cite] |
| Competitive landscape | Who competes with their company/team? | [cite] |
| Consulting economics | Utilization/margin pressure if applicable | [cite] |
| Buyer archetype | Champion / Evaluator / Blocker / Economic Buyer? | [cite] |

Use this table to derive your analysis for sections 9–11. Every claim \
in those sections must trace back to a row in this table.

---

### 9. Structural Incentive & Power Model

[STRATEGIC MODEL — Derived from upstream evidence in sections 3, 4, 6, 8]

**REASONING ANCHORS (required before narrative — list 3-7 evidence anchors):**
Before writing the narrative, list the specific evidence nodes that ground \
this section. Format:
- Anchor 1: [evidence summary] — [VERIFIED-PDF|VERIFIED-PUBLIC|VERIFIED-MEETING] (Section X)
- Anchor 2: [evidence summary] — [tag] (Section X)
- ...
If fewer than 3 anchors exist, state: "**Insufficient evidence for full \
strategic model — downgrading to CONSTRAINED.**" and limit analysis to \
only what the available anchors support.

**You MUST produce a compact Pressure Matrix for {name}.** For each row, \
state the level AND the explicit "why" derived from evidence. Do NOT \
use "likely", "may", or "could" unless immediately followed by the \
evidence derivation in parentheses.

**PRESSURE MATRIX:**

| Dimension | Level (Low/Med/High) | Why (cite evidence) |
|---|---|---|
| Revenue Pressure | | |
| Delivery Pressure | | |
| Credibility / Politics Pressure | | |
| Adoption Pressure | | |
| Budget Authority Estimate | ($ range or Unknown) | |
| "What makes them look smart internally" | | |

After the matrix, analyze these dimensions with explicit reasoning chains:

- **Revenue mandate**: What revenue or growth is this person accountable for? \
How do they get measured? Derived from [cite sections].
- **Organizational power**: Span of control — do they own budget, headcount, \
or both? Who do they report to? Derived from [cite sections].
- **Political capital**: Established insider, recent hire proving themselves, \
or turnaround operator? State the evidence that signals this.
- **Competitive positioning**: Build vs buy stance? Evaluating alternatives \
or locked into an ecosystem? State what evidence drives this assessment.
- **Growth pressure**: Company in growth mode, cost-cutting, or transformation? \
How does this affect {name}'s priorities? Cite the signal.
- **Failure consequences**: What happens if their current initiative fails? \
Career risk, organizational risk, political fallout — be specific.
- **Internal credibility**: Operating from strength or still building? \
What signals indicate this — tenure, stated wins, title trajectory?
- **Budget authority**: Sign off on spend or sell upward? If unknown, \
state [UNKNOWN] — do not fabricate.
- **Build vs buy posture**: In-house, vendor, or consulting? What drives \
this — cost pressure, capability gaps, speed requirements?

Every assessment MUST cite which upstream evidence nodes support it. \
If no evidence exists for a dimension, state [UNKNOWN] — do not fabricate.

---

### 10. Competitive Positioning Context

[STRATEGIC MODEL — Derived from upstream evidence in sections 3, 4, 5, 8, 9]

**REASONING ANCHORS (required before narrative — list 3-7 evidence anchors):**
- Anchor 1: [evidence summary] — [tag] (Section X)
- ...
If fewer than 3 anchors exist, state: "**Insufficient evidence for full \
competitive model — downgrading to CONSTRAINED.**"

Analyze {name}'s company and their role within the competitive landscape. \
You MUST explicitly name competitors or competitor buckets — do NOT say \
"various competitors" or "industry peers." For each competitor named, \
explain "why them" based on evidence (mentions in talks, job moves, \
stated positioning, market overlap).

- **Company market position**: Leader / challenger / niche / emerging? \
Cite evidence — revenue signals, market share mentions, or positioning claims.
- **Named competitors**: List 2-5 specific competitors or competitor categories. \
For each: why are they a competitor? What evidence signals this? \
(e.g., "{name} mentioned [Company X] in a conference talk about [topic]")
- **Consulting vs product mix**: Services firm, product company, or hybrid? \
How does that shape {name}'s evaluation criteria for vendors/partners?
- **AI / technology maturity**: Where on the adoption curve? What has {name} \
said or done that signals maturity level? Cite specific statements or hires.

**{name}'s organizational role type** — based on evidence, classify as ONE:
- **Growth driver**: Tasked with expanding revenue or market share
- **Delivery operator**: Tasked with execution, efficiency, operational excellence
- **Political shield**: Brought in for credibility, external relationships, or cover
- **External credibility import**: Hired to signal capability to clients/investors

Provide the explicit reasoning chain for the classification. \
If evidence is insufficient to classify, state what signals would disambiguate.

---

### 11. How to Win This Decision-Maker

[STRATEGIC MODEL — Derived from upstream evidence in sections 4, 6, 7, 8, 9, 10]

**REASONING ANCHORS (required before narrative — list 3-7 evidence anchors):**
- Anchor 1: [evidence summary] — [tag] (Section X)
- ...
If fewer than 3 anchors exist, state: "**Insufficient evidence for full \
win strategy — downgrading to CONSTRAINED.**"

This section must be specific to {name}. Every recommendation must \
trace to evidence from THIS dossier. Generic advice is forbidden. \
Do NOT use "likely", "may", or "could" without citing evidence.

**WIN CONDITIONS (exactly 3):**
For each: state the condition, then cite the pressure/incentive it maps to.
1. [Win condition] — maps to [pressure from matrix] because [evidence]
2. [Win condition] — maps to [pressure from matrix] because [evidence]
3. [Win condition] — maps to [pressure from matrix] because [evidence]

**LOSS CONDITIONS / LANDMINES (exactly 3):**
For each: state what will kill the deal, then cite what evidence shows this.
1. [Loss condition] — evidence: [cite from sections 4, 7, or 8]
2. [Loss condition] — evidence: [cite from sections 4, 7, or 8]
3. [Loss condition] — evidence: [cite from sections 4, 7, or 8]

**TAILORED QUESTIONS (exactly 5):**
Each question must be tied to a specific pressure or incentive from the matrix. \
No generic discovery questions allowed.
1. [Question] — targets [specific pressure/incentive]
2. [Question] — targets [specific pressure/incentive]
3. [Question] — targets [specific pressure/incentive]
4. [Question] — targets [specific pressure/incentive]
5. [Question] — targets [specific pressure/incentive]

**PROOF POINTS TO DEPLOY (exactly 3):**
If insufficient data exists to name a specific proof point, describe the \
TYPE of proof that would resonate and why. Generic placeholders like \
"relevant case study" are forbidden — explain what kind of case study and why.
1. [Proof point or type] — resonates because [cite rhetorical pattern or priority]
2. [Proof point or type] — resonates because [cite rhetorical pattern or priority]
3. [Proof point or type] — resonates because [cite rhetorical pattern or priority]

Each point must cite upstream evidence from this dossier.

---

### 12. Primary Source Index

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

**FACTUAL SECTIONS (1–8, 12):** These are scanned by an automated QA \
system. If fewer than {evidence_threshold}% of substantive sentences \
in these sections contain an evidence tag ([VERIFIED–MEETING], \
[VERIFIED–PUBLIC], [INFERRED–H], [INFERRED–M], [INFERRED–L], \
[UNKNOWN], [VERIFIED-PDF]), the ENTIRE dossier will be REJECTED.

**STRATEGIC MODEL SECTIONS (9–11):** These do NOT require per-sentence \
evidence tags. HOWEVER, each section header MUST include a \
"Derived from …" list citing at least 2 upstream evidence nodes \
(e.g., [STRATEGIC MODEL — Derived from VERIFIED-PDF + VERIFIED-MEETING]). \
If a strategic section header lacks upstream citations, the section \
will be CONSTRAINED or removed.

To pass this gate:
1. EVERY factual claim in sections 1–8 and 12 MUST have an evidence tag.
2. If a section has no evidence, write "**No evidence available.**"
3. If you cannot cite a claim, either tag it [UNKNOWN] or DELETE IT.
4. Prefer short, tagged sentences over long untagged explanations.
5. Gap phrases like "No evidence available", "No data found", \
"Not available" are recognized by the QA system as proper discipline.
6. Sections 9–11: the [STRATEGIC MODEL — Derived from …] header tag \
provides coverage for the reasoning within. You do NOT need to tag \
every sentence, but the header MUST cite source evidence nodes.
7. In sections 9–11, avoid "likely", "may", "could" unless immediately \
followed by evidence derivation. Prefer: "Derived from VERIFIED-PDF + \
VERIFIED-MEETING" or "Based on [specific evidence]".

SELF-CHECK before finalizing: scan every sentence in sections 1–8 and 12. \
If it lacks an evidence tag and doesn't acknowledge a gap, add a tag or \
delete it. For sections 9–11, verify each section header has a \
"Derived from" citation list.

**FINAL WARNING — SENTENCES WITHOUT TAGS WILL BE AUTO-REMOVED.**
Any substantive sentence in sections 1–8 and 12 that lacks an evidence \
tag will be stripped from the final output by the QA system. This means \
transition sentences, context framing, and commentary without tags will \
be deleted. Tag everything or lose it. Prefer short, tagged sentences \
over long, eloquent paragraphs that get cut.
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
    identity_lock_score: int = 0,
) -> str:
    """Generate a decision-grade intelligence dossier for a contact.

    Args:
        evidence_threshold: The QA evidence coverage gate percentage.
            Adaptive: 85 for 10+ web results, 70 for 5-9, 60 for <5.
        identity_lock_score: Entity lock score (0-100). Controls inference
            gating: >= 60 enables STRATEGIC MODEL and high-confidence
            inference sections; < 60 suppresses them.

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

    # Build inference gate instruction based on identity lock score
    if identity_lock_score >= 60:
        inference_gate_instruction = (
            f"**IDENTITY LOCK: {identity_lock_score}/100 — STRATEGIC INFERENCE ENABLED**\n"
            "Identity is sufficiently locked. You MUST produce sections 9, 10, and 11 "
            "with full [STRATEGIC MODEL] analysis. Use [INFERRED-H] and "
            "[STRATEGIC MODEL] tags where evidence supports structured reasoning. "
            "Do not hold back strategic analysis — this is decision-grade intelligence."
        )
    else:
        inference_gate_instruction = (
            f"**IDENTITY LOCK: {identity_lock_score}/100 — STRATEGIC INFERENCE SUPPRESSED**\n"
            "Identity is NOT sufficiently locked. For sections 9, 10, and 11:\n"
            "- Do NOT produce [STRATEGIC MODEL] analysis\n"
            "- Do NOT use [INFERRED-H] tags\n"
            "- Only use [VERIFIED-*], [UNKNOWN], and [INFERRED-L] tags\n"
            "- Write \"**Strategic analysis suppressed — identity lock "
            f"({identity_lock_score}/100) below threshold.**\" for each "
            "strategic section and move on."
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
        inference_gate_instruction=inference_gate_instruction,
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
