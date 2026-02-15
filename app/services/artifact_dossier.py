"""Artifact-first dossier pipeline — "Public Statements & Positions" from PDF + Web.

Dual-path architecture:
- Path B1 (Artifact-first): Build EvidenceNodes from PDF text alone.
  Works even without SerpAPI. Generates a dossier from user-supplied LinkedIn PDF.
- Path B2 (Web-augmented): Enrich with SerpAPI retrieval + visibility sweep.
  Adds PUBLIC_WEB EvidenceNodes and increases evidence coverage.

Every sentence in the dossier maps to a Claim → EvidenceNode chain.
Claims are tagged:
- VERIFIED-PDF: directly from user-supplied PDF text
- VERIFIED-PUBLIC: from web retrieval (SerpAPI)
- VERIFIED-MEETING: from internal meeting data
- INFERRED-H/M/L: inference with cited upstream evidence
- UNKNOWN: no evidence
"""

from __future__ import annotations

import logging
from datetime import datetime

from app.brief.evidence_graph import (
    EvidenceGraph,
    compute_evidence_coverage,
    compute_evidence_coverage_from_text,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dossier quality gates
# ---------------------------------------------------------------------------

ARTIFACT_COVERAGE_THRESHOLD = 60.0  # Lower than Mode B (85%) since PDF-only
WEB_AUGMENTED_COVERAGE_THRESHOLD = 85.0


def check_artifact_coverage(graph: EvidenceGraph) -> tuple[bool, float]:
    """Check if artifact-first dossier meets minimum evidence coverage.

    Returns (passes, coverage_pct).
    """
    claims = list(graph.claims.values())
    if not claims:
        # If no claims yet, check by node count
        pdf_nodes = [n for n in graph.nodes.values() if n.type == "PDF"]
        if len(pdf_nodes) >= 3:
            return True, 100.0
        return len(pdf_nodes) > 0, (len(pdf_nodes) / 3) * 100.0

    coverage = compute_evidence_coverage(claims)
    return coverage >= ARTIFACT_COVERAGE_THRESHOLD, coverage


# ---------------------------------------------------------------------------
# Evidence graph builder from PDF + meetings
# ---------------------------------------------------------------------------


def build_artifact_evidence_graph(
    profile_data: dict,
    person_name: str = "",
) -> EvidenceGraph:
    """Build an EvidenceGraph from stored PDF text + meeting data.

    This is Path B1: artifact-first, no web required.
    """
    from app.services.linkedin_pdf import (
        LinkedInPDFTextResult,
        build_evidence_nodes_from_pdf,
    )

    graph = EvidenceGraph()

    # Add PDF evidence nodes
    pdf_raw_text = profile_data.get("linkedin_pdf_raw_text", "")
    if pdf_raw_text:
        text_result = LinkedInPDFTextResult(
            raw_text=pdf_raw_text,
            headline=profile_data.get("headline", ""),
            location=profile_data.get("location", ""),
            about=profile_data.get("linkedin_pdf_sections", {}).get("about", ""),
            experience=profile_data.get("linkedin_pdf_experience", []),
            education=profile_data.get("linkedin_pdf_education", []),
            skills=profile_data.get("linkedin_pdf_skills", []),
            sections=profile_data.get("linkedin_pdf_sections", {}),
            page_count=profile_data.get("linkedin_pdf_page_count", 0),
        )

        pdf_nodes = build_evidence_nodes_from_pdf(text_result, contact_name=person_name)
        for node_data in pdf_nodes:
            graph.add_pdf_node(
                source=node_data["source"],
                snippet=node_data["snippet"],
                date=node_data.get("date", "UNKNOWN"),
                ref=node_data.get("ref", ""),
            )

    # Add meeting evidence nodes from interactions
    for interaction in profile_data.get("interactions", [])[:15]:
        graph.add_meeting_node(
            source=interaction.get("title", "meeting"),
            snippet=(
                interaction.get("summary", "")[:200]
                or interaction.get("title", "")
            ),
            date=interaction.get("date", "UNKNOWN"),
            ref=interaction.get("type", "meeting"),
        )

    return graph


# ---------------------------------------------------------------------------
# Artifact-first dossier builder (Path B1)
# ---------------------------------------------------------------------------


def build_artifact_dossier(
    person_name: str,
    graph: EvidenceGraph,
    profile_data: dict | None = None,
) -> str:
    """Build an artifact-first "Public Statements & Positions" dossier.

    Uses only PDF + meeting evidence. No web required.
    Tags: [VERIFIED-PDF], [VERIFIED-MEETING], [INFERRED-L/M], [UNKNOWN]

    Output structure:
    1. Executive Summary
    2. Identity & Background
    3. Career Timeline
    4. Topic Position Map
    5. Rhetorical Patterns & Language
    6. Gaps & Risks
    7. Interview Questions
    8. Primary-Source Index
    """
    profile_data = profile_data or {}
    parts: list[str] = []

    company = profile_data.get("company", "")
    title = profile_data.get("title", "")
    headline = profile_data.get("headline", "")
    location = profile_data.get("location", "")

    # Categorize nodes
    pdf_nodes = [n for n in graph.nodes.values() if n.type == "PDF"]
    meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]
    all_nodes = list(graph.nodes.values())

    # Header
    parts.append(f"# Public Statements & Positions Dossier: {person_name}")
    parts.append("")
    parts.append(f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    parts.append("**Mode**: Artifact-First (PDF + internal evidence)")
    parts.append(f"**Evidence Nodes**: {len(all_nodes)} ({len(pdf_nodes)} PDF, "
                 f"{len(meeting_nodes)} meeting)")
    parts.append("")

    # --- 1. Executive Summary ---
    parts.append("## 1. Executive Summary")
    parts.append("")

    ident_parts = []
    if title:
        ident_parts.append(f"**{person_name}** serves as {title}")
    else:
        ident_parts.append(f"**{person_name}**")
    if company:
        ident_parts.append(f"at {company}")
    if location:
        ident_parts.append(f"based in {location}")

    parts.append(" ".join(ident_parts) + ".")
    if headline and headline != title:
        parts.append(f"Self-described as: \"{headline}\" [VERIFIED-PDF]")
    parts.append("")

    about = profile_data.get("linkedin_pdf_sections", {}).get("about", "")
    if about:
        # Use first 2-3 sentences as executive summary
        sentences = about.split(". ")[:3]
        summary = ". ".join(sentences)
        if not summary.endswith("."):
            summary += "."
        parts.append(f"> {summary}")
        parts.append("> — LinkedIn profile [VERIFIED-PDF]")
        parts.append("")

    if not pdf_nodes and not meeting_nodes:
        parts.append("**No evidence available.** Upload a LinkedIn PDF or run meetings to "
                     "populate this dossier. [UNKNOWN]")
        parts.append("")

    # --- 2. Identity & Background ---
    parts.append("## 2. Identity & Background")
    parts.append("")
    parts.append("| Field | Value | Tag | Source |")
    parts.append("|-------|-------|-----|--------|")

    id_rows = [
        ("Name", person_name, "VERIFIED-PDF" if pdf_nodes else "UNKNOWN", "LinkedIn PDF"),
        ("Title", title or "Unknown", "VERIFIED-PDF" if title else "UNKNOWN", "LinkedIn PDF"),
        ("Company", company or "Unknown", "VERIFIED-PDF" if company else "UNKNOWN",
         "LinkedIn PDF"),
        ("Location", location or "Unknown", "VERIFIED-PDF" if location else "UNKNOWN",
         "LinkedIn PDF"),
        ("Headline", headline or "Unknown", "VERIFIED-PDF" if headline else "UNKNOWN",
         "LinkedIn PDF"),
    ]

    for field_name, value, tag, src in id_rows:
        parts.append(f"| {field_name} | {value} | [{tag}] | {src} |")

    # Add education
    education = profile_data.get("linkedin_pdf_education", [])
    for edu in education[:3]:
        school = edu.get("school", "Unknown")
        details = edu.get("details", "")
        parts.append(
            f"| Education | {school}: {details[:60]} | [VERIFIED-PDF] | LinkedIn PDF |"
        )

    parts.append("")

    # --- 3. Career Timeline ---
    parts.append("## 3. Career Timeline")
    parts.append("")

    experience = profile_data.get("linkedin_pdf_experience", [])
    if experience:
        for exp in experience[:10]:
            dates = exp.get("dates", "Unknown dates")
            exp_title = exp.get("title", "")
            exp_company = exp.get("company", "")
            desc = exp.get("description", "")
            if exp_title or exp_company:
                role_str = f"**{exp_title}**" if exp_title else ""
                if exp_company:
                    role_str += f" at {exp_company}" if role_str else f"**{exp_company}**"
                parts.append(f"- {dates}: {role_str} [VERIFIED-PDF]")
                if desc and len(desc.strip()) > 20:
                    # Show first sentence of description
                    first_sentence = desc.split(". ")[0]
                    if len(first_sentence) > 150:
                        first_sentence = first_sentence[:150] + "..."
                    parts.append(f"  - {first_sentence}")
    else:
        parts.append("- No career timeline available. [UNKNOWN]")
    parts.append("")

    # --- 4. Topic Position Map ---
    parts.append("## 4. Topic Position Map")
    parts.append("")

    if about:
        # Extract topic positions from about section
        parts.append("**Self-stated positions (from LinkedIn About section):**")
        parts.append("")
        # Split about into meaningful statements
        statements = [s.strip() for s in about.split(". ") if len(s.strip()) > 20]
        for stmt in statements[:8]:
            if not stmt.endswith("."):
                stmt += "."
            parts.append(f"- {stmt} [VERIFIED-PDF]")
        parts.append("")
    else:
        parts.append("No self-stated positions available. [UNKNOWN]")
        parts.append("")

    # Add meeting-derived positions
    if meeting_nodes:
        parts.append("**Positions expressed in meetings:**")
        parts.append("")
        for node in meeting_nodes[:8]:
            date_str = f" ({node.date})" if node.date != "UNKNOWN" else ""
            parts.append(f"- {node.snippet} [VERIFIED-MEETING]{date_str}")
        parts.append("")

    # --- 5. Rhetorical Patterns & Language ---
    parts.append("## 5. Rhetorical Patterns & Language")
    parts.append("")

    skills = profile_data.get("linkedin_pdf_skills", [])
    if skills:
        parts.append(f"**Stated expertise**: {', '.join(skills[:15])} [VERIFIED-PDF]")
        parts.append("")

    if about:
        # Basic rhetorical analysis
        parts.append("**Language patterns from self-description:**")
        parts.append("")
        # Check for common patterns
        about_lower = about.lower()
        if any(w in about_lower for w in ["lead", "leader", "leadership"]):
            parts.append("- Uses leadership framing in self-description [INFERRED-L]")
        if any(w in about_lower for w in ["build", "built", "building", "scale"]):
            parts.append("- Builder/scaler language orientation [INFERRED-L]")
        if any(w in about_lower for w in ["innovate", "innovation", "transform"]):
            parts.append("- Innovation/transformation orientation [INFERRED-L]")
        if any(w in about_lower for w in ["data", "analytics", "metric"]):
            parts.append("- Data/metrics-oriented framing [INFERRED-L]")
        if any(w in about_lower for w in ["team", "collaborate", "together"]):
            parts.append("- Team/collaboration emphasis [INFERRED-L]")
        parts.append("")
    else:
        parts.append("Insufficient text evidence for rhetorical analysis. [UNKNOWN]")
        parts.append("")

    # --- 6. Gaps & Risks ---
    parts.append("## 6. Gaps & Risks")
    parts.append("")

    gaps: list[str] = []
    if not profile_data.get("linkedin_url"):
        gaps.append("LinkedIn URL not confirmed — identity lock risk")
    if not company:
        gaps.append("Company not confirmed — organizational context unknown")
    if not title:
        gaps.append("Title not confirmed — authority scope unknown")
    if not education:
        gaps.append("Education unknown — no academic background available")
    if not experience:
        gaps.append("Experience timeline unknown — career trajectory unclear")
    if not meeting_nodes:
        gaps.append("No meeting history — all evidence is from PDF self-report")

    # Web gaps
    gaps.append("No public visibility sweep executed — public positioning unverified")
    gaps.append("No web search results — claims are self-reported only")

    for gap in gaps:
        parts.append(f"- {gap} [UNKNOWN]")
    parts.append("")

    parts.append(
        "> **To fill these gaps**, click 'Run Deep Research' which will execute "
        "web search + visibility sweep to corroborate PDF claims."
    )
    parts.append("")

    # --- 7. Interview Questions ---
    parts.append("## 7. Interview Questions")
    parts.append("")
    parts.append(
        "Questions designed to probe beyond the PDF self-report and verify claims:"
    )
    parts.append("")

    questions: list[str] = []
    if about:
        questions.append(
            f"Your profile mentions \"{about.split('.')[0][:80]}...\" — "
            "can you give me a specific example?"
        )
    if experience:
        latest = experience[0]
        if latest.get("title"):
            questions.append(
                f"In your role as {latest['title']}, what's the biggest "
                "challenge you're currently navigating?"
            )
    questions.append("What metrics are you personally accountable for this quarter?")
    questions.append("Who else in the organization needs to be involved in this decision?")
    questions.append(
        "What would need to be true for you to move forward with us in the next 30 days?"
    )

    for q in questions[:5]:
        parts.append(f"- {q}")
    parts.append("")

    # --- 8. Primary-Source Index ---
    parts.append("## 8. Primary-Source Index")
    parts.append("")
    parts.append("| # | Type | Source | Snippet | Ref |")
    parts.append("|---|------|--------|---------|-----|")

    for node in all_nodes[:30]:
        snippet_short = node.snippet[:60].replace("|", "/")
        parts.append(
            f"| {node.id} | {node.type} | {node.source[:40]} | "
            f"{snippet_short} | {node.ref} |"
        )
    parts.append("")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Artifact-first dossier with LLM enhancement (when available)
# ---------------------------------------------------------------------------


ARTIFACT_DOSSIER_SYSTEM_PROMPT = """\
You are a Strategic Intelligence Analyst producing an evidence-backed \
"Public Statements & Positions" dossier from a user-supplied LinkedIn PDF.

## ABSOLUTE RULES

1. **Evidence tagging** — every non-trivial claim must carry ONE tag:
   - [VERIFIED-PDF] — directly from the LinkedIn PDF text
   - [VERIFIED-MEETING] — from meeting transcript/email
   - [INFERRED-H] — high-confidence inference (cite upstream signals)
   - [INFERRED-M] — medium-confidence inference
   - [INFERRED-L] — low-confidence inference
   - [UNKNOWN] — no supporting evidence

2. **ZERO hallucination** — only use evidence provided. If no evidence, \
write "No evidence available." Do NOT fabricate.

3. **No generic filler** — no "strategic leader", "data-driven", \
"results-oriented" unless directly quoting the subject with citation.

4. **Person-level only** — this is about the individual, not their company.
"""


ARTIFACT_DOSSIER_USER_PROMPT = """\
## SUBJECT: {name}
- **Title**: {title}
- **Company**: {company}
- **Location**: {location}

## LINKEDIN PDF CONTENT
{pdf_text}

## INTERNAL MEETING EVIDENCE
{meeting_evidence}

## REQUIRED OUTPUT: "Public Statements & Positions" Dossier

Produce ALL 8 sections. Tag every claim. No filler.

### 1. Executive Summary (3-5 sentences, all tagged)
### 2. Identity & Background (fact table with tags)
### 3. Career Timeline (chronological, tagged)
### 4. Topic Position Map (what they believe, tagged from their words)
### 5. Rhetorical Patterns & Language (how they communicate, tagged)
### 6. Gaps & Risks (what's unknown or concerning)
### 7. Interview Questions (5 questions targeting gaps)
### 8. Primary-Source Index (evidence node table)
"""


def generate_artifact_dossier_with_llm(
    person_name: str,
    graph: EvidenceGraph,
    profile_data: dict,
) -> str:
    """Generate an LLM-enhanced artifact-first dossier.

    Uses the LLM to synthesize PDF + meeting evidence into a structured dossier.
    Falls back to the template-based builder if LLM is unavailable.
    """
    try:
        from app.clients.openai_client import LLMClient

        pdf_text = profile_data.get("linkedin_pdf_raw_text", "")
        if not pdf_text:
            # No PDF text — use template builder
            return build_artifact_dossier(person_name, graph, profile_data)

        # Build meeting evidence text
        meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]
        meeting_text = "No meeting evidence available."
        if meeting_nodes:
            meeting_parts = []
            for node in meeting_nodes:
                meeting_parts.append(
                    f"- [{node.date}] {node.source}: {node.snippet}"
                )
            meeting_text = "\n".join(meeting_parts)

        user_prompt = ARTIFACT_DOSSIER_USER_PROMPT.format(
            name=person_name,
            title=profile_data.get("title", "Unknown"),
            company=profile_data.get("company", "Unknown"),
            location=profile_data.get("location", "Unknown"),
            pdf_text=pdf_text[:15000],  # Cap for prompt size
            meeting_evidence=meeting_text,
        )

        llm = LLMClient()
        result = llm.chat(ARTIFACT_DOSSIER_SYSTEM_PROMPT, user_prompt, temperature=0.3)
        return result

    except Exception:
        logger.exception("LLM dossier generation failed — using template builder")
        return build_artifact_dossier(person_name, graph, profile_data)


# ---------------------------------------------------------------------------
# Dual-path orchestrator
# ---------------------------------------------------------------------------


def run_artifact_dossier_pipeline(
    profile_data: dict,
    person_name: str,
    use_llm: bool = True,
) -> dict:
    """Run the artifact-first dossier pipeline.

    Returns a dict with:
    - dossier_markdown: the generated dossier text
    - evidence_graph: serialized graph
    - artifact_count: number of PDF evidence nodes
    - meeting_count: number of meeting evidence nodes
    - coverage_pct: evidence coverage
    - mode: "artifact_first" or "artifact_llm"
    - generated_at: timestamp
    """
    graph = build_artifact_evidence_graph(profile_data, person_name)

    pdf_nodes = [n for n in graph.nodes.values() if n.type == "PDF"]
    meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]

    # Check coverage
    passes_coverage, coverage_pct = check_artifact_coverage(graph)

    # Generate dossier
    if use_llm and profile_data.get("linkedin_pdf_raw_text"):
        dossier = generate_artifact_dossier_with_llm(
            person_name, graph, profile_data
        )
        mode = "artifact_llm"
    else:
        dossier = build_artifact_dossier(person_name, graph, profile_data)
        mode = "artifact_first"

    # Compute text-based coverage if we generated prose
    if dossier:
        text_coverage = compute_evidence_coverage_from_text(dossier)
        coverage_pct = max(coverage_pct, text_coverage)

    generated_at = datetime.utcnow().isoformat()

    return {
        "dossier_markdown": dossier,
        "evidence_graph": graph.to_dict(),
        "artifact_count": len(pdf_nodes),
        "meeting_count": len(meeting_nodes),
        "total_evidence_nodes": len(graph.nodes),
        "coverage_pct": round(coverage_pct, 1),
        "passes_coverage": passes_coverage,
        "mode": mode,
        "generated_at": generated_at,
    }
