"""Fail-closed Evidence Graph engine for Contact Intelligence Dossiers.

This module implements:
1. Evidence Graph construction from retrieval results + meeting data
2. Retrieval Ledger with mandatory logging of every SerpAPI call
3. Fail-closed gate checks that HALT output when gates fail (Mode B only)
4. Evidence coverage computation per the formal definition
5. Two-mode architecture:
   - Mode A (Meeting-Prep Brief): fast, internal-only, no web required
   - Mode B (Deep Research Dossier): web-required, fail-closed

Mode A (Meeting-Prep Brief):
- NO SerpAPI required, NO visibility sweep, NO fail-closed gating
- Tags: [VERIFIED-MEETING], [INFERRED-L/M], [UNKNOWN]
- Sections: What we know, What to do next, Key risks, Missing intel

Mode B (Deep Research Dossier):
- VISIBILITY SWEEP NOT EXECUTED → halt
- EVIDENCE COVERAGE < 85% → halt
- ENTITY LOCK < 70 → constrain (no strong person-level claims)
- INTERNAL CONTRADICTION → halt
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from app.models import Claim, EvidenceNode, RetrievalLedgerRow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evidence Graph
# ---------------------------------------------------------------------------


class EvidenceGraph:
    """Container for EvidenceNodes, Claims, and RetrievalLedger rows.

    The graph is the single source of truth for what we know and how we know it.
    """

    def __init__(self) -> None:
        self.nodes: dict[str, EvidenceNode] = {}
        self.claims: dict[str, Claim] = {}
        self.ledger: list[RetrievalLedgerRow] = []
        self._next_node_id: int = 1
        self._next_claim_id: int = 1
        self._next_query_id: int = 1

    # --- Node management ---

    def add_node(
        self,
        type: str,
        source: str,
        snippet: str,
        ref: str = "",
        date: str = "UNKNOWN",
    ) -> EvidenceNode:
        """Add an EvidenceNode and return it."""
        node_id = f"E{self._next_node_id}"
        self._next_node_id += 1
        node = EvidenceNode(
            id=node_id,
            type=type,
            source=source,
            snippet=snippet[:200],  # hard limit
            ref=ref,
            date=date,
        )
        self.nodes[node_id] = node
        return node

    def add_meeting_node(
        self,
        source: str,
        snippet: str,
        date: str = "UNKNOWN",
        ref: str = "",
    ) -> EvidenceNode:
        """Add a MEETING-type EvidenceNode."""
        return self.add_node(type="MEETING", source=source, snippet=snippet, ref=ref, date=date)

    def add_public_node(
        self,
        source: str,
        snippet: str,
        date: str = "UNKNOWN",
        ref: str = "",
    ) -> EvidenceNode:
        """Add a PUBLIC-type EvidenceNode."""
        return self.add_node(type="PUBLIC", source=source, snippet=snippet, ref=ref, date=date)

    def add_pdf_node(
        self,
        source: str,
        snippet: str,
        date: str = "UNKNOWN",
        ref: str = "",
    ) -> EvidenceNode:
        """Add a PDF-type EvidenceNode (user-supplied LinkedIn PDF)."""
        return self.add_node(type="PDF", source=source, snippet=snippet, ref=ref, date=date)

    # --- Claim management ---

    def add_claim(
        self,
        text: str,
        tag: str,
        evidence_ids: list[str] | None = None,
        confidence: str = "L",
    ) -> Claim:
        """Register a Claim with evidence linkage."""
        claim_id = f"C{self._next_claim_id}"
        self._next_claim_id += 1
        claim = Claim(
            claim_id=claim_id,
            text=text,
            tag=tag,
            evidence_ids=evidence_ids or [],
            confidence=confidence,
        )
        self.claims[claim_id] = claim
        return claim

    # --- Retrieval Ledger ---

    def log_retrieval(
        self,
        query: str,
        intent: str,
        results: list[dict[str, Any]] | None = None,
        selected_evidence_ids: list[str] | None = None,
    ) -> RetrievalLedgerRow:
        """Log a retrieval query to the ledger. Always logs, even with 0 results."""
        query_id = f"Q{self._next_query_id}"
        self._next_query_id += 1

        top_results = []
        if results:
            for i, r in enumerate(results[:5], 1):
                top_results.append({
                    "rank": i,
                    "title": r.get("title", ""),
                    "url": r.get("link", r.get("url", "")),
                    "date": r.get("date", "UNKNOWN"),
                    "snippet": (r.get("snippet", ""))[:200],
                })

        row = RetrievalLedgerRow(
            query_id=query_id,
            query=query,
            intent=intent,
            top_results=top_results,
            selected_evidence_ids=selected_evidence_ids or [],
            result_count=len(results) if results else 0,
        )
        self.ledger.append(row)
        return row

    # --- Queries ---

    def get_visibility_ledger_rows(self) -> list[RetrievalLedgerRow]:
        """Return only visibility-intent ledger rows."""
        return [r for r in self.ledger if r.intent == "visibility"]

    def get_node(self, node_id: str) -> EvidenceNode | None:
        return self.nodes.get(node_id)

    def get_claim(self, claim_id: str) -> Claim | None:
        return self.claims.get(claim_id)

    # --- Serialization ---

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full graph for API responses / persistence."""
        return {
            "nodes": [n.model_dump() for n in self.nodes.values()],
            "claims": [c.model_dump() for c in self.claims.values()],
            "ledger": [r.model_dump() for r in self.ledger],
        }


# ---------------------------------------------------------------------------
# Evidence Coverage Computation
# ---------------------------------------------------------------------------

# Tags that count as "evidenced" (not UNKNOWN)
_VALID_EVIDENCE_TAGS = {
    "VERIFIED-MEETING", "VERIFIED-PUBLIC", "VERIFIED-PDF",
    "VERIFIED_MEETING", "VERIFIED_PUBLIC", "VERIFIED_PDF",
    "INFERRED-H", "INFERRED-M", "INFERRED-L",
    "INFERRED_HIGH", "INFERRED_MEDIUM", "INFERRED_LOW",
}


def compute_evidence_coverage(claims: list[Claim]) -> float:
    """Compute evidence coverage percentage.

    Coverage = (# substantive claims with >=1 evidence_id or tag != UNKNOWN) /
               (total claims)

    Returns 0.0 if no claims.
    """
    if not claims:
        return 0.0

    covered = 0
    for claim in claims:
        if claim.tag == "UNKNOWN" and not claim.evidence_ids:
            continue
        if claim.evidence_ids or claim.tag in _VALID_EVIDENCE_TAGS:
            covered += 1

    return (covered / len(claims)) * 100.0


def compute_evidence_coverage_from_text(text: str) -> float:
    """Compute coverage from raw dossier text by counting evidence tags.

    A substantive line is >20 chars, not a header, not a table delimiter.
    A covered line has an evidence tag pattern like [VERIFIED-*] or [INFERRED-*].
    """
    tag_pattern = re.compile(
        r"\[(?:VERIFIED[–-](?:MEETING|PUBLIC|PDF)|INFERRED[–-][HML]|UNKNOWN)\]",
        re.IGNORECASE,
    )
    lines = text.strip().split("\n")
    total = 0
    tagged = 0
    for line in lines:
        stripped = line.strip()
        if len(stripped) <= 20:
            continue
        if stripped.startswith("#") or stripped.startswith("---") or stripped.startswith("|"):
            continue
        total += 1
        if tag_pattern.search(stripped):
            tagged += 1

    if total == 0:
        return 100.0
    return (tagged / total) * 100.0


# ---------------------------------------------------------------------------
# Fail-Closed Gate Engine
# ---------------------------------------------------------------------------

EVIDENCE_COVERAGE_THRESHOLD = 85.0
ENTITY_LOCK_THRESHOLD = 70


@dataclass
class GateResult:
    """Result of a single fail-closed gate check."""
    gate_name: str
    passed: bool
    details: str = ""
    remediation: str = ""


@dataclass
class FailClosedReport:
    """Aggregate result of all fail-closed gates."""
    gates: list[GateResult] = field(default_factory=list)
    all_passed: bool = False
    is_constrained: bool = False  # True when entity lock < 70
    failure_output: str = ""  # The text to return when gates fail

    @property
    def should_halt(self) -> bool:
        """True if any hard gate failed (not just constrained)."""
        return not self.all_passed and not self.is_constrained


def check_visibility_sweep_gate(graph: EvidenceGraph) -> GateResult:
    """Gate 1: Visibility sweep must have been executed.

    Checks that the retrieval ledger contains visibility-intent rows.
    """
    visibility_rows = graph.get_visibility_ledger_rows()
    if not visibility_rows:
        queries = _get_required_visibility_queries("<full name>")
        query_text = "\n".join(f"  - {q}" for q in queries)
        return GateResult(
            gate_name="VISIBILITY_SWEEP",
            passed=False,
            details="No visibility-intent rows found in retrieval ledger.",
            remediation=(
                "FAIL: VISIBILITY SWEEP NOT EXECUTED\n"
                "Run the following queries and log results to the retrieval ledger:\n"
                f"{query_text}"
            ),
        )
    return GateResult(
        gate_name="VISIBILITY_SWEEP",
        passed=True,
        details=f"{len(visibility_rows)} visibility queries executed.",
    )


def check_evidence_coverage_gate(
    claims: list[Claim],
    dossier_text: str = "",
) -> GateResult:
    """Gate 2: Evidence coverage must be >= 85%.

    Uses claim-level coverage if claims exist, else falls back to text-level.
    """
    if claims:
        coverage = compute_evidence_coverage(claims)
    elif dossier_text:
        coverage = compute_evidence_coverage_from_text(dossier_text)
    else:
        coverage = 0.0

    if coverage < EVIDENCE_COVERAGE_THRESHOLD:
        return GateResult(
            gate_name="EVIDENCE_COVERAGE",
            passed=False,
            details=f"Coverage {coverage:.1f}% < {EVIDENCE_COVERAGE_THRESHOLD}% threshold.",
            remediation=(
                f"FAIL: EVIDENCE COVERAGE {coverage:.1f}%\n"
                "Uncited claims must be tagged with evidence IDs or removed.\n"
                "Run additional retrieval queries to gather missing evidence."
            ),
        )
    return GateResult(
        gate_name="EVIDENCE_COVERAGE",
        passed=True,
        details=f"Coverage {coverage:.1f}% >= {EVIDENCE_COVERAGE_THRESHOLD}%.",
    )


def check_entity_lock_gate(entity_lock_score: int) -> GateResult:
    """Gate 3: Entity lock score check.

    >= 70: LOCKED (full dossier)
    50-69: PARTIAL (constrained — only VERIFIED + UNKNOWN + INFERRED-L)
    < 50: NOT LOCKED (constrained — same restrictions)
    """
    if entity_lock_score >= ENTITY_LOCK_THRESHOLD:
        return GateResult(
            gate_name="ENTITY_LOCK",
            passed=True,
            details=f"Score {entity_lock_score}/100 — LOCKED.",
        )
    label = "PARTIAL" if entity_lock_score >= 50 else "NOT LOCKED"
    return GateResult(
        gate_name="ENTITY_LOCK",
        passed=False,
        details=(
            f"Score {entity_lock_score}/100 — IDENTITY {label}.\n"
            "Dossier will NOT include strong person-level claims.\n"
            "Only VERIFIED facts, UNKNOWNs, and safe INFERRED-L permitted."
        ),
        remediation=(
            f"IDENTITY {label}: score {entity_lock_score}/100.\n"
            "Fetch additional identity signals:\n"
            "  - Confirm LinkedIn URL\n"
            "  - Cross-reference employer in public sources\n"
            "  - Verify title on company website"
        ),
    )


def run_fail_closed_gates(
    graph: EvidenceGraph,
    entity_lock_score: int,
    dossier_text: str = "",
) -> FailClosedReport:
    """Run all fail-closed gates. Returns a report.

    Gate execution order:
    1. Visibility sweep (hard fail)
    2. Evidence coverage (hard fail)
    3. Entity lock (constrain, not halt)

    If any hard gate fails, the report includes the failure output
    that must be returned INSTEAD of a dossier.
    """
    report = FailClosedReport()

    # Gate 1: Visibility sweep
    vis_gate = check_visibility_sweep_gate(graph)
    report.gates.append(vis_gate)

    # Gate 2: Evidence coverage
    claims_list = list(graph.claims.values())
    cov_gate = check_evidence_coverage_gate(claims_list, dossier_text)
    report.gates.append(cov_gate)

    # Gate 3: Entity lock
    lock_gate = check_entity_lock_gate(entity_lock_score)
    report.gates.append(lock_gate)

    # Determine overall status
    hard_failures = [
        g for g in report.gates
        if not g.passed and g.gate_name in ("VISIBILITY_SWEEP", "EVIDENCE_COVERAGE")
    ]

    if hard_failures:
        report.all_passed = False
        report.is_constrained = False
        # Build failure output
        parts = ["DOSSIER GENERATION HALTED — FAIL-CLOSED GATES FAILED\n"]
        for gate in hard_failures:
            parts.append(f"--- {gate.gate_name} ---")
            parts.append(gate.remediation)
            parts.append("")
        # Include entity lock status for context
        parts.append("--- ENTITY_LOCK ---")
        parts.append(lock_gate.details)
        report.failure_output = "\n".join(parts)
    elif not lock_gate.passed:
        report.all_passed = False
        report.is_constrained = True
    else:
        report.all_passed = True
        report.is_constrained = False

    return report


# ---------------------------------------------------------------------------
# Visibility Sweep Query Battery
# ---------------------------------------------------------------------------

# Full 16-query visibility sweep per the spec
VISIBILITY_QUERY_TEMPLATES: list[tuple[str, str]] = [
    # A) TED/TEDx (explicit, 4 queries)
    ('"{name}" TED', "visibility"),
    ('"{name}" TEDx', "visibility"),
    ('site:ted.com "{name}"', "visibility"),
    ('site:youtube.com "{name}" TEDx', "visibility"),
    # B) Keynotes / Conferences (4 queries)
    ('"{name}" keynote', "visibility"),
    ('"{name}" conference talk', "visibility"),
    ('"{name}" summit speaker', "visibility"),
    ('"{name}" panel discussion', "visibility"),
    # C) Podcasts / Webinars / Interviews (4 queries)
    ('"{name}" podcast', "visibility"),
    ('"{name}" webinar', "visibility"),
    ('"{name}" interview video', "visibility"),
    ('"{name}" fireside chat', "visibility"),
    # D) YouTube / Vimeo / Slide decks (3 queries)
    ('"{name}" YouTube talk', "visibility"),
    ('"{name}" Vimeo talk', "visibility"),
    ('"{name}" SlideShare', "visibility"),
]

# Category groupings for audit
VISIBILITY_CATEGORY_GROUPS = {
    "ted_tedx": [0, 1, 2, 3],      # First 4 queries
    "keynote_conference": [4, 5, 6, 7],  # Next 4
    "podcast_webinar": [8, 9, 10, 11],   # Next 4
    "youtube_video": [12, 13, 14],        # Last 3
}


def _get_required_visibility_queries(name: str) -> list[str]:
    """Return the full list of required visibility queries for a person."""
    return [tpl[0].replace("{name}", name) for tpl in VISIBILITY_QUERY_TEMPLATES]


def build_visibility_queries(name: str, company: str = "") -> list[tuple[str, str]]:
    """Build the full visibility sweep query battery.

    Returns list of (query_string, intent) tuples.
    """
    queries = []
    for template, intent in VISIBILITY_QUERY_TEMPLATES:
        query = template.replace("{name}", name)
        queries.append((query, intent))

    # Add company-qualified variant for top queries if company is provided
    if company:
        queries.append((f'"{name}" "{company}" keynote OR conference OR podcast', "visibility"))

    return queries


def extract_highest_signal_artifacts(
    graph: EvidenceGraph,
    max_artifacts: int = 3,
) -> list[dict[str, str]]:
    """Extract the top 1-3 highest-signal visibility artifacts from the graph.

    Prioritizes: TED > Keynote > Conference > Podcast > Other
    Returns list of {title, venue, date, url, why_it_matters}.
    """
    visibility_rows = graph.get_visibility_ledger_rows()
    all_results: list[dict[str, Any]] = []

    # Priority keywords (higher index = higher priority)
    priority_keywords = [
        "slideshare", "vimeo", "webinar", "fireside",
        "interview", "panel", "podcast",
        "conference", "summit", "keynote",
        "tedx", "ted",
    ]

    for row in visibility_rows:
        for result in row.top_results:
            title = result.get("title", "").lower()
            url = result.get("url", "")
            priority = 0
            for i, kw in enumerate(priority_keywords):
                if kw in title or kw in row.query.lower():
                    priority = max(priority, i)

            all_results.append({
                "title": result.get("title", ""),
                "venue": _infer_venue(result.get("title", ""), url),
                "date": result.get("date", "UNKNOWN"),
                "url": url,
                "query": row.query,
                "priority": priority,
                "why_it_matters": _infer_signal_value(priority),
            })

    # Deduplicate by URL
    seen_urls: set[str] = set()
    unique: list[dict] = []
    for r in all_results:
        if r["url"] and r["url"] not in seen_urls:
            seen_urls.add(r["url"])
            unique.append(r)

    # Sort by priority descending
    unique.sort(key=lambda x: x["priority"], reverse=True)

    return [
        {
            "title": r["title"],
            "venue": r["venue"],
            "date": r["date"],
            "url": r["url"],
            "why_it_matters": r["why_it_matters"],
        }
        for r in unique[:max_artifacts]
    ]


def compute_visibility_coverage_confidence(graph: EvidenceGraph) -> int:
    """Compute 0-100 confidence score for visibility sweep coverage.

    Per spec:
    - +10 per query family with >=1 relevant result
    - +10 if TED/TEDx queries were explicitly executed (even with 0 results)
    - Cap at 100

    Query family mapping:
    0-1: ted/tedx, 2-3: ted/tedx (site-specific), 4: keynote, 5: conference,
    6: summit, 7: panel, 8: podcast, 9: webinar, 10-11: interview,
    12-14: youtube/video
    """
    visibility_rows = graph.get_visibility_ledger_rows()
    if not visibility_rows:
        return 0

    # Map query indices to category families
    _QUERY_TO_FAMILY: dict[int, str] = {
        0: "ted", 1: "tedx", 2: "ted", 3: "tedx",
        4: "keynote", 5: "conference", 6: "summit", 7: "panel",
        8: "podcast", 9: "webinar", 10: "interview_video", 11: "interview_video",
        12: "youtube_talk", 13: "youtube_talk", 14: "youtube_talk",
    }

    families_with_results: set[str] = set()
    ted_tedx_executed = False

    for i, row in enumerate(visibility_rows):
        family = _QUERY_TO_FAMILY.get(i)
        if family in ("ted", "tedx"):
            ted_tedx_executed = True
        if row.result_count > 0 and family:
            families_with_results.add(family)

    score = len(families_with_results) * 10

    # Bonus +10 for TED/TEDx execution (even 0 results — the point is it was checked)
    if ted_tedx_executed:
        score += 10

    return min(100, score)


def _infer_venue(title: str, url: str) -> str:
    """Best-effort venue extraction from title and URL."""
    if "ted.com" in url:
        return "TED"
    if "tedx" in title.lower():
        return "TEDx"
    if "youtube.com" in url:
        return "YouTube"
    if "vimeo.com" in url:
        return "Vimeo"
    if "slideshare" in url.lower():
        return "SlideShare"
    return "Unknown Venue"


def _infer_signal_value(priority: int) -> str:
    """Map priority score to strategic signal description."""
    if priority >= 10:
        return "TED/TEDx — top-tier thought leadership visibility"
    if priority >= 8:
        return "Keynote/conference — industry authority signal"
    if priority >= 6:
        return "Podcast/panel — public positioning and messaging patterns"
    if priority >= 4:
        return "Interview/webinar — topical engagement signal"
    return "Presentation material — expertise claim"


# ---------------------------------------------------------------------------
# Dossier Output Mode (A / B / C)
# ---------------------------------------------------------------------------


class DossierMode:
    """Output mode for the intelligence system.

    Two product modes:
    MEETING_PREP:  Mode A — fast, always available, internal-only, no web required.
    DEEP_RESEARCH: Mode B — web-required, fail-closed, full dossier.

    Deep Research sub-states:
    FULL:        entity_lock >= 70 AND visibility executed — unrestricted output.
    CONSTRAINED: entity_lock 50-69 — restrict inference strengths.
    HALTED:      pre-synthesis gate failure — do NOT call LLM.
    """
    # Product modes
    MEETING_PREP = "meeting_prep"
    DEEP_RESEARCH = "deep_research"

    # Deep Research sub-states
    FULL = "full"
    CONSTRAINED = "constrained"
    HALTED = "halted"

    # Deep research job status
    NOT_STARTED = "NOT_STARTED"
    RUNNING = "RUNNING"
    FAILED = "FAILED"
    SUCCEEDED = "SUCCEEDED"


def determine_dossier_mode(
    entity_lock_score: int,
    visibility_executed: bool,
    has_public_results: bool,
    person_name: str = "",
) -> tuple[str, str]:
    """Determine Deep Research (Mode B) output sub-state BEFORE calling LLM synthesis.

    Returns (mode, reason).
    If mode == HALTED, the caller must NOT proceed to synthesis.
    This is used ONLY for Mode B (Deep Research). Mode A bypasses this entirely.
    """
    if not visibility_executed:
        queries = _get_required_visibility_queries(person_name or "<name>")
        query_text = "\n".join(f"  - {q}" for q in queries)
        return DossierMode.HALTED, (
            "FAIL: VISIBILITY SWEEP NOT EXECUTED\n"
            "The retrieval ledger contains 0 visibility-intent rows.\n"
            "Cannot generate dossier without executing the visibility sweep.\n\n"
            "Run these queries:\n" + query_text
        )

    if not has_public_results:
        return DossierMode.HALTED, (
            "FAIL: NO PUBLIC RETRIEVAL RESULTS\n"
            "Entity Lock cannot be computed without at least one public retrieval result.\n"
            f"SerpAPI returned 0 results for \"{person_name}\".\n"
            "Verify the person name and run retrieval again."
        )

    if entity_lock_score >= ENTITY_LOCK_THRESHOLD:
        return DossierMode.FULL, f"Entity LOCKED ({entity_lock_score}/100) — full dossier"

    if entity_lock_score >= 50:
        return DossierMode.CONSTRAINED, (
            f"PARTIAL DOSSIER — IDENTITY NOT LOCKED ({entity_lock_score}/100)\n"
            "Restricting output to VERIFIED + UNKNOWN + INFERRED-L claims.\n"
            "Strong person-level inferences suppressed."
        )

    return DossierMode.CONSTRAINED, (
        f"PARTIAL DOSSIER — IDENTITY NOT LOCKED ({entity_lock_score}/100)\n"
        "Restricting output to VERIFIED-MEETING facts and safe inferences only.\n"
        "Prioritize disambiguation retrieval before generating a full dossier."
    )


# ---------------------------------------------------------------------------
# Mode A: Meeting-Prep Brief Builder
# ---------------------------------------------------------------------------


def build_meeting_prep_brief(
    person_name: str,
    graph: EvidenceGraph,
    profile_data: dict | None = None,
) -> str:
    """Build a Mode A Meeting-Prep Brief from internal evidence only.

    No SerpAPI required. No visibility sweep. No fail-closed gating.
    Tags: [VERIFIED-MEETING], [INFERRED-L/M], [UNKNOWN]

    Sections:
    1. What we know from our interactions
    2. What to do next (3-5 targeted questions + prep checklist)
    3. Key risks / watchouts
    4. Missing intel worth fetching
    """
    profile_data = profile_data or {}
    parts: list[str] = []

    # Header
    company = profile_data.get("company", "")
    title = profile_data.get("title", "")
    parts.append(f"# Meeting-Prep Brief: {person_name}")
    parts.append("")
    if title or company:
        ident = title
        if company:
            ident = f"{title} @ {company}" if title else company
        parts.append(f"**Role**: {ident}")
    parts.append("**Mode**: Meeting-Prep (internal evidence only)")
    parts.append("")

    # Section 1: What we know from our interactions
    parts.append("## 1. What We Know From Our Interactions")
    parts.append("")

    meeting_nodes = [n for n in graph.nodes.values() if n.type == "MEETING"]
    if meeting_nodes:
        for node in meeting_nodes:
            date_str = f" ({node.date})" if node.date != "UNKNOWN" else ""
            parts.append(
                f"- {node.snippet} [VERIFIED-MEETING]{date_str}"
            )
    else:
        parts.append("- No meeting or email history available. [UNKNOWN]")
    parts.append("")

    # Action items from profile data
    action_items = profile_data.get("action_items", [])
    if action_items:
        parts.append("**Open Action Items:**")
        for item in action_items[:10]:
            parts.append(f"- {item} [VERIFIED-MEETING]")
        parts.append("")

    # Section 2: What to do next
    parts.append("## 2. What To Do Next")
    parts.append("")
    parts.append("**Targeted Questions:**")

    # Generate questions based on what we know / don't know
    questions: list[str] = []
    if not title:
        questions.append(
            "What is your current role and scope of responsibility? [UNKNOWN]"
        )
    if not profile_data.get("location"):
        questions.append("Where are you based? [UNKNOWN]")

    interactions = profile_data.get("interactions", [])
    if interactions:
        last = interactions[0] if interactions else {}
        last_summary = last.get("summary", "")
        if last_summary:
            questions.append(
                f"Following up on our last conversation: how has the situation "
                f"evolved since we discussed \"{last_summary[:80]}\"? [INFERRED-L]"
            )
    if not questions:
        questions.append(
            "What are your top priorities for the next quarter? [UNKNOWN]"
        )
    questions.append(
        "What would make this meeting most valuable for you? [UNKNOWN]"
    )
    questions.append(
        "Are there any constraints or blockers I should know about? [UNKNOWN]"
    )

    for q in questions[:5]:
        parts.append(f"- {q}")
    parts.append("")

    parts.append("**Prep Checklist:**")
    parts.append("- [ ] Review last interaction notes")
    parts.append("- [ ] Check for any pending action items")
    if not profile_data.get("linkedin_url"):
        parts.append("- [ ] Find and review LinkedIn profile")
    parts.append("- [ ] Prepare agenda with 2-3 key discussion points")
    if not profile_data.get("deep_profile"):
        parts.append("- [ ] Consider running Deep Research for full dossier")
    parts.append("")

    # Section 3: Key risks / watchouts
    parts.append("## 3. Key Risks / Watchouts")
    parts.append("")

    risks: list[str] = []
    if not meeting_nodes:
        risks.append(
            "No prior interaction history — first meeting risk. "
            "Prepare broader discovery questions. [UNKNOWN]"
        )
    elif len(meeting_nodes) == 1:
        risks.append(
            "Only one prior interaction — limited context. "
            "Avoid assumptions based on single data point. [INFERRED-L]"
        )

    if action_items:
        risks.append(
            f"{len(action_items)} open action item(s) — ensure none are overdue "
            f"before the meeting. [VERIFIED-MEETING]"
        )

    if not company:
        risks.append(
            "Company not confirmed — verify organization context "
            "before discussing specifics. [UNKNOWN]"
        )

    if not risks:
        risks.append(
            "No significant risks identified from available meeting data. [INFERRED-L]"
        )

    for risk in risks:
        parts.append(f"- {risk}")
    parts.append("")

    # Section 4: Missing intel worth fetching
    parts.append("## 4. Missing Intel Worth Fetching")
    parts.append("")
    parts.append(
        "The following information would significantly improve preparation. "
        "**Recommend running Deep Research** to gather:"
    )
    parts.append("")

    missing: list[str] = []
    if not profile_data.get("linkedin_url"):
        missing.append("LinkedIn profile — career history, connections, endorsements")
    if not title:
        missing.append("Current title and scope of authority")
    if not company:
        missing.append("Company details — size, industry, recent news")
    missing.append("Public speaking history (TED, conferences, podcasts)")
    missing.append("Recent press mentions or published content")
    missing.append("Organizational structure and reporting lines")

    for item in missing:
        parts.append(f"- {item}")
    parts.append("")
    parts.append(
        "> **To get this intelligence, click 'Run Deep Research' in the "
        "contact profile.**"
    )
    parts.append("")

    return "\n".join(parts)


def filter_prose_by_mode(dossier_text: str, mode: str, entity_lock_score: int) -> str:
    """Filter dossier prose based on output mode.

    Mode FULL: return as-is.
    Mode CONSTRAINED (50-69): strip INFERRED-H/M claims, prepend banner.
    Mode CONSTRAINED (<50): strip all INFERRED claims, prepend banner.
    Mode HALTED: should not reach here (caller should not have generated prose).
    """
    if mode == DossierMode.FULL:
        return dossier_text

    if mode == DossierMode.HALTED:
        return dossier_text  # Shouldn't happen, but don't crash

    lines = dossier_text.split("\n")
    filtered: list[str] = []

    # Pattern for tags to strip
    if entity_lock_score >= 50:
        # CONSTRAINED (PARTIAL): strip INFERRED-H and INFERRED-M only
        strip_pattern = re.compile(r"\[INFERRED[–\-][HM]\]", re.IGNORECASE)
    else:
        # CONSTRAINED (NOT LOCKED): strip ALL INFERRED
        strip_pattern = re.compile(r"\[INFERRED[–\-][HML]\]", re.IGNORECASE)

    for line in lines:
        stripped = line.strip()
        # Keep headers, separators, and non-substantive lines
        if not stripped or len(stripped) <= 20 or stripped.startswith(("#", "|", "---", "*", ">")):
            filtered.append(line)
            continue
        # Drop lines containing banned inference tags
        if strip_pattern.search(stripped):
            continue
        filtered.append(line)

    lock_label = "PARTIAL LOCK" if entity_lock_score >= 50 else "NOT LOCKED"
    banner = (
        f"> **PARTIAL DOSSIER — IDENTITY {lock_label} ({entity_lock_score}/100)**\n"
        "> Strong person-level inferences have been suppressed.\n"
        "> Only VERIFIED facts and low-confidence inferences are shown.\n"
        "---\n"
    )

    return banner + "\n".join(filtered)


def build_failure_report(
    mode_reason: str,
    entity_lock_score: int,
    visibility_confidence: int,
    graph: EvidenceGraph,
    person_name: str = "",
) -> str:
    """Build a Mode A failure report when pre-synthesis gates halt output.

    Includes: which stage failed, why, exact queries to run, and what evidence is needed.
    """
    lock_label = (
        "LOCKED" if entity_lock_score >= 70
        else "PARTIAL" if entity_lock_score >= 50
        else "NOT LOCKED"
    )

    parts = [
        "=" * 60,
        "DOSSIER GENERATION HALTED — FAIL-CLOSED GATES",
        "=" * 60,
        "",
        mode_reason,
        "",
        "--- CURRENT STATE ---",
        f"Entity Lock:           {entity_lock_score}/100 ({lock_label})",
        f"Visibility Confidence: {visibility_confidence}/100",
        f"Evidence Nodes:        {len(graph.nodes)}",
        f"Retrieval Ledger Rows: {len(graph.ledger)}",
        "",
    ]

    # Retrieval Ledger summary
    if graph.ledger:
        parts.append("--- RETRIEVAL LEDGER ---")
        for row in graph.ledger:
            parts.append(
                f"  {row.query_id}: [{row.intent}] {row.query} → "
                f"{row.result_count} result(s)"
            )
        parts.append("")

    # What's needed to proceed
    parts.append("--- WHAT TO DO NEXT ---")
    if not graph.get_visibility_ledger_rows():
        parts.append("1. Execute the full visibility sweep query battery:")
        for q in _get_required_visibility_queries(person_name or "<name>"):
            parts.append(f"   - {q}")
    else:
        parts.append("1. Visibility sweep is logged (OK)")

    total_public = sum(1 for row in graph.ledger if row.result_count > 0)
    if total_public == 0:
        parts.append("2. Get at least 1 public retrieval result to compute Entity Lock")
    elif entity_lock_score < 70:
        parts.append("2. Increase Entity Lock by confirming:")
        parts.append("   - LinkedIn URL present → +10pts (weak)")
        parts.append("   - LinkedIn verified via retrieval → +30pts (strong)")
        parts.append("   - Meeting confirms identity → +20pts")
        parts.append("   - Employer in public source → +20pts")
        parts.append("   - Multiple independent domains agree → +20pts")
        parts.append("   - Title in public source → +10pts")
        parts.append("   - Location in public source → +10pts")

    parts.append("")
    parts.append("--- WHAT WILL CHANGE AFTER FIX ---")
    parts.append("Once the above is resolved, the system will:")
    parts.append("  - Re-run Evidence Graph assembly")
    parts.append("  - Re-score Entity Lock with new evidence")
    parts.append("  - Proceed to dossier synthesis (if gates pass)")

    return "\n".join(parts)
