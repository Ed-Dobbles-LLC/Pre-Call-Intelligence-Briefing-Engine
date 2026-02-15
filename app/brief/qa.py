"""Quality assurance gates for intelligence dossiers.

Implements:
1. Generic Filler Linter — rejects sentences that could describe 50% of executives
2. Evidence Coverage Gate — fails if < 85% of substantive sentences lack tags
3. Contradiction Detector — flags mismatched titles/dates/companies across sources
4. Identity Lock Scorer — scores identity confidence 0-100 with explicit weights
5. Person-Level Ratio — fails if >40% of content is company recap
6. Strategic Snapshot Validator — fails if >2 bullets don't mention the person
7. INFERRED-H Auditor — flags INFERRED-H claims without upstream signal citations
8. QA Report Generator — produces a structured QA report
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Generic Filler Detection
# ---------------------------------------------------------------------------

# Patterns that are generic enterprise filler when not followed by evidence
GENERIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\b(strategic leader|visionary leader|thought leader)\b", re.IGNORECASE),
    re.compile(r"\b(data-driven|results-driven|outcome-driven|metrics-driven)\b", re.IGNORECASE),
    re.compile(r"\b(passionate about|deeply committed to|focused on delivering)\b", re.IGNORECASE),
    re.compile(r"\b(transformative|game-changing|cutting-edge|world-class)\b", re.IGNORECASE),
    re.compile(r"\b(leveraging (AI|data|technology) to)\b", re.IGNORECASE),
    re.compile(r"\b(drives? (innovation|growth|results|value))\b", re.IGNORECASE),
    re.compile(r"\b(human-centered|customer-centric|people-first)\b", re.IGNORECASE),
    re.compile(
        r"\b(screens? for|looks? for|values?) (authenticity|integrity|excellence)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(proven track record|extensive experience|seasoned professional)\b", re.IGNORECASE),
    re.compile(r"\b(ROI-focused|ROI driven|bottom-line)\b", re.IGNORECASE),
    re.compile(r"\b(likely (data-driven|strategic|analytical|methodical))\b", re.IGNORECASE),
    re.compile(r"\b(strong (communicator|leader|advocate))\b", re.IGNORECASE),
    re.compile(r"\b(empowers? (teams?|people|employees|organizations?))\b", re.IGNORECASE),
    re.compile(r"\b(bridges? the gap between)\b", re.IGNORECASE),
    re.compile(r"\b(at the intersection of)\b", re.IGNORECASE),
    re.compile(r"\b(synergies?|synergistic|holistic approach)\b", re.IGNORECASE),
    re.compile(r"\b(best[- ]in[- ]class|industry[- ]leading|next[- ]gen(eration)?)\b", re.IGNORECASE),
    re.compile(r"\b(digital transformation journey)\b", re.IGNORECASE),
    re.compile(r"\b(results[- ]oriented|growth[- ]oriented)\b", re.IGNORECASE),
    re.compile(r"\b(likely implements corrective measures)\b", re.IGNORECASE),
    re.compile(r"\b(focuses on growth)\b", re.IGNORECASE),
]

# Evidence tags that make a sentence "cited"
EVIDENCE_TAG_PATTERN = re.compile(
    r"\[(VERIFIED|INFERRED|UNKNOWN|SOURCE)[^\]]*\]"
    r"|`\[(VERIFIED|INFERRED|UNKNOWN)[^\]]*\]`",
    re.IGNORECASE,
)


@dataclass
class GenericFillerResult:
    """Result of scanning text for generic filler."""
    flagged_sentences: list[dict] = field(default_factory=list)
    # Each dict: {"sentence": str, "pattern": str, "line": int}
    total_sentences: int = 0
    generic_count: int = 0

    @property
    def genericness_score(self) -> int:
        """0-100 score. Higher = more generic (worse)."""
        if self.total_sentences == 0:
            return 0
        return min(100, int((self.generic_count / self.total_sentences) * 100))


def lint_generic_filler(text: str) -> GenericFillerResult:
    """Scan text for generic enterprise filler.

    A sentence is flagged if it matches a generic pattern AND does not
    contain an evidence tag (which would anchor it to evidence).
    """
    result = GenericFillerResult()

    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        # Split into sentences (rough)
        sentences = re.split(r'(?<=[.!?])\s+', line.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            # Skip markdown headers, table rows, metadata
            if sentence.startswith(("#", "|", "---", "*Generated", "*No ")):
                continue

            result.total_sentences += 1

            # Check if sentence has an evidence tag
            has_tag = bool(EVIDENCE_TAG_PATTERN.search(sentence))

            for pattern in GENERIC_PATTERNS:
                match = pattern.search(sentence)
                if match and not has_tag:
                    result.flagged_sentences.append({
                        "sentence": sentence[:200],
                        "pattern": match.group(0),
                        "line": line_num,
                    })
                    result.generic_count += 1
                    break  # One flag per sentence

    return result


# ---------------------------------------------------------------------------
# Evidence Coverage Gate
# ---------------------------------------------------------------------------

@dataclass
class EvidenceCoverageResult:
    """Result of checking evidence/citation coverage."""
    total_substantive: int = 0
    tagged_count: int = 0
    untagged_sentences: list[dict] = field(default_factory=list)
    # Each dict: {"sentence": str, "line": int}

    @property
    def coverage_pct(self) -> float:
        """Percentage of substantive sentences with evidence tags."""
        if self.total_substantive == 0:
            return 100.0
        return (self.tagged_count / self.total_substantive) * 100

    @property
    def passes(self) -> bool:
        """True if coverage >= 85%."""
        return self.coverage_pct >= 85.0


def check_evidence_coverage(text: str) -> EvidenceCoverageResult:
    """Check what percentage of substantive sentences have evidence tags/citations."""
    result = EvidenceCoverageResult()

    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        # Skip non-substantive lines
        if not line or len(line) < 20:
            continue
        if line.startswith(("#", "|", "---", "*", ">")):
            continue
        skip_prefixes = (
            "**", "Leverage", "Stress", "Credibility", "Contrarian",
            "High-Upside", "Rank ", "Scenario:", "Confidence",
        )
        if line.startswith(skip_prefixes):
            # These are section labels
            if ":" in line and len(line.split(":")[0]) < 40:
                continue

        sentences = re.split(r'(?<=[.!?])\s+', line)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            result.total_substantive += 1

            if EVIDENCE_TAG_PATTERN.search(sentence):
                result.tagged_count += 1
            else:
                result.untagged_sentences.append({
                    "sentence": sentence[:200],
                    "line": line_num,
                })

    return result


# ---------------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------------

@dataclass
class Contradiction:
    """A detected contradiction between sources."""
    field: str  # e.g. "title", "company", "location"
    value_a: str
    source_a: str
    value_b: str
    source_b: str
    severity: str = "medium"  # low | medium | high


def detect_contradictions(
    claims: list[dict],
) -> list[Contradiction]:
    """Detect contradictions across a list of claims.

    Each claim dict should have:
    - field: str (e.g. "title", "company", "date")
    - value: str
    - source: str (e.g. "meeting", "linkedin", "company_site")

    Returns a list of contradictions found.
    """
    contradictions = []

    # Group claims by field
    by_field: dict[str, list[dict]] = {}
    for claim in claims:
        f = claim.get("field", "")
        if f not in by_field:
            by_field[f] = []
        by_field[f].append(claim)

    for field_name, field_claims in by_field.items():
        if len(field_claims) < 2:
            continue

        # Compare all pairs
        for i in range(len(field_claims)):
            for j in range(i + 1, len(field_claims)):
                a = field_claims[i]
                b = field_claims[j]

                val_a = a.get("value", "").strip().lower()
                val_b = b.get("value", "").strip().lower()

                if not val_a or not val_b:
                    continue

                # Skip if values are the same
                if val_a == val_b:
                    continue

                # Check if they're actually contradictory (not just different granularity)
                if val_a in val_b or val_b in val_a:
                    continue

                severity = "medium"
                if field_name in ("title", "company", "role"):
                    severity = "high"
                elif field_name in ("location", "date"):
                    severity = "medium"

                contradictions.append(Contradiction(
                    field=field_name,
                    value_a=a.get("value", ""),
                    source_a=a.get("source", "unknown"),
                    value_b=b.get("value", ""),
                    source_b=b.get("source", "unknown"),
                    severity=severity,
                ))

    return contradictions


# ---------------------------------------------------------------------------
# Identity Lock Scorer
# ---------------------------------------------------------------------------

@dataclass
class DisambiguationResult:
    """Result of identity lock scoring.

    Weights (100 total):
    - LinkedIn confirmed: 40 points
    - Employer match: 20 points
    - Meeting cross-confirmation: 15 points
    - Title match: 10 points
    - Secondary source match: 10 points
    - Location match: 5 points

    Lock thresholds:
    - >= 70: LOCKED
    - 50-69: PARTIAL LOCK
    - < 50: NOT LOCKED
    """
    score: int = 0  # 0-100
    evidence: list[dict] = field(default_factory=list)
    # Each dict: {"signal": str, "weight": int, "source": str}
    linkedin_confirmed: bool = False
    employer_match: bool = False
    meeting_confirmed: bool = False
    title_match: bool = False
    secondary_source_match: bool = False
    location_match: bool = False
    # Legacy compat aliases
    name_match: bool = False
    company_match: bool = False
    photo_available: bool = False
    multiple_sources_agree: bool = False

    @property
    def is_locked(self) -> bool:
        """True if score >= 70 (LOCKED)."""
        return self.score >= 70

    @property
    def is_partial(self) -> bool:
        """True if 50 <= score < 70 (PARTIAL LOCK)."""
        return 50 <= self.score < 70

    @property
    def lock_status(self) -> str:
        """Human-readable lock status."""
        if self.score >= 70:
            return "LOCKED"
        if self.score >= 50:
            return "PARTIAL LOCK"
        return "NOT LOCKED"


def score_disambiguation(
    name: str,
    company: str = "",
    title: str = "",
    linkedin_url: str = "",
    location: str = "",
    search_results: dict | None = None,
    apollo_data: dict | None = None,
    has_meeting_data: bool = False,
) -> DisambiguationResult:
    """Score identity lock confidence from 0-100.

    Weights:
    +40 LinkedIn confirmed (URL provided and matches name/company in search)
    +20 Employer match (company found across multiple sources)
    +15 Meeting cross-confirmation (internal meeting data references this person)
    +10 Title match (title confirmed in secondary sources)
    +10 Secondary source match (news, registry, or authored content)
    +5  Location match (geographic location confirmed)
    """
    result = DisambiguationResult()
    search_results = search_results or {}
    apollo_data = apollo_data or {}

    name_lower = name.lower()
    company_lower = company.lower() if company else ""

    # LinkedIn confirmed (40 pts) — URL exists + name appears in LinkedIn results
    if linkedin_url and linkedin_url.startswith("http"):
        result.linkedin_confirmed = True
        pts = 25  # base points for having a URL
        result.evidence.append({
            "signal": f"LinkedIn URL provided: {linkedin_url}",
            "weight": 25,
            "source": "user_input",
        })

        # Additional points if LinkedIn search results confirm the name
        linkedin_results = search_results.get("linkedin", [])
        for lr in linkedin_results:
            lr_text = f"{lr.get('title', '')} {lr.get('snippet', '')}".lower()
            if name_lower in lr_text:
                result.name_match = True
                pts += 15
                result.evidence.append({
                    "signal": f"Name confirmed in LinkedIn search: {lr.get('title', '')[:80]}",
                    "weight": 15,
                    "source": lr.get("link", "LinkedIn"),
                })
                break

        result.score += min(40, pts)
    else:
        # No URL — check if LinkedIn search alone finds them
        linkedin_results = search_results.get("linkedin", [])
        for lr in linkedin_results:
            lr_text = f"{lr.get('title', '')} {lr.get('snippet', '')}".lower()
            if name_lower in lr_text:
                result.name_match = True
                result.score += 20
                result.evidence.append({
                    "signal": f"Name found in LinkedIn result (no URL): {lr.get('title', '')[:80]}",
                    "weight": 20,
                    "source": lr.get("link", "LinkedIn"),
                })
                break

    # Employer match (20 pts) — company confirmed across sources
    if company_lower:
        employer_sources = 0
        for category in search_results:
            for r in search_results.get(category, []):
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if company_lower in r_text and name_lower in r_text:
                    employer_sources += 1
                    break

        if employer_sources >= 1:
            result.employer_match = True
            result.company_match = True
            pts = min(20, employer_sources * 10)
            result.score += pts
            result.evidence.append({
                "signal": f"Employer '{company}' confirmed in {employer_sources} source(s)",
                "weight": pts,
                "source": "cross-reference",
            })

        # Apollo as employer source
        if apollo_data and apollo_data.get("title"):
            apollo_company = (apollo_data.get("organization", {}).get("name", "") or "").lower()
            if company_lower and (company_lower in apollo_company or apollo_company in company_lower):
                if not result.employer_match:
                    result.employer_match = True
                    result.company_match = True
                    result.score += 10
                    result.evidence.append({
                        "signal": f"Employer confirmed via Apollo enrichment: {company}",
                        "weight": 10,
                        "source": "apollo",
                    })

    # Meeting cross-confirmation (15 pts)
    if has_meeting_data:
        result.meeting_confirmed = True
        result.score += 15
        result.evidence.append({
            "signal": "Person appears in internal meeting/email records",
            "weight": 15,
            "source": "internal_data",
        })

    # Title match (10 pts) — title confirmed in non-LinkedIn sources
    if title:
        title_lower = title.lower()
        for category in ["general", "news", "company_site"]:
            for r in search_results.get(category, []):
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                # Check for title keywords (words > 3 chars)
                title_words = [w for w in title_lower.split() if len(w) >= 3]
                if title_words and any(w in r_text for w in title_words):
                    result.title_match = True
                    result.score += 10
                    result.evidence.append({
                        "signal": f"Title '{title}' matched in {category}: {r.get('title', '')[:80]}",
                        "weight": 10,
                        "source": r.get("link", category),
                    })
                    break
            if result.title_match:
                break

    # Secondary source match (10 pts) — news, registry, or authored content
    secondary_categories = ["news", "registry", "talks", "authored", "registry_us"]
    for category in secondary_categories:
        for r in search_results.get(category, []):
            r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
            if name_lower in r_text:
                result.secondary_source_match = True
                result.score += 10
                result.evidence.append({
                    "signal": f"Secondary source ({category}): {r.get('title', '')[:80]}",
                    "weight": 10,
                    "source": r.get("link", category),
                })
                break
        if result.secondary_source_match:
            break

    # Location match (5 pts)
    if location:
        location_lower = location.lower()
        for category in search_results:
            for r in search_results.get(category, []):
                if location_lower in (r.get("snippet") or "").lower():
                    result.location_match = True
                    result.score += 5
                    result.evidence.append({
                        "signal": f"Location '{location}' found in search results",
                        "weight": 5,
                        "source": r.get("link", category),
                    })
                    break
            if result.location_match:
                break

    # Photo available (informational, no points)
    if apollo_data and apollo_data.get("photo_url"):
        result.photo_available = True

    # Multiple sources agree (informational flag)
    source_count = sum([
        result.linkedin_confirmed,
        result.employer_match,
        result.title_match,
        result.secondary_source_match,
    ])
    if source_count >= 3:
        result.multiple_sources_agree = True

    # Cap at 100
    result.score = min(100, result.score)

    return result


# ---------------------------------------------------------------------------
# Person-Level Ratio
# ---------------------------------------------------------------------------

@dataclass
class PersonLevelResult:
    """Result of checking person-level vs company-level content ratio."""
    total_lines: int = 0
    person_lines: int = 0
    company_lines: int = 0
    person_pct: float = 100.0

    @property
    def passes(self) -> bool:
        """True if person-level content >= 60%."""
        return self.person_pct >= 60.0


# Patterns indicating person-level content
_PERSON_PRONOUNS = re.compile(
    r"\b(he|she|they|him|her|their|his|the subject|this person)\b",
    re.IGNORECASE,
)

# Patterns indicating company-level content (generic company statements)
_COMPANY_PATTERNS = re.compile(
    r"\b(the company|the organization|the firm|the business|corporate strategy|"
    r"company revenue|company growth|the market|industry trends|sector)\b",
    re.IGNORECASE,
)


def check_person_level_ratio(text: str, person_name: str = "") -> PersonLevelResult:
    """Check whether the dossier is person-focused or company-recap.

    A line is person-level if it contains the person's name, a personal
    pronoun, or evidence tags tied to individual actions.
    A line is company-level if it discusses the company generically without
    connecting to the individual.
    """
    result = PersonLevelResult()
    name_lower = person_name.lower() if person_name else ""
    name_parts = [p for p in name_lower.split() if len(p) > 2]

    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line or len(line) < 15:
            continue
        if line.startswith(("#", "|", "---", "*", ">")):
            continue

        result.total_lines += 1
        line_lower = line.lower()

        is_person = False
        is_company = False

        # Check person signals
        if name_parts and any(p in line_lower for p in name_parts):
            is_person = True
        elif _PERSON_PRONOUNS.search(line):
            is_person = True

        # Check company signals
        if _COMPANY_PATTERNS.search(line) and not is_person:
            is_company = True

        if is_person:
            result.person_lines += 1
        elif is_company:
            result.company_lines += 1
        else:
            # Ambiguous lines count as person-level (benefit of doubt)
            result.person_lines += 1

    if result.total_lines > 0:
        result.person_pct = (result.person_lines / result.total_lines) * 100
    else:
        result.person_pct = 100.0

    return result


# ---------------------------------------------------------------------------
# Strategic Snapshot Validator
# ---------------------------------------------------------------------------

@dataclass
class SnapshotValidation:
    """Result of validating Strategic Snapshot bullet focus."""
    total_bullets: int = 0
    person_bullets: int = 0
    non_person_bullets: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """True if no more than 2 bullets lack person reference."""
        return len(self.non_person_bullets) <= 2


def check_snapshot_person_focus(text: str, person_name: str = "") -> SnapshotValidation:
    """Validate that Strategic Snapshot bullets mention the person directly.

    Extracts the Strategic Snapshot section and checks each bullet point.
    Fails if more than 2 bullets don't mention the person.
    """
    result = SnapshotValidation()
    name_lower = person_name.lower() if person_name else ""
    name_parts = [p for p in name_lower.split() if len(p) > 2]

    # Find the Strategic Snapshot section
    in_snapshot = False
    for line in text.split("\n"):
        stripped = line.strip()

        # Detect section headers
        if re.match(r"^#{1,4}\s.*\b(strategic\s+(identity\s+)?snapshot)\b", stripped, re.IGNORECASE):
            in_snapshot = True
            continue
        if in_snapshot and re.match(r"^#{1,4}\s", stripped) and "snapshot" not in stripped.lower():
            break  # Next section

        if not in_snapshot:
            continue

        # Check bullets
        if stripped.startswith(("-", "*", "•")) and len(stripped) > 10:
            result.total_bullets += 1
            line_lower = stripped.lower()

            has_person_ref = False
            if name_parts and any(p in line_lower for p in name_parts):
                has_person_ref = True
            elif _PERSON_PRONOUNS.search(stripped):
                has_person_ref = True

            if has_person_ref:
                result.person_bullets += 1
            else:
                result.non_person_bullets.append(stripped[:150])

    return result


# ---------------------------------------------------------------------------
# INFERRED-H Auditor
# ---------------------------------------------------------------------------

@dataclass
class InferredHAudit:
    """Result of auditing INFERRED-H usage."""
    total_inferred_h: int = 0
    with_upstream: int = 0
    without_upstream: list[dict] = field(default_factory=list)
    # Each dict: {"sentence": str, "line": int}

    @property
    def passes(self) -> bool:
        """True if all INFERRED-H claims cite upstream signals."""
        return len(self.without_upstream) == 0


_INFERRED_H_PATTERN = re.compile(r"\[INFERRED[–\-]H(IGH)?\]", re.IGNORECASE)
_UPSTREAM_PATTERN = re.compile(
    r"(because|based on|from|per|citing|signals?|evidence|meeting|transcript|source|"
    r"indicated by|confirmed by|suggests|observed in|per .{3,30} call)",
    re.IGNORECASE,
)


def audit_inferred_h(text: str) -> InferredHAudit:
    """Audit that INFERRED-H claims cite upstream signals.

    INFERRED-H should only be used when there are strong converging signals.
    Each usage should be accompanied by reasoning.
    """
    result = InferredHAudit()

    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not _INFERRED_H_PATTERN.search(line):
            continue

        result.total_inferred_h += 1

        if _UPSTREAM_PATTERN.search(line):
            result.with_upstream += 1
        else:
            result.without_upstream.append({
                "sentence": line[:200],
                "line": line_num,
            })

    return result


# ---------------------------------------------------------------------------
# QA Report
# ---------------------------------------------------------------------------

@dataclass
class QAReport:
    """Complete QA report for a generated dossier."""
    genericness: GenericFillerResult = field(default_factory=GenericFillerResult)
    evidence_coverage: EvidenceCoverageResult = field(default_factory=EvidenceCoverageResult)
    contradictions: list[Contradiction] = field(default_factory=list)
    disambiguation: DisambiguationResult = field(default_factory=DisambiguationResult)
    person_level: PersonLevelResult = field(default_factory=PersonLevelResult)
    snapshot_validation: SnapshotValidation = field(default_factory=SnapshotValidation)
    inferred_h_audit: InferredHAudit = field(default_factory=InferredHAudit)
    top_claims_to_verify: list[str] = field(default_factory=list)
    hallucination_risk_flags: list[str] = field(default_factory=list)

    @property
    def passes_all(self) -> bool:
        """True if all gates pass."""
        return (
            self.genericness.genericness_score <= 20
            and self.evidence_coverage.passes
            and len(self.contradictions) == 0
            and self.person_level.passes
        )


def generate_qa_report(
    dossier_text: str,
    claims: list[dict] | None = None,
    disambiguation: DisambiguationResult | None = None,
    person_name: str = "",
) -> QAReport:
    """Run all QA gates on a generated dossier and produce a report."""
    report = QAReport()

    # 1. Generic filler lint
    report.genericness = lint_generic_filler(dossier_text)

    # 2. Evidence coverage (target >= 85%)
    report.evidence_coverage = check_evidence_coverage(dossier_text)

    # 3. Contradictions
    if claims:
        report.contradictions = detect_contradictions(claims)

    # 4. Disambiguation / identity lock
    if disambiguation:
        report.disambiguation = disambiguation

    # 5. Person-level ratio (target >= 60%)
    report.person_level = check_person_level_ratio(dossier_text, person_name)

    # 6. Strategic Snapshot bullet validation
    report.snapshot_validation = check_snapshot_person_focus(dossier_text, person_name)

    # 7. INFERRED-H audit
    report.inferred_h_audit = audit_inferred_h(dossier_text)

    # 8. Hallucination risk flags
    if report.genericness.genericness_score > 30:
        report.hallucination_risk_flags.append(
            f"High genericness score ({report.genericness.genericness_score}%) — "
            "output may contain filler not grounded in evidence"
        )
    if not report.evidence_coverage.passes:
        report.hallucination_risk_flags.append(
            f"Low evidence coverage ({report.evidence_coverage.coverage_pct:.0f}%) — "
            f"{len(report.evidence_coverage.untagged_sentences)} sentences lack citations"
        )
    if report.contradictions:
        high_sev = [c for c in report.contradictions if c.severity == "high"]
        if high_sev:
            report.hallucination_risk_flags.append(
                f"{len(high_sev)} high-severity contradictions detected across sources"
            )
    if not report.person_level.passes:
        report.hallucination_risk_flags.append(
            f"Company-heavy content ({100 - report.person_level.person_pct:.0f}% company-level) — "
            "dossier is a company recap, not a person profile"
        )
    if not report.snapshot_validation.passes:
        report.hallucination_risk_flags.append(
            f"{len(report.snapshot_validation.non_person_bullets)} Strategic Snapshot bullets "
            "do not mention the person directly"
        )
    if not report.inferred_h_audit.passes:
        report.hallucination_risk_flags.append(
            f"{len(report.inferred_h_audit.without_upstream)} INFERRED-H claims lack "
            "upstream signal citations"
        )

    return report


def render_qa_report_markdown(report: QAReport) -> str:
    """Render a QA report as markdown."""
    lines = []
    lines.append("## QA Report")
    lines.append("")

    # Genericness
    score = report.genericness.genericness_score
    status = "PASS" if score <= 20 else "WARN" if score < 40 else "FAIL"
    lines.append(f"**Genericness Score:** {score}/100 [{status}]")
    if report.genericness.flagged_sentences:
        lines.append("")
        lines.append("Flagged generic phrases:")
        for f in report.genericness.flagged_sentences[:5]:
            lines.append(f"- Line {f['line']}: \"{f['pattern']}\" in: {f['sentence'][:100]}...")
    lines.append("")

    # Evidence coverage
    cov = report.evidence_coverage
    status = "PASS" if cov.passes else "FAIL"
    lines.append(
        f"**Evidence Coverage:** {cov.coverage_pct:.0f}% "
        f"({cov.tagged_count}/{cov.total_substantive}) [{status}]"
    )
    if not cov.passes and cov.untagged_sentences:
        lines.append("")
        lines.append("Untagged sentences (first 5):")
        for u in cov.untagged_sentences[:5]:
            lines.append(f"- Line {u['line']}: {u['sentence'][:100]}...")
    lines.append("")

    # Person-level ratio
    plr = report.person_level
    status = "PASS" if plr.passes else "FAIL"
    lines.append(f"**Person-Level Ratio:** {plr.person_pct:.0f}% [{status}]")
    if not plr.passes:
        lines.append(
            f"  {plr.company_lines} company-level lines vs {plr.person_lines} person-level"
        )
    lines.append("")

    # Strategic Snapshot validation
    sv = report.snapshot_validation
    if sv.total_bullets > 0:
        status = "PASS" if sv.passes else "FAIL"
        lines.append(
            f"**Snapshot Focus:** {sv.person_bullets}/{sv.total_bullets} bullets "
            f"mention person [{status}]"
        )
        if not sv.passes:
            for b in sv.non_person_bullets[:3]:
                lines.append(f"  - Missing person ref: {b[:100]}...")
        lines.append("")

    # INFERRED-H audit
    ih = report.inferred_h_audit
    if ih.total_inferred_h > 0:
        status = "PASS" if ih.passes else "WARN"
        lines.append(
            f"**INFERRED-H Audit:** {ih.with_upstream}/{ih.total_inferred_h} cite "
            f"upstream signals [{status}]"
        )
        if not ih.passes:
            for u in ih.without_upstream[:3]:
                lines.append(f"  - Line {u['line']}: {u['sentence'][:100]}...")
        lines.append("")

    # Contradictions
    if report.contradictions:
        lines.append(f"**Contradictions:** {len(report.contradictions)} found")
        for c in report.contradictions:
            lines.append(
                f"- [{c.severity.upper()}] {c.field}: "
                f"\"{c.value_a}\" ({c.source_a}) vs \"{c.value_b}\" ({c.source_b})"
            )
    else:
        lines.append("**Contradictions:** None detected")
    lines.append("")

    # Disambiguation / identity lock
    d = report.disambiguation
    lines.append(f"**Identity Lock:** {d.score}/100 ({d.lock_status})")
    if d.evidence:
        for e in d.evidence:
            lines.append(f"- +{e['weight']}pts: {e['signal']}")
    lines.append("")

    # Hallucination risk
    if report.hallucination_risk_flags:
        lines.append("**Hallucination Risk Flags:**")
        for flag in report.hallucination_risk_flags:
            lines.append(f"- {flag}")
    else:
        lines.append("**Hallucination Risk:** Low")
    lines.append("")

    # Top claims to verify
    if report.top_claims_to_verify:
        lines.append("**Top 5 Claims to Verify Next:**")
        for i, claim in enumerate(report.top_claims_to_verify[:5], 1):
            lines.append(f"{i}. {claim}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Strict QA Gates
# ---------------------------------------------------------------------------

STRICT_EVIDENCE_THRESHOLD = 95.0

# Additional genericness patterns for strict mode
STRICT_GENERIC_PATTERNS: list[re.Pattern] = [
    re.compile(r"\blikely\b", re.IGNORECASE),
    re.compile(r"\bmay\b", re.IGNORECASE),
    re.compile(r"\bcould\b", re.IGNORECASE),
    re.compile(r"\bgenerally\b", re.IGNORECASE),
    re.compile(r"\btypically\b", re.IGNORECASE),
    re.compile(r"\brank [123]/[123] scenario\b", re.IGNORECASE),
]


def prune_uncited_claims(text: str) -> str:
    """Remove substantive lines that lack any evidence tag.

    Splits *text* by newlines and drops lines that are substantive (>20 chars,
    not headers/table rows) yet contain no evidence tag matching
    ``EVIDENCE_TAG_PATTERN``.  All other lines are kept as-is.

    Returns the pruned text with uncited substantive lines removed.
    """
    lines = text.split("\n")
    kept: list[str] = []
    for line in lines:
        stripped = line.strip()
        # Keep non-substantive lines unconditionally
        if not stripped or len(stripped) <= 20:
            kept.append(line)
            continue
        if stripped.startswith(("#", "|", "---", "*", ">")):
            kept.append(line)
            continue
        # Substantive line — keep only if it has an evidence tag
        if EVIDENCE_TAG_PATTERN.search(stripped):
            kept.append(line)
        # else: drop the line (uncited substantive claim)
    return "\n".join(kept)


def compute_gate_status(
    identity_lock_score: int,
    evidence_coverage_pct: float,
    genericness_score: int,
    strict: bool = False,
) -> str:
    """Determine the overall gate status from individual QA scores.

    Returns one of:
    - ``"passed"``      — all gates pass
    - ``"constrained"`` — identity lock < 70; brief must be limited to
                          meeting/email evidence only
    - ``"failed"``      — evidence coverage below threshold or genericness > 20
    - ``"not_run"``     — default / indeterminate
    """
    threshold = STRICT_EVIDENCE_THRESHOLD if strict else 85.0

    if evidence_coverage_pct < threshold or genericness_score > 20:
        return "failed"

    if identity_lock_score < 70:
        return "constrained"

    if (
        identity_lock_score >= 70
        and evidence_coverage_pct >= threshold
        and genericness_score <= 20
    ):
        return "passed"

    return "not_run"


def lint_generic_filler_strict(text: str) -> GenericFillerResult:
    """Scan text for generic enterprise filler using both standard and strict patterns.

    Behaves like :func:`lint_generic_filler` but additionally checks against
    ``STRICT_GENERIC_PATTERNS`` for a more aggressive detection pass.
    """
    result = GenericFillerResult()
    all_patterns = GENERIC_PATTERNS + STRICT_GENERIC_PATTERNS

    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        sentences = re.split(r'(?<=[.!?])\s+', line.strip())
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:
                continue
            if sentence.startswith(("#", "|", "---", "*Generated", "*No ")):
                continue

            result.total_sentences += 1

            has_tag = bool(EVIDENCE_TAG_PATTERN.search(sentence))

            for pattern in all_patterns:
                match = pattern.search(sentence)
                if match and not has_tag:
                    result.flagged_sentences.append({
                        "sentence": sentence[:200],
                        "pattern": match.group(0),
                        "line": line_num,
                    })
                    result.generic_count += 1
                    break  # One flag per sentence

    return result


def check_strict_coverage(result: EvidenceCoverageResult) -> bool:
    """Check if coverage passes the strict 95% threshold."""
    return result.coverage_pct >= STRICT_EVIDENCE_THRESHOLD


# ---------------------------------------------------------------------------
# Public Visibility Sweep QA Gates
# ---------------------------------------------------------------------------

# The 3 mandatory sweep category groups
SWEEP_TED_TEDX = {"ted", "tedx"}
SWEEP_PODCAST_WEBINAR = {"podcast", "webinar"}
SWEEP_CONFERENCE_KEYNOTE = {"conference", "keynote", "summit"}


@dataclass
class VisibilitySweepAudit:
    """Result of auditing public visibility sweep execution."""
    sweep_executed: bool = False
    categories_searched: list[str] = field(default_factory=list)
    ted_tedx_searched: bool = False
    podcast_webinar_searched: bool = False
    conference_keynote_searched: bool = False
    total_results: int = 0
    hard_failures: list[str] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """True if all mandatory sweep groups were searched."""
        return len(self.hard_failures) == 0


def audit_visibility_sweep(
    categories_searched: list[str],
    sweep_executed: bool = False,
) -> VisibilitySweepAudit:
    """Audit whether the mandatory visibility sweep was fully executed.

    Hard failures:
    - TED/TEDx sweep not executed
    - No podcast/webinar sweep
    - No conference/keynote sweep
    """
    result = VisibilitySweepAudit(
        sweep_executed=sweep_executed,
        categories_searched=categories_searched,
    )

    searched_set = set(categories_searched)

    # Check TED/TEDx group
    if searched_set & SWEEP_TED_TEDX:
        result.ted_tedx_searched = True
    else:
        result.hard_failures.append("TED/TEDx sweep not executed")

    # Check podcast/webinar group
    if searched_set & SWEEP_PODCAST_WEBINAR:
        result.podcast_webinar_searched = True
    else:
        result.hard_failures.append("No podcast/webinar sweep executed")

    # Check conference/keynote group
    if searched_set & SWEEP_CONFERENCE_KEYNOTE:
        result.conference_keynote_searched = True
    else:
        result.hard_failures.append("No conference/keynote sweep executed")

    if not sweep_executed:
        result.hard_failures.insert(0, "Visibility sweep was not executed at all")

    return result


def generate_dossier_qa_report(
    dossier_text: str,
    claims: list[dict] | None = None,
    disambiguation: DisambiguationResult | None = None,
    person_name: str = "",
    visibility_categories: list[str] | None = None,
    visibility_sweep_executed: bool = False,
) -> QAReport:
    """Run all QA gates on a generated dossier including visibility sweep audit.

    Extends generate_qa_report with visibility sweep hard failure checks.
    """
    report = generate_qa_report(
        dossier_text=dossier_text,
        claims=claims,
        disambiguation=disambiguation,
        person_name=person_name,
    )

    # Visibility sweep audit
    if visibility_categories is not None:
        sweep_audit = audit_visibility_sweep(
            categories_searched=visibility_categories,
            sweep_executed=visibility_sweep_executed,
        )
        if not sweep_audit.passes:
            for failure in sweep_audit.hard_failures:
                report.hallucination_risk_flags.append(f"VISIBILITY SWEEP: {failure}")

    return report
