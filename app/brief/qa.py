"""Quality assurance gates for intelligence dossiers.

Implements:
1. Generic Filler Linter — rejects sentences that could describe 50% of executives
2. Evidence Coverage Gate — fails if < 80% of substantive sentences lack tags
3. Contradiction Detector — flags mismatched titles/dates/companies across sources
4. Disambiguation Scorer — scores identity lock confidence 0-100
5. QA Report Generator — produces a structured QA report
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
    re.compile(r"\b(screens? for|looks? for|values?) (authenticity|integrity|excellence)\b", re.IGNORECASE),
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
        """True if coverage >= 80%."""
        return self.coverage_pct >= 80.0


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
        if line.startswith(("**", "Leverage", "Stress", "Credibility", "Contrarian", "High-Upside")):
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
# Disambiguation Scorer
# ---------------------------------------------------------------------------

@dataclass
class DisambiguationResult:
    """Result of identity disambiguation scoring."""
    score: int = 0  # 0-100
    evidence: list[dict] = field(default_factory=list)
    # Each dict: {"signal": str, "weight": int, "source": str}
    name_match: bool = False
    company_match: bool = False
    title_match: bool = False
    linkedin_confirmed: bool = False
    location_match: bool = False
    photo_available: bool = False
    multiple_sources_agree: bool = False

    @property
    def is_locked(self) -> bool:
        """True if score >= 70 (high enough to trust identity)."""
        return self.score >= 70


def score_disambiguation(
    name: str,
    company: str = "",
    title: str = "",
    linkedin_url: str = "",
    location: str = "",
    search_results: dict | None = None,
    apollo_data: dict | None = None,
) -> DisambiguationResult:
    """Score identity disambiguation confidence from 0-100.

    Weights:
    - Name + company match in LinkedIn: 30 points
    - LinkedIn URL confirmed: 20 points
    - Title match across sources: 15 points
    - Apollo enrichment data: 15 points
    - Location match: 10 points
    - Multiple source agreement: 10 points
    """
    result = DisambiguationResult()
    search_results = search_results or {}
    apollo_data = apollo_data or {}

    name_lower = name.lower()
    company_lower = company.lower() if company else ""

    # LinkedIn match (30 pts)
    linkedin_results = search_results.get("linkedin", [])
    for lr in linkedin_results:
        lr_title = (lr.get("title") or "").lower()
        lr_snippet = (lr.get("snippet") or "").lower()

        if name_lower in lr_title or name_lower in lr_snippet:
            result.name_match = True
            result.score += 15
            result.evidence.append({
                "signal": f"Name found in LinkedIn result: {lr.get('title', '')[:80]}",
                "weight": 15,
                "source": lr.get("link", "LinkedIn"),
            })

            if company_lower and (company_lower in lr_title or company_lower in lr_snippet):
                result.company_match = True
                result.score += 15
                result.evidence.append({
                    "signal": f"Company match in LinkedIn: {company}",
                    "weight": 15,
                    "source": lr.get("link", "LinkedIn"),
                })
            break

    # LinkedIn URL confirmed (20 pts)
    if linkedin_url and linkedin_url.startswith("http"):
        result.linkedin_confirmed = True
        result.score += 20
        result.evidence.append({
            "signal": f"LinkedIn URL provided: {linkedin_url}",
            "weight": 20,
            "source": "user_input",
        })

    # Title match (15 pts)
    if title:
        title_lower = title.lower()
        for category in ["general", "linkedin", "news"]:
            for r in search_results.get(category, []):
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if title_lower in r_text or any(
                    word in r_text for word in title_lower.split() if len(word) > 3
                ):
                    result.title_match = True
                    result.score += 15
                    result.evidence.append({
                        "signal": f"Title '{title}' found in {category}: {r.get('title', '')[:80]}",
                        "weight": 15,
                        "source": r.get("link", category),
                    })
                    break
            if result.title_match:
                break

    # Apollo data (15 pts)
    if apollo_data:
        if apollo_data.get("name") or apollo_data.get("title"):
            result.score += 15
            result.photo_available = bool(apollo_data.get("photo_url"))
            result.evidence.append({
                "signal": "Apollo enrichment data available",
                "weight": 15,
                "source": "apollo",
            })

    # Location match (10 pts)
    if location:
        location_lower = location.lower()
        for category in search_results:
            for r in search_results.get(category, []):
                if location_lower in (r.get("snippet") or "").lower():
                    result.location_match = True
                    result.score += 10
                    result.evidence.append({
                        "signal": f"Location '{location}' found in search results",
                        "weight": 10,
                        "source": r.get("link", category),
                    })
                    break
            if result.location_match:
                break

    # Multiple source agreement (10 pts)
    sources_with_match = sum([
        result.name_match,
        result.linkedin_confirmed,
        result.title_match,
        bool(apollo_data),
    ])
    if sources_with_match >= 3:
        result.multiple_sources_agree = True
        result.score += 10
        result.evidence.append({
            "signal": f"{sources_with_match} independent sources agree on identity",
            "weight": 10,
            "source": "cross-reference",
        })

    # Cap at 100
    result.score = min(100, result.score)

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
    top_claims_to_verify: list[str] = field(default_factory=list)
    hallucination_risk_flags: list[str] = field(default_factory=list)

    @property
    def passes_all(self) -> bool:
        """True if all gates pass."""
        return (
            self.genericness.genericness_score < 20
            and self.evidence_coverage.passes
            and len(self.contradictions) == 0
        )


def generate_qa_report(
    dossier_text: str,
    claims: list[dict] | None = None,
    disambiguation: DisambiguationResult | None = None,
) -> QAReport:
    """Run all QA gates on a generated dossier and produce a report."""
    report = QAReport()

    # 1. Generic filler lint
    report.genericness = lint_generic_filler(dossier_text)

    # 2. Evidence coverage
    report.evidence_coverage = check_evidence_coverage(dossier_text)

    # 3. Contradictions
    if claims:
        report.contradictions = detect_contradictions(claims)

    # 4. Disambiguation
    if disambiguation:
        report.disambiguation = disambiguation

    # 5. Hallucination risk flags
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

    return report


def render_qa_report_markdown(report: QAReport) -> str:
    """Render a QA report as markdown."""
    lines = []
    lines.append("## QA Report")
    lines.append("")

    # Genericness
    score = report.genericness.genericness_score
    status = "PASS" if score < 20 else "WARN" if score < 40 else "FAIL"
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
    lines.append(f"**Evidence Coverage:** {cov.coverage_pct:.0f}% ({cov.tagged_count}/{cov.total_substantive}) [{status}]")
    if not cov.passes and cov.untagged_sentences:
        lines.append("")
        lines.append("Untagged sentences (first 5):")
        for u in cov.untagged_sentences[:5]:
            lines.append(f"- Line {u['line']}: {u['sentence'][:100]}...")
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

    # Disambiguation
    d = report.disambiguation
    lines.append(f"**Identity Disambiguation:** {d.score}/100 ({'LOCKED' if d.is_locked else 'UNCONFIRMED'})")
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
