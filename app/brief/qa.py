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

# Explicit gap acknowledgments that count as properly tagged
# (the LLM is correctly flagging missing evidence, not making uncited claims)
GAP_ACKNOWLEDGMENT_PATTERN = re.compile(
    r"no evidence available|no evidence found|not available|"
    r"no (public |internal )?data|no (public |internal )?evidence|"
    r"no appearances found|no results found|"
    r"no (search |visibility )?sweep|not executed|"
    r"category (was )?not searched|remain(s)? unsearched|"
    r"unknown at this time|insufficient (evidence|data)|"
    r"no supporting evidence|cannot be determined|"
    r"no (recorded|documented) (interactions?|meetings?|emails?)",
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
    """Check what percentage of substantive sentences have evidence tags/citations.

    A sentence is considered "tagged" if it contains:
    1. An evidence tag like [VERIFIED-MEETING], [INFERRED-H], [UNKNOWN], etc.
    2. An explicit gap acknowledgment like "No evidence available"

    Lines that are structural (headers, tables, labels, bold-label: value pairs)
    are skipped and don't count toward the total.
    """
    result = EvidenceCoverageResult()

    lines = text.split("\n")
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        # Skip non-substantive lines
        if not line or len(line) < 20:
            continue
        if line.startswith(("#", "|", "---", ">", "- -")):
            continue
        # Skip markdown list items that are just labels (e.g., "* **Category**: value")
        if line.startswith(("*", "-")) and "**" in line:
            # Structural label line — count as non-substantive
            if ":" in line and len(line.split(":")[0]) < 60:
                continue
        # Skip bold-prefix structural lines (section labels, field headers)
        skip_prefixes = (
            "**", "Leverage", "Stress", "Credibility", "Contrarian",
            "High-Upside", "Rank ", "Scenario:", "Confidence",
            "Total:", "Summary", "Each ", "For EACH", "Include:",
            "Output format", "Calculate", "Identify", "Based on",
        )
        if line.startswith(skip_prefixes):
            if ":" in line and len(line.split(":")[0]) < 50:
                continue

        sentences = re.split(r'(?<=[.!?])\s+', line)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:
                continue

            result.total_substantive += 1

            if EVIDENCE_TAG_PATTERN.search(sentence):
                result.tagged_count += 1
            elif GAP_ACKNOWLEDGMENT_PATTERN.search(sentence):
                # Explicit gap acknowledgment counts as proper evidence discipline
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
    - Public evidence: LinkedIn profile resolved: 30 points
    - Employer confirmed by public source: 20 points
    - Multiple independent domains agree: 20 points
    - Title confirmed by public source: 10 points
    - Location confirmed by public source: 10 points
    - Meeting <> public cross-confirmation: 10 points

    Lock thresholds:
    - >= 70: LOCKED
    - 50-69: PARTIAL LOCK
    - < 50: NOT LOCKED

    LinkedIn is only "confirmed" if we can retrieve at least a page title
    or snippet from a public result. If auth-blocked, it becomes
    "LinkedIn URL present but not verifiable without auth."
    """
    score: int = 0  # 0-100
    evidence: list[dict] = field(default_factory=list)
    # Each dict: {"signal": str, "weight": int, "source": str}
    linkedin_confirmed: bool = False
    linkedin_url_present: bool = False  # URL exists but may not be verified
    linkedin_verified_by_retrieval: bool = False  # Verified via public evidence node
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
    pdl_data: dict | None = None,
    pdf_data: dict | None = None,
) -> DisambiguationResult:
    """Score identity lock confidence from 0-100.

    Weights:
    +10 LinkedIn URL present (weak internal evidence)
    +30 LinkedIn verified by retrieval (strong public evidence; replaces +10)
    +20 Meeting confirms name/employer (internal verified)
    +20 Employer confirmed by public source (or by PDL/PDF data)
    +10 Title confirmed by public source (or by PDL/PDF data)
    +10 Location confirmed by public source (or by PDL/PDF data)
    +20 Multiple independent domains agree

    PDL enrichment counts as an independent confirming domain.
    PDL-confirmed company/title/location each contribute their full points
    even without SerpAPI results (PDL is a verified data provider).

    LinkedIn PDF data counts as an independent confirming domain.
    PDF-confirmed company/title/location contribute full points.

    LinkedIn URL present (+10) is REPLACED (not stacked) by verified (+30).
    Meeting data gives +20 for internal confirmation regardless of public signals.
    """
    result = DisambiguationResult()
    search_results = search_results or {}
    apollo_data = apollo_data or {}
    pdl_data = pdl_data or {}
    pdf_data = pdf_data or {}

    name_lower = name.lower()
    company_lower = company.lower() if company else ""

    # Track independent confirming domains for multi-domain bonus
    confirming_domains: set[str] = set()

    # --- PDL enrichment credit (runs before search-based scoring) ---
    pdl_company = (pdl_data.get("canonical_company") or "").lower()
    pdl_title = (pdl_data.get("canonical_title") or "").lower()
    pdl_location = (pdl_data.get("canonical_location") or "").lower()
    pdl_confidence = pdl_data.get("pdl_match_confidence", 0)

    pdl_company_matched = False
    pdl_title_matched = False
    pdl_location_matched = False

    if pdl_confidence and pdl_confidence > 0.5:
        confirming_domains.add("pdl")

        # PDL confirms employer
        if pdl_company and company_lower:
            if (
                company_lower in pdl_company
                or pdl_company in company_lower
            ):
                pdl_company_matched = True
                result.employer_match = True
                result.company_match = True
                result.score += 20
                result.evidence.append({
                    "signal": (
                        f"Employer '{company}' confirmed by PDL enrichment "
                        f"(confidence: {pdl_confidence:.0%})"
                    ),
                    "weight": 20,
                    "source": "pdl",
                })
            elif pdl_company:
                # PDL returned a company but it doesn't match — still note it
                result.evidence.append({
                    "signal": (
                        f"PDL company mismatch: expected '{company}', "
                        f"got '{pdl_data.get('canonical_company', '')}'"
                    ),
                    "weight": 0,
                    "source": "pdl",
                })
        elif pdl_company and not company_lower:
            # No company provided but PDL has one — accept as confirmed
            pdl_company_matched = True
            result.employer_match = True
            result.company_match = True
            result.score += 15
            result.evidence.append({
                "signal": (
                    f"Employer set by PDL: '{pdl_data.get('canonical_company', '')}' "
                    f"(confidence: {pdl_confidence:.0%})"
                ),
                "weight": 15,
                "source": "pdl",
            })

        # PDL confirms title
        if pdl_title:
            title_lower = title.lower() if title else ""
            pdl_title_words = [w for w in pdl_title.split() if len(w) >= 3]
            user_title_words = [w for w in title_lower.split() if len(w) >= 2]
            # Match if: any PDL word in user title, OR any user word in PDL title,
            # OR exact substring match in either direction
            title_matched = (
                (pdl_title_words and any(w in title_lower for w in pdl_title_words))
                or (user_title_words and any(w in pdl_title for w in user_title_words))
                or (title_lower and (
                    title_lower in pdl_title or pdl_title in title_lower
                ))
            )
            if title_lower and title_matched:
                pdl_title_matched = True
                result.title_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Title '{title}' confirmed by PDL: "
                        f"'{pdl_data.get('canonical_title', '')}'"
                    ),
                    "weight": 10,
                    "source": "pdl",
                })
            elif not title_lower:
                # No title provided but PDL has one
                pdl_title_matched = True
                result.title_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Title set by PDL: '{pdl_data.get('canonical_title', '')}'"
                    ),
                    "weight": 10,
                    "source": "pdl",
                })

        # PDL confirms location
        if pdl_location:
            location_lower = location.lower() if location else ""
            if location_lower and (
                location_lower in pdl_location or pdl_location in location_lower
            ):
                pdl_location_matched = True
                result.location_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Location '{location}' confirmed by PDL: "
                        f"'{pdl_data.get('canonical_location', '')}'"
                    ),
                    "weight": 10,
                    "source": "pdl",
                })
            elif not location_lower:
                pdl_location_matched = True
                result.location_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Location set by PDL: '{pdl_data.get('canonical_location', '')}'"
                    ),
                    "weight": 10,
                    "source": "pdl",
                })

    # --- LinkedIn PDF credit (user-uploaded LinkedIn profile export) ---
    pdf_company = (pdf_data.get("company") or "").lower()
    pdf_title = (pdf_data.get("title") or pdf_data.get("headline") or "").lower()
    pdf_location = (pdf_data.get("location") or "").lower()
    pdf_has_text = pdf_data.get("text_usable", False)

    pdf_company_matched = False
    pdf_title_matched = False
    pdf_location_matched = False

    if pdf_has_text:
        confirming_domains.add("linkedin_pdf")

        # PDF confirms employer (only if not already confirmed by PDL)
        if pdf_company and company_lower and not pdl_company_matched:
            if (
                company_lower in pdf_company
                or pdf_company in company_lower
            ):
                pdf_company_matched = True
                result.employer_match = True
                result.company_match = True
                result.score += 20
                result.evidence.append({
                    "signal": (
                        f"Employer '{company}' confirmed by LinkedIn PDF"
                    ),
                    "weight": 20,
                    "source": "linkedin_pdf",
                })
            elif pdf_company:
                result.evidence.append({
                    "signal": (
                        f"PDF company mismatch: expected '{company}', "
                        f"got '{pdf_data.get('company', '')}'"
                    ),
                    "weight": 0,
                    "source": "linkedin_pdf",
                })
        elif pdf_company and not company_lower and not pdl_company_matched:
            # No company provided but PDF has one — accept as confirmed
            pdf_company_matched = True
            result.employer_match = True
            result.company_match = True
            result.score += 15
            result.evidence.append({
                "signal": (
                    f"Employer set by LinkedIn PDF: '{pdf_data.get('company', '')}'"
                ),
                "weight": 15,
                "source": "linkedin_pdf",
            })

        # PDF confirms title (only if not already confirmed by PDL)
        if pdf_title and not pdl_title_matched:
            title_lower = title.lower() if title else ""
            pdf_title_words = [w for w in pdf_title.split() if len(w) >= 3]
            user_title_words = [w for w in title_lower.split() if len(w) >= 2]
            title_matched = (
                (pdf_title_words and any(w in title_lower for w in pdf_title_words))
                or (user_title_words and any(w in pdf_title for w in user_title_words))
                or (title_lower and (
                    title_lower in pdf_title or pdf_title in title_lower
                ))
            )
            if title_lower and title_matched:
                pdf_title_matched = True
                result.title_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Title '{title}' confirmed by LinkedIn PDF: "
                        f"'{pdf_data.get('title') or pdf_data.get('headline', '')}'"
                    ),
                    "weight": 10,
                    "source": "linkedin_pdf",
                })
            elif not title_lower:
                pdf_title_matched = True
                result.title_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Title set by LinkedIn PDF: "
                        f"'{pdf_data.get('title') or pdf_data.get('headline', '')}'"
                    ),
                    "weight": 10,
                    "source": "linkedin_pdf",
                })

        # PDF confirms location (only if not already confirmed by PDL)
        if pdf_location and not pdl_location_matched:
            location_lower = location.lower() if location else ""
            if location_lower and (
                location_lower in pdf_location or pdf_location in location_lower
            ):
                pdf_location_matched = True
                result.location_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Location '{location}' confirmed by LinkedIn PDF: "
                        f"'{pdf_data.get('location', '')}'"
                    ),
                    "weight": 10,
                    "source": "linkedin_pdf",
                })
            elif not location_lower:
                pdf_location_matched = True
                result.location_match = True
                result.score += 10
                result.evidence.append({
                    "signal": (
                        f"Location set by LinkedIn PDF: '{pdf_data.get('location', '')}'"
                    ),
                    "weight": 10,
                    "source": "linkedin_pdf",
                })

    # LinkedIn scoring: URL present (+10) OR verified by retrieval (+30)
    if linkedin_url and linkedin_url.startswith("http"):
        result.linkedin_url_present = True

        # Check if we can retrieve any public evidence from LinkedIn search
        linkedin_results = search_results.get("linkedin", [])
        linkedin_verified = False
        for lr in linkedin_results:
            lr_title = lr.get("title", "")
            lr_snippet = lr.get("snippet", "")
            lr_text = f"{lr_title} {lr_snippet}".lower()
            if name_lower in lr_text and (lr_title or lr_snippet):
                linkedin_verified = True
                result.linkedin_confirmed = True
                result.linkedin_verified_by_retrieval = True
                result.name_match = True
                result.score += 30
                confirming_domains.add("linkedin")
                result.evidence.append({
                    "signal": (
                        f"LinkedIn verified via retrieval: {lr_title[:80]}"
                    ),
                    "weight": 30,
                    "source": lr.get("link", "LinkedIn"),
                })
                break

        if not linkedin_verified:
            # URL present but not verified — weak internal evidence (+10)
            result.score += 10
            result.evidence.append({
                "signal": (
                    "LinkedIn URL present (not yet verified via retrieval). "
                    f"URL: {linkedin_url}"
                ),
                "weight": 10,
                "source": "user_input",
            })
    else:
        # No URL — check if LinkedIn search finds them (reduced points)
        linkedin_results = search_results.get("linkedin", [])
        for lr in linkedin_results:
            lr_text = f"{lr.get('title', '')} {lr.get('snippet', '')}".lower()
            if name_lower in lr_text:
                result.name_match = True
                result.score += 15  # Partial credit for search-only match
                confirming_domains.add("linkedin")
                result.evidence.append({
                    "signal": (
                        f"Name found in LinkedIn result (no URL): "
                        f"{lr.get('title', '')[:80]}"
                    ),
                    "weight": 15,
                    "source": lr.get("link", "LinkedIn"),
                })
                break

    # Meeting confirmation (+20) — internal verified evidence
    # Awarded whenever meeting data exists, regardless of public signals.
    if has_meeting_data:
        result.meeting_confirmed = True
        result.score += 20
        has_public_signal = bool(confirming_domains)
        if has_public_signal:
            result.evidence.append({
                "signal": (
                    "Meeting data confirms identity (cross-confirms with public evidence)"
                ),
                "weight": 20,
                "source": "internal_data",
            })
        else:
            result.evidence.append({
                "signal": (
                    "Meeting/email records confirm this person exists internally"
                ),
                "weight": 20,
                "source": "internal_data",
            })

    # Employer match (20 pts) — company confirmed across sources
    # Skip if already fully confirmed by PDL or PDF (avoids double-counting)
    if company_lower and not pdl_company_matched and not pdf_company_matched:
        employer_sources = 0
        for category in search_results:
            for r in search_results.get(category, []):
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                if company_lower in r_text and name_lower in r_text:
                    employer_sources += 1
                    confirming_domains.add(category)
                    break

        if employer_sources >= 1:
            result.employer_match = True
            result.company_match = True
            pts = min(20, employer_sources * 10)
            result.score += pts
            result.evidence.append({
                "signal": (
                    f"Employer '{company}' confirmed in "
                    f"{employer_sources} source(s)"
                ),
                "weight": pts,
                "source": "cross-reference",
            })

        # Apollo as employer source
        if apollo_data and apollo_data.get("title"):
            apollo_company = (
                apollo_data.get("organization", {}).get("name", "") or ""
            ).lower()
            if company_lower and (
                company_lower in apollo_company or apollo_company in company_lower
            ):
                confirming_domains.add("apollo")
                if not result.employer_match:
                    result.employer_match = True
                    result.company_match = True
                    result.score += 10
                    result.evidence.append({
                        "signal": (
                            f"Employer confirmed via Apollo enrichment: {company}"
                        ),
                        "weight": 10,
                        "source": "apollo",
                    })

    # Title match (10 pts) — title confirmed in non-LinkedIn sources
    # Skip if already confirmed by PDL or PDF
    if title and not pdl_title_matched and not pdf_title_matched:
        title_lower = title.lower()
        for category in ["general", "news", "company_site"]:
            for r in search_results.get(category, []):
                r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
                title_words = [w for w in title_lower.split() if len(w) >= 3]
                if title_words and any(w in r_text for w in title_words):
                    result.title_match = True
                    result.score += 10
                    confirming_domains.add(category)
                    result.evidence.append({
                        "signal": (
                            f"Title '{title}' matched in {category}: "
                            f"{r.get('title', '')[:80]}"
                        ),
                        "weight": 10,
                        "source": r.get("link", category),
                    })
                    break
            if result.title_match:
                break

    # Location match (10 pts)
    # Skip if already confirmed by PDL or PDF
    if location and not pdl_location_matched and not pdf_location_matched:
        location_lower = location.lower()
        for category in search_results:
            for r in search_results.get(category, []):
                if location_lower in (r.get("snippet") or "").lower():
                    result.location_match = True
                    result.score += 10
                    confirming_domains.add(category)
                    result.evidence.append({
                        "signal": (
                            f"Location '{location}' found in search results"
                        ),
                        "weight": 10,
                        "source": r.get("link", category),
                    })
                    break
            if result.location_match:
                break

    # Multiple independent domains agree (20 pts)
    # Requires 3+ confirming domains (e.g. linkedin + news + company_site)
    if len(confirming_domains) >= 3:
        result.multiple_sources_agree = True
        result.score += 20
        result.evidence.append({
            "signal": (
                f"Multiple independent domains agree "
                f"({len(confirming_domains)} domains: "
                f"{', '.join(sorted(confirming_domains))})"
            ),
            "weight": 20,
            "source": "cross-reference",
        })
    elif len(confirming_domains) >= 2:
        # Partial credit for 2 domains
        result.multiple_sources_agree = True
        result.score += 10
        result.evidence.append({
            "signal": (
                f"Two independent domains agree "
                f"({', '.join(sorted(confirming_domains))})"
            ),
            "weight": 10,
            "source": "cross-reference",
        })

    # Photo available (informational, no points)
    if apollo_data and apollo_data.get("photo_url"):
        result.photo_available = True

    # Secondary source match (informational flag, counted via multi-domain)
    secondary_categories = ["news", "registry", "talks", "authored", "registry_us"]
    for category in secondary_categories:
        for r in search_results.get(category, []):
            r_text = f"{r.get('title', '')} {r.get('snippet', '')}".lower()
            if name_lower in r_text:
                result.secondary_source_match = True
                confirming_domains.add(category)
                result.evidence.append({
                    "signal": (
                        f"Secondary source ({category}): "
                        f"{r.get('title', '')[:80]}"
                    ),
                    "weight": 0,
                    "source": r.get("link", category),
                })
                break
        if result.secondary_source_match:
            break

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


# ---------------------------------------------------------------------------
# Fail-Closed Enforcement (Evidence Graph integration)
# ---------------------------------------------------------------------------


def enforce_fail_closed_gates(
    dossier_text: str,
    entity_lock_score: int,
    visibility_ledger_count: int,
    evidence_coverage_pct: float,
    person_name: str = "",
    has_public_results: bool = True,
    web_results_count: int = 0,
) -> tuple[bool, str]:
    """Enforce fail-closed gates and return (should_output, message).

    The evidence coverage gate is adaptive:
    - High-visibility contacts (10+ web results): 85% threshold
    - Medium-visibility contacts (5-9 web results): 70% threshold
    - Low-visibility contacts (<5 web results): 60% threshold

    This prevents the system from permanently blocking dossiers for
    contacts who simply don't have much public presence.

    Returns:
        (True, "") if all gates pass — output the dossier.
        (False, failure_text) if any hard gate fails — output the failure text INSTEAD.
    """
    failures: list[str] = []

    # Gate 0: Must have at least one public retrieval result
    if not has_public_results:
        failures.append(
            "FAIL: NO PUBLIC RETRIEVAL RESULTS\n"
            f'SerpAPI returned 0 results for "{person_name}".\n'
            "Entity Lock cannot be computed without public evidence.\n"
            "Verify the person name and re-run retrieval."
        )

    # Gate 1: Visibility sweep must have been executed (>= 8 queries)
    # Lowered from 12 to 8 — some visibility categories consistently
    # return 0 results for non-public figures, and forcing 12+ queries
    # doesn't improve coverage.
    if visibility_ledger_count == 0:
        failures.append(
            "FAIL: VISIBILITY SWEEP NOT EXECUTED\n"
            "The retrieval ledger contains 0 visibility-intent rows.\n"
            "The dossier cannot be produced without executing the visibility sweep.\n"
            f'Run the visibility query battery for "{person_name}" and log each result.'
        )
    elif visibility_ledger_count < 8:
        failures.append(
            f"FAIL: INSUFFICIENT VISIBILITY QUERIES ({visibility_ledger_count}/8)\n"
            f"Only {visibility_ledger_count} visibility queries logged. "
            "Cannot claim 'none found' unless at least 8 queries executed.\n"
            f'Run remaining queries for "{person_name}" and log each result.'
        )

    # Gate 2: Evidence coverage — adaptive threshold based on evidence availability
    # Contacts with sparse public presence shouldn't be permanently blocked.
    if web_results_count >= 10:
        coverage_threshold = 85.0
    elif web_results_count >= 5:
        coverage_threshold = 70.0
    else:
        coverage_threshold = 60.0

    if evidence_coverage_pct < coverage_threshold:
        failures.append(
            f"FAIL: EVIDENCE COVERAGE {evidence_coverage_pct:.1f}%\n"
            f"Coverage must be >= {coverage_threshold:.0f}% "
            f"(adaptive: {web_results_count} web results). "
            f"Current: {evidence_coverage_pct:.1f}%.\n"
            "Sentences without evidence tags must be cited or removed."
        )

    if failures:
        header = "DOSSIER GENERATION HALTED — FAIL-CLOSED GATES FAILED\n"
        header += "=" * 60 + "\n"
        body = "\n\n".join(failures)
        lock_status = "LOCKED" if entity_lock_score >= 70 else (
            "PARTIAL" if entity_lock_score >= 50 else "NOT LOCKED"
        )
        footer = (
            f"\n\nEntity Lock: {entity_lock_score}/100 ({lock_status})\n"
            "Fix the above failures and re-run."
        )
        return False, header + body + footer

    return True, ""
