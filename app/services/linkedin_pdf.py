"""LinkedIn PDF ingestion service — text extraction + headshot cropping.

Accepts a LinkedIn profile PDF, extracts:
1. Structured text (for dossier EvidenceNodes)
2. Headshot image (for contact photo)

Text extraction uses PyMuPDF (fitz) when available, falls back to
pdfplumber, then to a basic regex-on-binary approach.

Headshot extraction renders page 1 and crops a fixed region where
LinkedIn places the profile photo. No OpenCV dependency required.

Storage:
- Raw PDF  → pdf_uploads/{contact_id}_{timestamp}.pdf
- Cropped  → image_cache/linkedin_crop_{contact_id}.jpg
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------

PDF_UPLOAD_DIR = Path("./pdf_uploads")
IMAGE_CACHE_DIR = Path("./image_cache")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class LinkedInPDFTextResult:
    """Structured text extracted from a LinkedIn PDF."""
    raw_text: str = ""
    name: str = ""
    headline: str = ""
    location: str = ""
    about: str = ""
    experience: list[dict] = field(default_factory=list)
    education: list[dict] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    sections: dict[str, str] = field(default_factory=dict)
    page_count: int = 0


@dataclass
class LinkedInPDFCropResult:
    """Result of headshot cropping from a LinkedIn PDF."""
    success: bool = False
    image_path: str = ""
    image_bytes: bytes = b""
    width: int = 0
    height: int = 0
    method: str = ""  # "fitz_render" | "pillow_crop" | "failed"
    error: str = ""


@dataclass
class LinkedInPDFIngestResult:
    """Full ingestion result combining text + photo."""
    text_result: LinkedInPDFTextResult = field(default_factory=LinkedInPDFTextResult)
    crop_result: LinkedInPDFCropResult = field(default_factory=LinkedInPDFCropResult)
    pdf_path: str = ""
    pdf_hash: str = ""
    ingested_at: str = ""
    error: str = ""


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_bytes: bytes) -> LinkedInPDFTextResult:
    """Extract structured text from a LinkedIn PDF.

    Tries PyMuPDF first, then falls back to basic extraction.
    """
    result = LinkedInPDFTextResult()

    raw_text = _extract_raw_text(pdf_bytes)
    if not raw_text:
        return result

    result.raw_text = raw_text
    result.page_count = _count_pages(pdf_bytes)

    # Parse LinkedIn-specific structure
    _parse_linkedin_sections(raw_text, result)

    return result


def _extract_raw_text(pdf_bytes: bytes) -> str:
    """Extract raw text from PDF bytes using available library."""
    # Try PyMuPDF (fitz)
    try:
        import fitz  # noqa: F811

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages = []
        for page in doc:
            pages.append(page.get_text())
        doc.close()
        text = "\n\n".join(pages)
        if text.strip():
            return text
    except ImportError:
        logger.debug("PyMuPDF not available, trying fallback")
    except Exception as e:
        logger.warning("PyMuPDF extraction failed: %s", e)

    # Try pdfplumber
    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
            text = "\n\n".join(pages)
            if text.strip():
                return text
    except ImportError:
        logger.debug("pdfplumber not available, trying fallback")
    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)

    # Basic fallback: decode printable ASCII from PDF stream
    try:
        text = pdf_bytes.decode("latin-1", errors="ignore")
        # Extract text between BT/ET (PDF text operators)
        parts = re.findall(r"\(([^)]+)\)", text)
        if parts:
            return " ".join(parts)
    except Exception as e:
        logger.warning("Fallback text extraction failed: %s", e)

    return ""


def _count_pages(pdf_bytes: bytes) -> int:
    """Count pages in PDF."""
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        count = len(doc)
        doc.close()
        return count
    except Exception:
        pass

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return len(pdf.pages)
    except Exception:
        pass

    return 0


def _parse_linkedin_sections(raw_text: str, result: LinkedInPDFTextResult) -> None:
    """Parse LinkedIn PDF text into structured sections.

    LinkedIn PDFs follow a predictable layout:
    - Name (first line or prominent text)
    - Headline (below name)
    - Location
    - About
    - Experience (with company, title, dates)
    - Education
    - Skills
    """
    lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
    if not lines:
        return

    # LinkedIn PDFs typically start with the name
    result.name = lines[0] if lines else ""

    # Build section map from common LinkedIn headers
    section_headers = {
        "about": ["about", "summary"],
        "experience": ["experience"],
        "education": ["education"],
        "skills": ["skills", "skills & endorsements"],
        "licenses": ["licenses & certifications", "certifications"],
        "languages": ["languages"],
        "honors": ["honors & awards", "honors-awards"],
        "volunteer": ["volunteer experience", "volunteering"],
        "publications": ["publications"],
        "projects": ["projects"],
    }

    # Find sections by matching headers
    current_section = "header"
    section_content: dict[str, list[str]] = {"header": []}

    for line in lines:
        line_lower = line.lower().strip()
        matched = False
        for section_key, headers in section_headers.items():
            if line_lower in headers:
                current_section = section_key
                section_content.setdefault(section_key, [])
                matched = True
                break
        if not matched:
            section_content.setdefault(current_section, []).append(line)

    # Extract header info (name, headline, location)
    header_lines = section_content.get("header", [])
    if len(header_lines) >= 1:
        result.name = header_lines[0]
    if len(header_lines) >= 2:
        result.headline = header_lines[1]
    if len(header_lines) >= 3:
        # Location is often the 3rd line in LinkedIn PDFs
        potential_loc = header_lines[2]
        # Simple heuristic: location often contains commas or common patterns
        if "," in potential_loc or any(
            w in potential_loc.lower()
            for w in ["area", "region", "city", "state", "uk", "us", "usa"]
        ):
            result.location = potential_loc

    # About section
    about_lines = section_content.get("about", [])
    if about_lines:
        result.about = "\n".join(about_lines)

    # Experience section
    exp_lines = section_content.get("experience", [])
    if exp_lines:
        result.experience = _parse_experience_section(exp_lines)

    # Education section
    edu_lines = section_content.get("education", [])
    if edu_lines:
        result.education = _parse_education_section(edu_lines)

    # Skills
    skills_lines = section_content.get("skills", [])
    if skills_lines:
        result.skills = [s.strip() for s in skills_lines if s.strip() and len(s.strip()) < 100]

    # Store all sections for dossier use
    for key, content_lines in section_content.items():
        if content_lines:
            result.sections[key] = "\n".join(content_lines)


def _parse_experience_section(lines: list[str]) -> list[dict]:
    """Parse experience entries from LinkedIn PDF text."""
    entries: list[dict] = []
    current: dict = {}

    # Date pattern: "Jan 2020 - Present", "2019 - 2021", etc.
    date_pattern = re.compile(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4}\s*[-–]\s*"
        r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)?\s*\d{4}|Present)",
        re.IGNORECASE,
    )

    for line in lines:
        if date_pattern.search(line):
            if current:
                entries.append(current)
            current = {"dates": line, "lines": []}
        elif current:
            current.setdefault("lines", []).append(line)
        else:
            # First entry without date
            if not current:
                current = {"title": line, "lines": []}

    if current:
        entries.append(current)

    # Structure entries
    structured = []
    for entry in entries:
        item = {
            "dates": entry.get("dates", ""),
            "description": "\n".join(entry.get("lines", [])),
        }
        entry_lines = entry.get("lines", [])
        if entry_lines:
            item["title"] = entry_lines[0] if entry_lines else ""
            if len(entry_lines) > 1:
                item["company"] = entry_lines[1]
        structured.append(item)

    return structured


def _parse_education_section(lines: list[str]) -> list[dict]:
    """Parse education entries from LinkedIn PDF text."""
    entries: list[dict] = []
    current: dict = {}

    for line in lines:
        # Education entries often start with school name (capitalized)
        if line and line[0].isupper() and not current.get("lines"):
            if current:
                entries.append(current)
            current = {"school": line, "lines": []}
        elif current:
            current.setdefault("lines", []).append(line)

    if current:
        entries.append(current)

    return [
        {
            "school": e.get("school", ""),
            "details": "\n".join(e.get("lines", [])),
        }
        for e in entries
    ]


# ---------------------------------------------------------------------------
# Headshot cropping
# ---------------------------------------------------------------------------


def crop_headshot_from_pdf(
    pdf_bytes: bytes,
    contact_id: int,
) -> LinkedInPDFCropResult:
    """Crop the headshot from page 1 of a LinkedIn PDF.

    LinkedIn profile PDFs place the avatar in a predictable position
    on page 1 (typically top-left quadrant, circular).

    Strategy:
    1. Render page 1 to image using PyMuPDF
    2. Crop the avatar region (fixed coordinates based on LinkedIn layout)
    3. Save as JPEG

    Falls back gracefully if PyMuPDF is unavailable.
    """
    result = LinkedInPDFCropResult()

    # Try PyMuPDF rendering
    try:
        import fitz

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]

        # Render at 2x resolution for quality
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        # Crop the avatar region using Pillow
        crop_result = _crop_avatar_from_image(img_bytes, contact_id)
        if crop_result.success:
            return crop_result

        result.method = "fitz_render"
        result.error = "Avatar crop region empty or too small"
    except ImportError:
        logger.debug("PyMuPDF not available for page rendering")
        result.method = "failed"
        result.error = "PyMuPDF not installed — cannot render PDF page for cropping"
    except Exception as e:
        logger.warning("PDF page rendering failed: %s", e)
        result.method = "failed"
        result.error = str(e)

    return result


def _crop_avatar_from_image(
    page_image_bytes: bytes,
    contact_id: int,
) -> LinkedInPDFCropResult:
    """Crop the avatar from a rendered LinkedIn PDF page image.

    LinkedIn PDF layout (at 2x render):
    - Page width: ~1190px (595pt * 2)
    - Page height: ~1684px (842pt * 2)
    - Avatar: circular, ~200x200px, positioned at approximately (60, 60)

    We crop a generous region and let the client display it circular.
    """
    result = LinkedInPDFCropResult()

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(page_image_bytes))
        width, height = img.size

        # LinkedIn PDF avatar region (at 2x render resolution)
        # Typically top-left, roughly 5-20% from left, 3-15% from top
        # Avatar is approximately 130-200px at 2x
        crop_left = int(width * 0.04)
        crop_top = int(height * 0.03)
        crop_right = int(width * 0.20)
        crop_bottom = int(height * 0.15)

        # Ensure square crop centered on the expected avatar position
        crop_size = min(crop_right - crop_left, crop_bottom - crop_top)
        center_x = (crop_left + crop_right) // 2
        center_y = (crop_top + crop_bottom) // 2
        half = crop_size // 2

        avatar = img.crop((
            max(0, center_x - half),
            max(0, center_y - half),
            min(width, center_x + half),
            min(height, center_y + half),
        ))

        # Validate: avatar should have some variance (not blank)
        if not _image_has_content(avatar):
            result.success = False
            result.error = "Cropped region appears blank"
            result.method = "pillow_crop"
            return result

        # Resize to standard avatar size
        avatar = avatar.resize((200, 200), Image.LANCZOS)

        # Save
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = IMAGE_CACHE_DIR / f"linkedin_crop_{contact_id}.jpg"
        avatar_io = io.BytesIO()
        avatar.save(avatar_io, format="JPEG", quality=90)
        avatar_bytes = avatar_io.getvalue()
        output_path.write_bytes(avatar_bytes)

        result.success = True
        result.image_path = str(output_path)
        result.image_bytes = avatar_bytes
        result.width = 200
        result.height = 200
        result.method = "pillow_crop"
        return result

    except ImportError:
        result.error = "Pillow not installed — cannot crop avatar"
        result.method = "failed"
    except Exception as e:
        result.error = f"Avatar cropping failed: {e}"
        result.method = "failed"
        logger.warning("Avatar cropping failed: %s", e)

    return result


def _image_has_content(img) -> bool:
    """Check if a PIL Image has meaningful content (not blank/uniform)."""
    try:
        from PIL import ImageStat

        stat = ImageStat.Stat(img)
        # Check standard deviation — blank images have low stddev
        # Use mean stddev across channels
        stddev = sum(stat.stddev) / len(stat.stddev)
        return stddev > 10.0  # Threshold: uniform images have stddev ~0
    except Exception:
        return True  # Assume content if we can't check


# ---------------------------------------------------------------------------
# Full ingestion pipeline
# ---------------------------------------------------------------------------


def ingest_linkedin_pdf(
    pdf_bytes: bytes,
    contact_id: int,
    contact_name: str = "",
) -> LinkedInPDFIngestResult:
    """Full LinkedIn PDF ingestion: store, extract text, crop headshot.

    Steps:
    1. Store raw PDF to pdf_uploads/
    2. Extract structured text
    3. Crop headshot from page 1
    4. Return combined result

    Never raises — returns error details in the result object.
    """
    result = LinkedInPDFIngestResult(
        ingested_at=datetime.utcnow().isoformat(),
    )

    if not pdf_bytes:
        result.error = "Empty PDF data"
        return result

    # Compute hash for dedup
    result.pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()[:16]

    # Store raw PDF
    try:
        PDF_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"{contact_id}_{timestamp}.pdf"
        pdf_path = PDF_UPLOAD_DIR / pdf_filename
        pdf_path.write_bytes(pdf_bytes)
        result.pdf_path = str(pdf_path)
        logger.info("Stored LinkedIn PDF for contact %d at %s", contact_id, pdf_path)
    except Exception as e:
        logger.exception("Failed to store PDF for contact %d", contact_id)
        result.error = f"PDF storage failed: {e}"
        return result

    # Extract text
    try:
        result.text_result = extract_text_from_pdf(pdf_bytes)
        logger.info(
            "Extracted %d chars from LinkedIn PDF for %s (pages=%d)",
            len(result.text_result.raw_text),
            contact_name or f"contact_{contact_id}",
            result.text_result.page_count,
        )
    except Exception:
        logger.exception("Text extraction failed for contact %d", contact_id)
        result.text_result = LinkedInPDFTextResult()
        # Continue — text failure shouldn't block photo extraction

    # Crop headshot
    try:
        result.crop_result = crop_headshot_from_pdf(pdf_bytes, contact_id)
        if result.crop_result.success:
            logger.info(
                "Cropped headshot for contact %d (%dx%d, method=%s)",
                contact_id,
                result.crop_result.width,
                result.crop_result.height,
                result.crop_result.method,
            )
        else:
            logger.info(
                "Headshot crop failed for contact %d: %s",
                contact_id,
                result.crop_result.error,
            )
    except Exception as e:
        logger.exception("Headshot cropping failed for contact %d", contact_id)
        result.crop_result = LinkedInPDFCropResult(
            success=False, error=str(e), method="failed"
        )

    return result


# ---------------------------------------------------------------------------
# Text-to-EvidenceNodes builder (for Workstream B)
# ---------------------------------------------------------------------------


def build_evidence_nodes_from_pdf(
    text_result: LinkedInPDFTextResult,
    contact_name: str = "",
) -> list[dict]:
    """Convert LinkedIn PDF text into EvidenceNode-compatible dicts.

    Each meaningful section becomes a PDF-type EvidenceNode with:
    - type: "PDF"
    - source: "linkedin_pdf:{contact_name}"
    - snippet: <=200 chars of content
    - ref: section name
    - date: "UNKNOWN" (PDFs don't carry dates reliably)

    Returns list of dicts ready to be added to an EvidenceGraph.
    """
    nodes: list[dict] = []
    source = f"linkedin_pdf:{contact_name}"

    # About section — often the richest content
    if text_result.about:
        # Split about into meaningful chunks (by paragraph or sentence)
        chunks = _split_into_chunks(text_result.about, max_chars=200)
        for i, chunk in enumerate(chunks):
            nodes.append({
                "type": "PDF",
                "source": source,
                "snippet": chunk,
                "ref": f"about:{i+1}",
                "date": "UNKNOWN",
            })

    # Headline
    if text_result.headline:
        nodes.append({
            "type": "PDF",
            "source": source,
            "snippet": text_result.headline[:200],
            "ref": "headline",
            "date": "UNKNOWN",
        })

    # Experience entries
    for i, exp in enumerate(text_result.experience[:10]):
        desc = exp.get("description", "")
        title = exp.get("title", "")
        company = exp.get("company", "")
        dates = exp.get("dates", "")
        summary = f"{title} at {company}" if title and company else title or desc
        if summary:
            nodes.append({
                "type": "PDF",
                "source": source,
                "snippet": summary[:200],
                "ref": f"experience:{i+1}",
                "date": _extract_date_from_text(dates),
            })

    # Education entries
    for i, edu in enumerate(text_result.education[:5]):
        school = edu.get("school", "")
        details = edu.get("details", "")
        summary = f"{school}: {details}" if details else school
        if summary:
            nodes.append({
                "type": "PDF",
                "source": source,
                "snippet": summary[:200],
                "ref": f"education:{i+1}",
                "date": "UNKNOWN",
            })

    # Skills (grouped)
    if text_result.skills:
        skills_text = ", ".join(text_result.skills[:20])
        nodes.append({
            "type": "PDF",
            "source": source,
            "snippet": f"Skills: {skills_text}"[:200],
            "ref": "skills",
            "date": "UNKNOWN",
        })

    # Other sections
    for section_name, content in text_result.sections.items():
        if section_name in ("header", "about", "experience", "education", "skills"):
            continue
        if content.strip():
            chunks = _split_into_chunks(content, max_chars=200)
            for i, chunk in enumerate(chunks[:3]):  # Max 3 chunks per misc section
                nodes.append({
                    "type": "PDF",
                    "source": source,
                    "snippet": chunk,
                    "ref": f"{section_name}:{i+1}",
                    "date": "UNKNOWN",
                })

    return nodes


def _split_into_chunks(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks respecting sentence boundaries."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: list[str] = []
    current = ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip() if current else sentence
        else:
            if current:
                chunks.append(current)
            current = sentence[:max_chars]

    if current:
        chunks.append(current)

    return chunks


def _extract_date_from_text(text: str) -> str:
    """Try to extract a YYYY-MM-DD or YYYY date from text."""
    if not text:
        return "UNKNOWN"

    # Match year ranges like "2020 - Present" or "Jan 2020 - Dec 2022"
    year_match = re.search(r"(\d{4})", text)
    if year_match:
        return year_match.group(1)

    return "UNKNOWN"
