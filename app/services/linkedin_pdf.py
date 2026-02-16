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
    """Extract raw text from PDF bytes using available library.

    Browser-saved LinkedIn PDFs often use CIDFonts with non-standard encodings
    that produce garbled text from basic extraction. This function tries multiple
    strategies in order of reliability:

    1. PyMuPDF get_text() — standard text extraction
    2. PyMuPDF get_text("blocks") — block-level extraction (may handle some fonts better)
    3. PyMuPDF get_text("html") — HTML extraction with tag stripping
    4. OCR via pytesseract — render pages to images and OCR (most robust for CID fonts)
    5. pdfplumber — alternative PDF library
    6. Regex fallback — last resort, usually garbled

    After each attempt, the text is checked for quality (non-garbled content).
    """
    # Try PyMuPDF with multiple strategies
    try:
        import fitz  # noqa: F811

        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # Strategy 1: Standard text extraction
        pages = []
        for page in doc:
            pages.append(page.get_text())
        text = "\n\n".join(pages)
        if text.strip() and not _is_garbled_text(text):
            doc.close()
            return text
        logger.info(
            "PyMuPDF standard extraction returned garbled text (%d chars, "
            "%.0f%% non-printable), trying alternatives",
            len(text), _garbled_ratio(text) * 100,
        )

        # Strategy 2: Block-level extraction (sorted by position)
        pages = []
        for page in doc:
            blocks = page.get_text("blocks")
            # Sort by vertical position then horizontal
            blocks.sort(key=lambda b: (b[1], b[0]))
            page_text = "\n".join(
                b[4].strip() for b in blocks
                if b[6] == 0 and b[4].strip()  # type 0 = text blocks
            )
            if page_text:
                pages.append(page_text)
        text = "\n\n".join(pages)
        if text.strip() and not _is_garbled_text(text):
            doc.close()
            return text

        # Strategy 3: HTML extraction with tag stripping
        pages = []
        for page in doc:
            html = page.get_text("html")
            # Strip HTML tags to get plain text
            clean = re.sub(r"<[^>]+>", " ", html)
            clean = re.sub(r"&[a-zA-Z]+;", " ", clean)
            clean = re.sub(r"\s+", " ", clean).strip()
            if clean:
                pages.append(clean)
        text = "\n\n".join(pages)
        if text.strip() and not _is_garbled_text(text):
            doc.close()
            return text

        # Strategy 4: OCR fallback — render each page and run tesseract
        ocr_text = _ocr_pdf_pages(doc)
        doc.close()
        if ocr_text and not _is_garbled_text(ocr_text):
            logger.info("OCR extraction succeeded (%d chars)", len(ocr_text))
            return ocr_text

        # If all fitz strategies returned garbled text, DO NOT return it.
        # Garbled text is worse than empty: it gets stored in the profile,
        # used in dossier generation, and shown to the user as mojibake.
        if text.strip():
            logger.warning(
                "All PyMuPDF strategies returned garbled text (%d chars, "
                "%.0f%% non-printable); returning empty to avoid storing garbage",
                len(text), _garbled_ratio(text) * 100,
            )
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
            if text.strip() and not _is_garbled_text(text):
                return text
    except ImportError:
        logger.debug("pdfplumber not available, trying fallback")
    except Exception as e:
        logger.warning("pdfplumber extraction failed: %s", e)

    # Basic fallback: decode printable ASCII from PDF stream
    try:
        text = pdf_bytes.decode("latin-1", errors="ignore")
        # Extract text between parentheses (PDF text operators)
        parts = re.findall(r"\(([^)]+)\)", text)
        if parts:
            joined = " ".join(parts)
            if not _is_garbled_text(joined):
                return joined
    except Exception as e:
        logger.warning("Fallback text extraction failed: %s", e)

    return ""


def _is_garbled_text(text: str) -> bool:
    """Detect if extracted text is garbled/binary rather than readable.

    Browser-saved LinkedIn PDFs with CIDFont encodings produce text where
    many characters are non-ASCII Latin-1 codepoints that look correct
    individually (À, Ø, ¼, etc.) but form no coherent words.

    Detection strategy:
    1. Check ratio of non-ASCII characters (> 0x7F). Real English text
       with accents has <5% non-ASCII; garbled CIDFont output has 20%+.
    2. Check for very short texts that lack enough content to be useful.
    """
    if not text or len(text.strip()) < 20:
        return True
    return _garbled_ratio(text) > 0.25


def _garbled_ratio(text: str) -> float:
    """Calculate ratio of non-ASCII/non-printable characters in text.

    Normal English text: ~0-3% non-ASCII (occasional accents)
    Garbled CIDFont text: 20-60% non-ASCII (random Latin-1 chars)
    """
    if not text:
        return 1.0
    non_ascii = 0
    total = 0
    for ch in text:
        if ch in ("\n", "\r", "\t", " "):
            continue
        total += 1
        code = ord(ch)
        # Count anything outside printable ASCII (32-126) as suspicious
        if code < 32 or code > 126:
            non_ascii += 1
    return non_ascii / total if total > 0 else 1.0


def _ocr_pdf_pages(doc) -> str:
    """OCR PDF pages using pytesseract as fallback for garbled text.

    Renders each page to a high-resolution image, then runs OCR.
    Only used when standard text extraction produces garbled output.
    """
    try:
        import pytesseract  # noqa: F811
        import fitz
        from PIL import Image

        pages = []
        for page_num, page in enumerate(doc):
            if page_num >= 10:  # Limit to first 10 pages
                break
            # Render at 2x for OCR quality
            mat = fitz.Matrix(2, 2)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img, lang="eng")
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages)
    except ImportError:
        logger.debug("pytesseract not available for OCR fallback")
        return ""
    except Exception as e:
        logger.warning("OCR extraction failed: %s", e)
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
    """Extract the headshot from a LinkedIn PDF.

    DISABLED: Browser-saved LinkedIn PDFs do not reliably embed profile photos.
    The images that ARE embedded are typically LinkedIn UI elements (icons,
    banners, placeholder graphics) that produce garbage crops. This causes
    good enrichment photos (from Apollo/PDL) to be overwritten with corrupted
    images.

    This function now always returns a failed result to prevent photo regression.
    The enrichment pipeline (Apollo, PDL) remains the authoritative photo source.
    Users can manually upload photos via the photo upload feature.
    """
    result = LinkedInPDFCropResult()
    result.method = "disabled"
    result.error = (
        "PDF headshot extraction disabled — browser-saved PDFs do not embed "
        "profile photos reliably. Use enrichment or manual upload instead."
    )
    logger.info(
        "PDF headshot extraction skipped for contact %d (disabled — unreliable)",
        contact_id,
    )
    return result


def _extract_embedded_headshot(doc, contact_id: int) -> LinkedInPDFCropResult:
    """Extract the profile photo from embedded images in the PDF.

    Browser-saved LinkedIn PDFs embed the profile photo as an image object.
    We look for images that match profile-photo characteristics:
    - On page 1 (or first 2 pages)
    - Roughly square aspect ratio
    - Reasonable size (not tiny icons, not full-page backgrounds)
    - Positioned in the upper portion of the page
    """
    result = LinkedInPDFCropResult(method="embedded_extract")

    try:
        from PIL import Image

        candidates = []

        for page_num in range(min(2, len(doc))):
            page = doc[page_num]
            image_list = page.get_images(full=True)

            for img_index, img_info in enumerate(image_list):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    if not base_image or not base_image.get("image"):
                        continue

                    img_data = base_image["image"]
                    img_w = base_image.get("width", 0)
                    img_h = base_image.get("height", 0)

                    # Skip tiny images (icons, bullets, etc.)
                    if img_w < 50 or img_h < 50:
                        continue

                    # Skip very large images (backgrounds, banners)
                    if img_w > 2000 or img_h > 2000:
                        continue

                    # Calculate aspect ratio — profile photos are roughly square
                    aspect = max(img_w, img_h) / max(min(img_w, img_h), 1)

                    # Score the image: prefer square, medium-sized, on page 1
                    area = img_w * img_h
                    score = 0.0
                    # Prefer square (aspect 1.0 = perfect, >2 = too wide/tall)
                    if aspect <= 1.5:
                        score += 40
                    elif aspect <= 2.0:
                        score += 20
                    # Prefer reasonable avatar size (100-800px range)
                    if 80 <= min(img_w, img_h) <= 800:
                        score += 30
                    # Prefer page 1
                    if page_num == 0:
                        score += 20
                    # Prefer medium area (not too small, not banner-sized)
                    if 5000 < area < 500000:
                        score += 10

                    candidates.append({
                        "xref": xref,
                        "data": img_data,
                        "width": img_w,
                        "height": img_h,
                        "aspect": aspect,
                        "score": score,
                        "page": page_num,
                    })
                except Exception as e:
                    logger.debug("Failed to extract image xref %d: %s", xref, e)
                    continue

        if not candidates:
            result.error = "No suitable embedded images found"
            return result

        # Pick the best candidate
        candidates.sort(key=lambda c: c["score"], reverse=True)
        best = candidates[0]

        logger.info(
            "Found %d candidate images; best: %dx%d, aspect=%.1f, score=%.0f, page=%d",
            len(candidates), best["width"], best["height"],
            best["aspect"], best["score"], best["page"],
        )

        # Open and resize to standard avatar size
        img = Image.open(io.BytesIO(best["data"]))

        # Convert to RGB if necessary (CMYK, P, LA, etc.)
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        # Make square by center-cropping
        w, h = img.size
        if w != h:
            side = min(w, h)
            left = (w - side) // 2
            top = (h - side) // 2
            img = img.crop((left, top, left + side, top + side))

        # Validate content
        if not _image_has_content(img):
            result.error = "Best candidate image appears blank"
            return result

        # Resize to standard 200x200
        img = img.resize((200, 200), Image.LANCZOS)

        # Save
        IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        output_path = IMAGE_CACHE_DIR / f"linkedin_crop_{contact_id}.jpg"
        avatar_io = io.BytesIO()
        img.save(avatar_io, format="JPEG", quality=90)
        avatar_bytes = avatar_io.getvalue()
        output_path.write_bytes(avatar_bytes)

        result.success = True
        result.image_path = str(output_path)
        result.image_bytes = avatar_bytes
        result.width = 200
        result.height = 200
        return result

    except ImportError:
        result.error = "Pillow not installed — cannot process extracted images"
        result.method = "failed"
    except Exception as e:
        result.error = f"Embedded image extraction failed: {e}"
        logger.warning("Embedded image extraction failed: %s", e)

    return result


def _crop_avatar_from_rendered(
    page_image_bytes: bytes,
    contact_id: int,
) -> LinkedInPDFCropResult:
    """Crop the avatar from a rendered LinkedIn PDF page image.

    Fallback strategy: renders page 1 at 2x and crops the region where
    LinkedIn typically places the profile photo (upper-left area).

    LinkedIn PDF layout (at 2x render):
    - Page width: ~1190px (595pt * 2)
    - Page height: ~1684px (842pt * 2)
    - Avatar: circular, ~200x200px, positioned at approximately (60, 60)
    """
    result = LinkedInPDFCropResult(method="pillow_crop")

    try:
        from PIL import Image

        img = Image.open(io.BytesIO(page_image_bytes))
        width, height = img.size

        # LinkedIn PDF avatar region (at 2x render resolution)
        # Typically top-left, roughly 5-20% from left, 3-15% from top
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

        if not _image_has_content(avatar):
            result.error = "Cropped region appears blank"
            return result

        avatar = avatar.resize((200, 200), Image.LANCZOS)

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
