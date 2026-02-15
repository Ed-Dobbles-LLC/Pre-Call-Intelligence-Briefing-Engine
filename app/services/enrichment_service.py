"""Contact enrichment service using People Data Labs (PDL).

Enriches a contact by:
1. Loading contact from DB
2. Choosing best identifier (email > linkedin_url > name+company > name+location)
3. Calling PDL API
4. Persisting enrichment results
5. Downloading and storing photo server-side (never wipe existing photo)

NEVER overwrites an existing RESOLVED photo unless the new one downloads successfully.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime
from pathlib import Path

import httpx

from app.clients.pdl_client import PDLClient, PDLEnrichResult
from app.services.photo_resolution import PhotoSource, PhotoStatus

logger = logging.getLogger(__name__)

IMAGE_CACHE_DIR = Path("./image_cache")


async def enrich_contact(
    profile_data: dict,
    contact_id: int,
    contact_name: str = "",
) -> dict:
    """Enrich a contact using PDL.

    Returns dict with:
    - success: bool
    - fields_updated: list of field names that were updated
    - photo_updated: bool
    - match_confidence: float
    - error: str (if failed)
    """
    result = {
        "success": False,
        "fields_updated": [],
        "photo_updated": False,
        "match_confidence": 0.0,
        "error": "",
        "pdl_person_id": "",
    }

    # Determine best identifiers
    emails = profile_data.get("emails", [])
    email = emails[0] if isinstance(emails, list) and emails else profile_data.get("email", "")
    linkedin_url = profile_data.get("linkedin_url", "")
    name = contact_name or profile_data.get("name", "")
    company = profile_data.get("company", "")
    location = profile_data.get("location", "")

    # Must have at least one identifier
    if not email and not linkedin_url and not name:
        result["error"] = "No identifiers available for enrichment"
        return result

    # Call PDL
    client = PDLClient()
    pdl_result: PDLEnrichResult = await client.enrich_person(
        email=email or None,
        linkedin_url=linkedin_url or None,
        name=name or None,
        company=company or None,
        location=location or None,
    )

    if pdl_result.status == "no_match":
        result["error"] = "No matching person found in PDL"
        logger.info(
            "PDL enrichment: no match for contact %d (%s)",
            contact_id, name,
        )
        return result

    if pdl_result.status == "error":
        result["error"] = pdl_result.error
        logger.warning(
            "PDL enrichment error for contact %d (%s): %s",
            contact_id, name, pdl_result.error,
        )
        return result

    # Success — persist enrichment metadata
    result["success"] = True
    result["match_confidence"] = pdl_result.match_confidence
    result["pdl_person_id"] = pdl_result.person_id

    profile_data["pdl_person_id"] = pdl_result.person_id
    profile_data["pdl_match_confidence"] = pdl_result.match_confidence
    profile_data["enrichment_json"] = pdl_result.raw_response
    profile_data["enriched_at"] = datetime.utcnow().isoformat()

    # Update canonical fields — only if PDL returned a value AND field is currently empty
    fields = pdl_result.fields
    updated: list[str] = []

    if fields.title and not profile_data.get("title"):
        profile_data["title"] = fields.title
        updated.append("title")
    if fields.company and not profile_data.get("company"):
        profile_data["company"] = fields.company
        updated.append("company")
    if fields.location and not profile_data.get("location"):
        profile_data["location"] = fields.location
        updated.append("location")
    if fields.linkedin_url and not profile_data.get("linkedin_url"):
        profile_data["linkedin_url"] = fields.linkedin_url
        updated.append("linkedin_url")
    if fields.name and not contact_name:
        # Only update name if we don't already have one
        updated.append("name")

    # Handle photo — NEVER wipe existing resolved photo
    if fields.photo_url:
        photo_result = await _download_and_store_photo(
            photo_url=fields.photo_url,
            contact_id=contact_id,
            existing_photo_url=profile_data.get("photo_url", ""),
            existing_photo_status=profile_data.get("photo_status", ""),
            existing_photo_source=profile_data.get("photo_source", ""),
        )
        if photo_result["stored"]:
            profile_data["photo_url"] = photo_result["local_url"]
            profile_data["photo_source"] = PhotoSource.ENRICHMENT_PROVIDER
            profile_data["photo_status"] = PhotoStatus.RESOLVED
            profile_data["photo_last_checked_at"] = datetime.utcnow().isoformat()
            result["photo_updated"] = True
            updated.append("photo_url")

    result["fields_updated"] = updated

    logger.info(
        "PDL enrichment succeeded for contact %d (%s): "
        "confidence=%.2f, fields=%s, photo=%s",
        contact_id, name, pdl_result.match_confidence,
        updated, result["photo_updated"],
    )

    return result


async def _download_and_store_photo(
    photo_url: str,
    contact_id: int,
    existing_photo_url: str = "",
    existing_photo_status: str = "",
    existing_photo_source: str = "",
) -> dict:
    """Download a photo and store it locally.

    NEVER replaces an existing RESOLVED photo unless download succeeds.
    Returns {"stored": bool, "local_url": str, "error": str}.
    """
    result = {"stored": False, "local_url": "", "error": ""}

    # Check if we should even try to update
    is_existing_resolved = (
        existing_photo_url
        and existing_photo_status in (PhotoStatus.RESOLVED, "RESOLVED")
        and existing_photo_source not in (
            PhotoSource.GRAVATAR, PhotoSource.COMPANY_LOGO,
            PhotoSource.INITIALS, "",
        )
    )

    if is_existing_resolved:
        # Only replace if we can download successfully
        logger.info(
            "Contact %d has existing RESOLVED photo from %s — will only replace on success",
            contact_id, existing_photo_source,
        )

    if not photo_url or not photo_url.startswith("http"):
        result["error"] = "Invalid photo URL"
        return result

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(photo_url, follow_redirects=True)
            if resp.status_code != 200:
                result["error"] = f"Photo download failed: HTTP {resp.status_code}"
                return result

            content_type = resp.headers.get("content-type", "")
            if not content_type.startswith("image/"):
                result["error"] = f"Not an image: {content_type}"
                return result

            image_bytes = resp.content
            if len(image_bytes) < 100:
                result["error"] = "Image too small (likely broken)"
                return result

            # Store locally
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            url_hash = hashlib.sha256(photo_url.encode()).hexdigest()[:16]
            ext = ".jpg"
            if "png" in content_type:
                ext = ".png"
            elif "webp" in content_type:
                ext = ".webp"

            filename = f"pdl_{contact_id}_{url_hash}{ext}"
            local_path = IMAGE_CACHE_DIR / filename
            local_path.write_bytes(image_bytes)

            result["stored"] = True
            result["local_url"] = f"/api/local-image/{local_path}"
            logger.info(
                "Stored PDL photo for contact %d: %s (%d bytes)",
                contact_id, filename, len(image_bytes),
            )
            return result

    except httpx.TimeoutException:
        result["error"] = "Photo download timed out"
    except Exception as exc:
        result["error"] = f"Photo download failed: {exc}"
        logger.warning("Photo download failed for contact %d: %s", contact_id, exc)

    return result
