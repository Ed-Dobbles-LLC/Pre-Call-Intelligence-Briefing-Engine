"""Action item extraction and management service.

Extracts action items from:
1. Fireflies transcript action_items field
2. Gmail email body text (pattern matching + optional LLM)

Supports priority inference based on urgency signals.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timedelta

from app.config import settings
from app.store.database import (
    ActionItemRecord,
    EntityRecord,
    SourceRecord,
    get_session,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Priority inference patterns
# ---------------------------------------------------------------------------

_CRITICAL_PATTERNS = [
    re.compile(r"\b(urgent|asap|immediately|critical|emergency|blocking|blocker)\b", re.I),
    re.compile(r"\b(today|by\s+eod|end\s+of\s+day|right\s+away)\b", re.I),
]

_HIGH_PATTERNS = [
    re.compile(r"\b(important|priority|deadline|must|required|essential)\b", re.I),
    re.compile(r"\b(this\s+week|by\s+friday|by\s+monday|tomorrow)\b", re.I),
    re.compile(r"\b(follow.?up\s+immediately|time.?sensitive)\b", re.I),
]

_LOW_PATTERNS = [
    re.compile(r"\b(when\s+you\s+get\s+a\s+chance|no\s+rush|low\s+priority)\b", re.I),
    re.compile(r"\b(eventually|someday|nice\s+to\s+have|optional|fyi)\b", re.I),
    re.compile(r"\b(whenever|backlog|parking\s+lot)\b", re.I),
]


def infer_priority(text: str) -> str:
    """Infer action item priority from text signals.

    Returns: 'critical', 'high', 'medium', or 'low'.
    """
    if not text:
        return "medium"

    if any(p.search(text) for p in _CRITICAL_PATTERNS):
        return "critical"
    if any(p.search(text) for p in _HIGH_PATTERNS):
        return "high"
    if any(p.search(text) for p in _LOW_PATTERNS):
        return "low"
    return "medium"


# ---------------------------------------------------------------------------
# Email action item extraction (pattern-based)
# ---------------------------------------------------------------------------

_ACTION_ITEM_PATTERNS = [
    # Explicit markers
    re.compile(r"(?:action\s+item|todo|to.do|task)[:\s]+(.+?)(?:\n|$)", re.I),
    # "Please [verb]..." patterns
    re.compile(r"(?:please|pls|kindly)\s+(send|review|prepare|schedule|follow|update|share|confirm|check|create|set\s+up|draft|submit|complete|finalize)[\s:]+(.+?)(?:\.|;|\n|$)", re.I),
    # "Can you [verb]..." patterns
    re.compile(r"(?:can|could|would)\s+you\s+(?:please\s+)?(send|review|prepare|schedule|follow|update|share|confirm|check|create|set\s+up|draft|submit)[\s:]+(.+?)(?:\?|\.|;|\n|$)", re.I),
    # "I need you to..." patterns
    re.compile(r"(?:i\s+need|we\s+need)\s+(?:you\s+)?to\s+(.+?)(?:\.|;|\n|$)", re.I),
    # "Next steps:" section
    re.compile(r"next\s+steps?[:\s]+(.+?)(?:\n\n|\Z)", re.I | re.S),
]


def extract_action_items_from_email(
    body: str,
    subject: str = "",
) -> list[dict]:
    """Extract action items from email body text.

    Returns list of {"title": str, "priority": str}.
    """
    if not body:
        return []

    items = []
    seen_titles = set()

    for pattern in _ACTION_ITEM_PATTERNS:
        for match in pattern.finditer(body):
            # Get the full match or the most relevant group
            groups = match.groups()
            if len(groups) >= 2:
                title = f"{groups[0]} {groups[1]}".strip()
            else:
                title = groups[0].strip() if groups else match.group(0).strip()

            # Clean up
            title = re.sub(r"\s+", " ", title)
            title = title.strip("- •·")

            # Skip duplicates and too-short items
            if len(title) < 10 or title.lower() in seen_titles:
                continue
            if len(title) > 200:
                title = title[:200] + "..."

            seen_titles.add(title.lower())
            items.append({
                "title": title,
                "priority": infer_priority(f"{subject} {title}"),
            })

    return items[:10]  # Cap at 10 per email


def extract_action_items_from_transcript(
    action_items_raw: list[str],
    title: str = "",
) -> list[dict]:
    """Extract and normalize action items from Fireflies transcript action_items.

    The raw items come from the Fireflies API's action_items field.
    Returns list of {"title": str, "priority": str}.
    """
    items = []
    seen_titles = set()

    for raw in action_items_raw:
        if not raw or not raw.strip():
            continue

        title_text = raw.strip()
        # Clean common prefixes
        title_text = re.sub(r"^[-•·\d.)\]]+\s*", "", title_text)
        title_text = re.sub(r"\s+", " ", title_text).strip()

        if len(title_text) < 5 or title_text.lower() in seen_titles:
            continue
        if len(title_text) > 300:
            title_text = title_text[:300] + "..."

        seen_titles.add(title_text.lower())
        items.append({
            "title": title_text,
            "priority": infer_priority(f"{title} {title_text}"),
        })

    return items


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def persist_action_items(
    items: list[dict],
    source_type: str,
    source_id: str,
    source_record_id: int | None = None,
    entity_id: int | None = None,
    project_id: int | None = None,
) -> list[ActionItemRecord]:
    """Persist extracted action items to the database.

    Deduplicates by (source_id, title) to avoid re-extracting on re-sync.
    Returns list of created ActionItemRecords.
    """
    if not items:
        return []

    session = get_session()
    created = []
    try:
        # Get existing items for this source to dedup
        existing_titles = set()
        if source_id:
            existing = session.query(ActionItemRecord.title).filter(
                ActionItemRecord.source_id == source_id,
            ).all()
            existing_titles = {r.title.lower() for r in existing}

        for item in items:
            title = item["title"]
            if title.lower() in existing_titles:
                continue

            record = ActionItemRecord(
                title=title,
                description=item.get("description"),
                source_type=source_type,
                source_id=source_id,
                source_record_id=source_record_id,
                entity_id=entity_id,
                project_id=project_id,
                priority=item.get("priority", "medium"),
                status="open",
            )
            session.add(record)
            created.append(record)
            existing_titles.add(title.lower())

        session.commit()
        if created:
            logger.info(
                "Persisted %d action items from %s:%s",
                len(created), source_type, source_id,
            )
        return created
    except Exception:
        session.rollback()
        logger.exception("Failed to persist action items")
        return []
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Dashboard stats
# ---------------------------------------------------------------------------

def get_action_item_stats() -> dict:
    """Get action item statistics for the dashboard."""
    session = get_session()
    try:
        total = session.query(ActionItemRecord).count()
        open_count = session.query(ActionItemRecord).filter(
            ActionItemRecord.status == "open",
        ).count()
        in_progress = session.query(ActionItemRecord).filter(
            ActionItemRecord.status == "in_progress",
        ).count()
        done = session.query(ActionItemRecord).filter(
            ActionItemRecord.status == "done",
        ).count()

        # Priority breakdown for open items
        critical = session.query(ActionItemRecord).filter(
            ActionItemRecord.status.in_(["open", "in_progress"]),
            ActionItemRecord.priority == "critical",
        ).count()
        high = session.query(ActionItemRecord).filter(
            ActionItemRecord.status.in_(["open", "in_progress"]),
            ActionItemRecord.priority == "high",
        ).count()

        return {
            "total": total,
            "open": open_count,
            "in_progress": in_progress,
            "done": done,
            "critical": critical,
            "high": high,
        }
    except Exception:
        logger.exception("Failed to get action item stats")
        return {"total": 0, "open": 0, "in_progress": 0, "done": 0, "critical": 0, "high": 0}
    finally:
        session.close()
