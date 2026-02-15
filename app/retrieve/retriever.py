"""Retrieval module: query stored artifacts for brief generation.

For a given person/company, retrieves:
- Last meeting summary
- Last 90 days of interactions
- Open action items / promises
- Objection / concern snippets

All results carry source provenance for citation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta

from sqlalchemy import desc

from app.config import settings
from app.store.database import EmbeddingRecord, SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)


class RetrievedEvidence:
    """Container for all evidence retrieved for brief generation."""

    def __init__(self):
        self.interactions: list[dict] = []
        self.last_interaction: dict | None = None
        self.action_items: list[dict] = []
        self.concern_snippets: list[dict] = []
        self.all_source_records: list[SourceRecord] = []

    @property
    def has_data(self) -> bool:
        return bool(self.interactions)

    @property
    def source_count(self) -> int:
        return len(self.all_source_records)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _semantic_search(
    session,
    query: str,
    since: datetime,
    exclude_ids: set[int],
    top_k: int = 10,
    threshold: float = 0.3,
) -> list[SourceRecord]:
    """Find source records semantically similar to the query string."""
    if not settings.openai_api_key or not query.strip():
        return []

    try:
        from app.clients.openai_client import EmbeddingClient
        client = EmbeddingClient()
        query_vec = client.embed_single(query)
    except Exception:
        logger.debug("Semantic search skipped – embedding failed")
        return []

    if not query_vec:
        return []

    embeddings = (
        session.query(EmbeddingRecord)
        .join(SourceRecord)
        .filter(SourceRecord.date >= since)
        .all()
    )

    scored: list[tuple[float, int]] = []
    for emb in embeddings:
        if emb.source_record_id in exclude_ids:
            continue
        try:
            vec = json.loads(emb.embedding)
        except (json.JSONDecodeError, TypeError):
            continue
        sim = _cosine_similarity(query_vec, vec)
        if sim >= threshold:
            scored.append((sim, emb.source_record_id))

    # Dedupe by source_record_id, keep highest score
    best: dict[int, float] = {}
    for sim, rid in scored:
        if rid not in best or sim > best[rid]:
            best[rid] = sim
    ranked = sorted(best.items(), key=lambda x: x[1], reverse=True)[:top_k]

    record_ids = [rid for rid, _ in ranked]
    if not record_ids:
        return []

    records = session.query(SourceRecord).filter(SourceRecord.id.in_(record_ids)).all()
    logger.info("Semantic search found %d additional records", len(records))
    return records


def retrieve_for_entity(
    entity_id: int | None = None,
    person_name: str | None = None,
    company_name: str | None = None,
    emails: list[str] | None = None,
    aliases: list[str] | None = None,
    domains: list[str] | None = None,
    window_days: int | None = None,
) -> RetrievedEvidence:
    """Retrieve all relevant evidence for an entity.

    Uses multiple matching strategies:
    1. entity_id direct match
    2. Participant name/email matching in source_records
    3. Body text search for company domains
    """
    init_db()
    session = get_session()
    evidence = RetrievedEvidence()
    window = window_days or settings.retrieval_window_days
    since = datetime.utcnow() - timedelta(days=window)

    try:
        # Build candidate records
        candidates: list[SourceRecord] = []

        # Strategy 1: Direct entity_id match
        if entity_id:
            records = (
                session.query(SourceRecord)
                .filter(SourceRecord.entity_id == entity_id)
                .filter(SourceRecord.date >= since)
                .order_by(desc(SourceRecord.date))
                .all()
            )
            candidates.extend(records)

        # Strategy 2: Participant matching
        search_terms = set()
        if person_name:
            search_terms.add(person_name.lower())
        if emails:
            search_terms.update(e.lower() for e in emails)
        if aliases:
            search_terms.update(a.lower() for a in aliases)
        if domains:
            search_terms.update(d.lower() for d in domains)
        if company_name:
            search_terms.add(company_name.lower())

        if search_terms:
            all_records = (
                session.query(SourceRecord)
                .filter(SourceRecord.date >= since)
                .order_by(desc(SourceRecord.date))
                .all()
            )
            seen_ids = {r.id for r in candidates}
            for record in all_records:
                if record.id in seen_ids:
                    continue
                # Check participants
                participants = json.loads(record.participants) if record.participants else []
                participant_text = " ".join(str(p).lower() for p in participants)
                body_text = (record.body or "").lower()
                title_text = (record.title or "").lower()
                searchable = f"{participant_text} {title_text}"

                for term in search_terms:
                    if term in searchable or term in body_text:
                        candidates.append(record)
                        seen_ids.add(record.id)
                        break

        # Strategy 3: Semantic search (boost with embeddings)
        query_parts = [p for p in [person_name, company_name] if p]
        if query_parts:
            seen_ids = {r.id for r in candidates}
            sem_records = _semantic_search(
                session,
                query=" ".join(query_parts),
                since=since,
                exclude_ids=seen_ids,
            )
            candidates.extend(sem_records)

        # Sort by date descending
        candidates.sort(key=lambda r: r.date or datetime.min, reverse=True)
        evidence.all_source_records = candidates

        # Build interactions list
        for record in candidates:
            action_items_raw = json.loads(record.action_items) if record.action_items else []

            interaction = {
                "source_type": record.source_type,
                "source_id": record.source_id,
                "title": record.title,
                "date": record.date.isoformat() if record.date else None,
                "summary": record.summary,
                "participants": json.loads(record.participants) if record.participants else [],
                "action_items": action_items_raw,
                "body_preview": (record.body or "")[:1000],
                "db_id": record.id,
            }
            evidence.interactions.append(interaction)

            # Collect action items
            for item in action_items_raw:
                evidence.action_items.append({
                    "description": item,
                    "source_type": record.source_type,
                    "source_id": record.source_id,
                    "date": record.date.isoformat() if record.date else None,
                })

            # Extract concern/objection snippets from body
            body = record.body or ""
            concern_keywords = [
                "concern", "worried", "risk", "issue", "problem",
                "objection", "pushback", "blocker", "hesitant",
                "not sure", "disagree", "budget", "timeline",
            ]
            body_lower = body.lower()
            for keyword in concern_keywords:
                idx = body_lower.find(keyword)
                if idx >= 0:
                    # Extract surrounding context (~200 chars)
                    start = max(0, idx - 100)
                    end = min(len(body), idx + 100)
                    snippet = body[start:end].strip()
                    evidence.concern_snippets.append({
                        "keyword": keyword,
                        "snippet": snippet,
                        "source_type": record.source_type,
                        "source_id": record.source_id,
                        "date": record.date.isoformat() if record.date else None,
                    })

        # Set last interaction
        if evidence.interactions:
            evidence.last_interaction = evidence.interactions[0]

        logger.info(
            "Retrieved %d interactions, %d action items, %d concern snippets",
            len(evidence.interactions),
            len(evidence.action_items),
            len(evidence.concern_snippets),
        )

        return evidence
    finally:
        session.close()
