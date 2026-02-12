"""Generate and store embeddings for source record chunks.

Embeddings enable semantic retrieval for future briefs.
Uses a simple chunking strategy: split body text into ~500-token chunks.
"""

from __future__ import annotations

import json
import logging

from app.clients.openai_client import EmbeddingClient
from app.config import settings
from app.store.database import EmbeddingRecord, SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)

CHUNK_SIZE_CHARS = 2000  # ~500 tokens


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_CHARS) -> list[str]:
    """Split text into chunks by character count, breaking at sentence boundaries."""
    if not text:
        return []
    sentences = text.replace("\n", " ").split(". ")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current_len + len(sentence) > chunk_size and current:
            chunks.append(". ".join(current) + ".")
            current = []
            current_len = 0
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(". ".join(current))

    return chunks


def embed_source_record(record_id: int) -> int:
    """Generate embeddings for a single source record. Returns count of chunks embedded."""
    init_db()
    session = get_session()
    try:
        record = session.query(SourceRecord).get(record_id)
        if not record:
            logger.warning("Source record %d not found", record_id)
            return 0

        # Check if already embedded
        existing = session.query(EmbeddingRecord).filter_by(source_record_id=record_id).count()
        if existing > 0:
            return existing

        text = record.body or record.summary or ""
        if not text.strip():
            return 0

        chunks = chunk_text(text)
        if not chunks:
            return 0

        try:
            client = EmbeddingClient()
            vectors = client.embed(chunks)
        except Exception:
            logger.exception("Failed to generate embeddings for record %d", record_id)
            return 0

        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            emb = EmbeddingRecord(
                source_record_id=record_id,
                chunk_index=i,
                chunk_text=chunk,
                embedding=json.dumps(vector),
                model=settings.openai_embedding_model,
            )
            session.add(emb)

        session.commit()
        return len(chunks)
    finally:
        session.close()


def embed_all_pending() -> int:
    """Embed all source records that don't have embeddings yet."""
    init_db()
    session = get_session()
    try:
        # Find records without embeddings
        records = session.query(SourceRecord).all()
        total = 0
        for record in records:
            count = session.query(EmbeddingRecord).filter_by(source_record_id=record.id).count()
            if count == 0:
                total += embed_source_record(record.id)
        return total
    finally:
        session.close()
