"""Project classifier service.

Classifies interactions into project types using:
1. LLM classifier (GPT-4o-mini) for nuanced classification
2. Rule-based fallback when LLM is unavailable or for speed

Project types: job_interview, sales_deal, partnership, internal, other
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime

from app.config import settings
from app.store.database import (
    EntityRecord,
    ProjectRecord,
    PROJECT_STAGE_PIPELINES,
    get_session,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based classifier patterns
# ---------------------------------------------------------------------------

_JOB_INTERVIEW_PATTERNS = [
    r"\b(interview|hiring|recruiter|recruiting|job\s+opening|career|resume|cv)\b",
    r"\b(talent\s+acquisition|headhunter|offer\s+letter|compensation|salary)\b",
    r"\b(screening|phone\s+screen|panel\s+interview|final\s+round|onsite)\b",
    r"\b(candidate|applicant|job\s+search|position|role\s+at)\b",
]

_SALES_DEAL_PATTERNS = [
    r"\b(proposal|pricing|contract|deal|pipeline|close|rfp|rfq)\b",
    r"\b(demo|pilot|poc|proof\s+of\s+concept|trial)\b",
    r"\b(quota|revenue|arrs?|mrr|upsell|cross.?sell)\b",
    r"\b(procurement|vendor|buyer|purchase\s+order|invoice)\b",
    r"\b(decision\s+maker|stakeholder|champion|budget)\b",
]

_PARTNERSHIP_PATTERNS = [
    r"\b(partnership|partner|alliance|collaboration|co-brand)\b",
    r"\b(joint\s+venture|strategic\s+alliance|integration|api\s+partner)\b",
    r"\b(channel\s+partner|reseller|distribution|affiliate)\b",
    r"\b(mou|memorandum|term\s+sheet|co-marketing)\b",
]

_INTERNAL_PATTERNS = [
    r"\b(standup|sprint|retro|retrospective|planning|all.?hands)\b",
    r"\b(1.on.1|one.on.one|team\s+meeting|staff\s+meeting)\b",
    r"\b(okr|kpi|quarterly\s+review|performance\s+review)\b",
    r"\b(roadmap|backlog|architecture|tech\s+debt|migration)\b",
]


def _compile_patterns(patterns: list[str]) -> list[re.Pattern]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_JOB_COMPILED = _compile_patterns(_JOB_INTERVIEW_PATTERNS)
_SALES_COMPILED = _compile_patterns(_SALES_DEAL_PATTERNS)
_PARTNERSHIP_COMPILED = _compile_patterns(_PARTNERSHIP_PATTERNS)
_INTERNAL_COMPILED = _compile_patterns(_INTERNAL_PATTERNS)


def classify_rule_based(text: str) -> tuple[str, float]:
    """Classify text into a project type using regex rules.

    Returns (project_type, confidence) where confidence is 0.0–1.0.
    """
    if not text:
        return "other", 0.0

    scores = {
        "job_interview": sum(1 for p in _JOB_COMPILED if p.search(text)),
        "sales_deal": sum(1 for p in _SALES_COMPILED if p.search(text)),
        "partnership": sum(1 for p in _PARTNERSHIP_COMPILED if p.search(text)),
        "internal": sum(1 for p in _INTERNAL_COMPILED if p.search(text)),
    }

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score == 0:
        return "other", 0.0

    total = sum(scores.values())
    confidence = best_score / total if total > 0 else 0.0
    # Scale confidence: 1 match = 0.3, 2 = 0.5, 3+ = 0.7
    if best_score == 1:
        confidence = min(confidence, 0.3)
    elif best_score == 2:
        confidence = min(confidence, 0.5)
    else:
        confidence = min(confidence, 0.7)

    return best_type, round(confidence, 2)


async def classify_llm(text: str) -> tuple[str, float]:
    """Classify text into a project type using GPT-4o-mini.

    Returns (project_type, confidence).
    Falls back to rule-based if OpenAI is unavailable.
    """
    if not settings.openai_api_key:
        return classify_rule_based(text)

    try:
        import openai

        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=100,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the following meeting/interaction context into exactly one "
                        "project type. Respond with JSON only: {\"type\": \"...\", \"confidence\": 0.X}\n\n"
                        "Valid types:\n"
                        "- job_interview: recruiting, hiring, career discussions\n"
                        "- sales_deal: sales pipeline, demos, proposals, contracts\n"
                        "- partnership: strategic alliances, integrations, channel partners\n"
                        "- internal: team meetings, standups, planning, reviews\n"
                        "- other: anything that doesn't fit above"
                    ),
                },
                {"role": "user", "content": text[:2000]},  # Truncate for token limits
            ],
        )

        content = response.choices[0].message.content.strip()
        # Parse JSON response
        result = json.loads(content)
        project_type = result.get("type", "other")
        confidence = float(result.get("confidence", 0.5))

        # Validate type
        if project_type not in PROJECT_STAGE_PIPELINES:
            project_type = "other"

        return project_type, round(confidence, 2)

    except Exception as e:
        logger.warning("LLM classifier failed, falling back to rules: %s", e)
        return classify_rule_based(text)


def classify_interaction(
    title: str = "",
    summary: str = "",
    participants: list[str] | None = None,
    use_llm: bool = True,
) -> tuple[str, float, str]:
    """Classify an interaction and return (type, confidence, source).

    Combines title, summary, and participants into context text,
    then runs LLM (if enabled) with rule-based fallback.

    Returns (project_type, confidence, classifier_source).
    """
    parts = []
    if title:
        parts.append(f"Title: {title}")
    if summary:
        parts.append(f"Summary: {summary}")
    if participants:
        parts.append(f"Participants: {', '.join(participants)}")
    text = "\n".join(parts)

    if not text.strip():
        return "other", 0.0, "rule"

    # Try rule-based first for speed
    rule_type, rule_conf = classify_rule_based(text)

    # If rule-based is confident enough (>=0.5), use it
    if rule_conf >= 0.5:
        return rule_type, rule_conf, "rule"

    # If LLM requested and available, try LLM
    if use_llm and settings.openai_api_key:
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, need to handle differently
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    llm_type, llm_conf = pool.submit(
                        lambda: asyncio.run(classify_llm(text))
                    ).result(timeout=10)
            else:
                llm_type, llm_conf = asyncio.run(classify_llm(text))
            return llm_type, llm_conf, "llm"
        except Exception as e:
            logger.debug("LLM classification failed: %s", e)

    # Fall back to rule-based result
    return rule_type, rule_conf, "rule"


def find_or_create_project(
    name: str,
    project_type: str,
    entity_id: int | None = None,
    source_id: str | None = None,
    metadata: dict | None = None,
    classifier_source: str = "rule",
    classifier_confidence: float = 0.0,
) -> ProjectRecord:
    """Find an existing project by name+type or create a new one.

    Deduplicates by normalized name + type. Adds entity_id and source_id
    to existing projects if not already present.
    """
    session = get_session()
    try:
        # Try to find existing project with same name and type
        existing = session.query(ProjectRecord).filter(
            ProjectRecord.name == name,
            ProjectRecord.project_type == project_type,
        ).first()

        if existing:
            # Add entity and source IDs if new
            if entity_id:
                existing.add_entity_id(entity_id)
            if source_id:
                src_ids = json.loads(existing.source_ids or "[]")
                if source_id not in src_ids:
                    src_ids.append(source_id)
                    existing.source_ids = json.dumps(src_ids)
            session.commit()
            return existing

        # Create new project
        project = ProjectRecord(
            name=name,
            project_type=project_type,
            stage=PROJECT_STAGE_PIPELINES.get(project_type, ["identified"])[0],
            entity_ids=json.dumps([entity_id] if entity_id else []),
            source_ids=json.dumps([source_id] if source_id else []),
            metadata_json=json.dumps(metadata or {}),
            classifier_source=classifier_source,
            classifier_confidence=classifier_confidence,
        )
        session.add(project)
        session.commit()
        logger.info(
            "Created project: %s (type=%s, source=%s, conf=%.2f)",
            name, project_type, classifier_source, classifier_confidence,
        )
        return project

    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def advance_project_stage(project_id: int, new_stage: str) -> ProjectRecord | None:
    """Advance a project to a new stage within its type's pipeline."""
    session = get_session()
    try:
        project = session.query(ProjectRecord).get(project_id)
        if not project:
            return None

        pipeline = PROJECT_STAGE_PIPELINES.get(project.project_type, [])
        if new_stage not in pipeline:
            logger.warning(
                "Invalid stage '%s' for project type '%s'",
                new_stage, project.project_type,
            )
            return None

        project.stage = new_stage
        project.updated_at = datetime.utcnow()
        session.commit()
        return project
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
