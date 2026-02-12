"""Entity resolution: map a person/company input to known identifiers.

Maps a "person" input to emails/aliases and Fireflies participant names.
Supports "company" search across domains and signature lines.
"""

from __future__ import annotations

import json
import logging
import re

from app.store.database import EntityRecord, SourceRecord, get_session, init_db

logger = logging.getLogger(__name__)


class ResolvedEntity:
    """Result of entity resolution."""

    def __init__(
        self,
        entity_id: int | None = None,
        name: str = "",
        entity_type: str = "person",
        emails: list[str] | None = None,
        aliases: list[str] | None = None,
        domains: list[str] | None = None,
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type
        self.emails = emails or []
        self.aliases = aliases or []
        self.domains = domains or []


def _extract_email_from_header(header_value: str) -> str | None:
    """Extract email address from a header like 'John Doe <john@example.com>'."""
    match = re.search(r"<([^>]+@[^>]+)>", header_value)
    if match:
        return match.group(1).lower()
    if "@" in header_value:
        return header_value.strip().lower()
    return None


def _extract_domain(email: str) -> str | None:
    """Extract domain from email address."""
    if "@" in email:
        return email.split("@")[1].lower()
    return None


def resolve_person(name: str, email: str | None = None) -> ResolvedEntity:
    """Resolve a person input to a known entity or create one.

    Strategy:
    1. Look up by exact email match in entities table
    2. Look up by name match (case-insensitive) in entities table
    3. Scan source_records for participant matches to discover aliases/emails
    4. Create a new entity if not found
    """
    init_db()
    session = get_session()
    try:
        # 1. Try exact email match
        if email:
            for entity in session.query(EntityRecord).filter(
                EntityRecord.entity_type == "person"
            ).all():
                stored_emails = entity.get_emails()
                if email.lower() in [e.lower() for e in stored_emails]:
                    return ResolvedEntity(
                        entity_id=entity.id,
                        name=entity.name,
                        entity_type="person",
                        emails=stored_emails,
                        aliases=entity.get_aliases(),
                    )

        # 2. Try name match
        entity = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "person",
            EntityRecord.name.ilike(f"%{name}%"),
        ).first()
        if entity:
            return ResolvedEntity(
                entity_id=entity.id,
                name=entity.name,
                entity_type="person",
                emails=entity.get_emails(),
                aliases=entity.get_aliases(),
            )

        # 3. Scan source records for discoveries
        discovered_emails: set[str] = set()
        discovered_aliases: set[str] = {name.lower()}

        if email:
            discovered_emails.add(email.lower())

        # Look through stored participants
        for record in session.query(SourceRecord).all():
            participants = json.loads(record.participants) if record.participants else []
            for p in participants:
                p_lower = p.lower() if isinstance(p, str) else ""
                if name.lower() in p_lower or (email and email.lower() in p_lower):
                    extracted = _extract_email_from_header(p)
                    if extracted:
                        discovered_emails.add(extracted)
                    else:
                        discovered_aliases.add(p_lower)

        # 4. Create new entity
        new_entity = EntityRecord(
            name=name,
            entity_type="person",
        )
        new_entity.set_emails(sorted(discovered_emails))
        new_entity.set_aliases(sorted(discovered_aliases))
        session.add(new_entity)
        session.commit()
        session.refresh(new_entity)

        logger.info(
            "Created entity '%s' (id=%d) with emails=%s, aliases=%s",
            name, new_entity.id, discovered_emails, discovered_aliases,
        )

        return ResolvedEntity(
            entity_id=new_entity.id,
            name=name,
            entity_type="person",
            emails=sorted(discovered_emails),
            aliases=sorted(discovered_aliases),
        )
    finally:
        session.close()


def resolve_company(company_name: str, domain: str | None = None) -> ResolvedEntity:
    """Resolve a company input to a known entity or create one.

    Strategy:
    1. Look up by domain in entities table
    2. Look up by company name in entities table
    3. Scan source_records for domain / signature matches
    4. Create a new entity if not found
    """
    init_db()
    session = get_session()
    try:
        # 1. Domain match
        if domain:
            for entity in session.query(EntityRecord).filter(
                EntityRecord.entity_type == "company"
            ).all():
                stored_domains = entity.get_domains()
                if domain.lower() in [d.lower() for d in stored_domains]:
                    return ResolvedEntity(
                        entity_id=entity.id,
                        name=entity.name,
                        entity_type="company",
                        domains=stored_domains,
                        aliases=entity.get_aliases(),
                    )

        # 2. Name match
        entity = session.query(EntityRecord).filter(
            EntityRecord.entity_type == "company",
            EntityRecord.name.ilike(f"%{company_name}%"),
        ).first()
        if entity:
            return ResolvedEntity(
                entity_id=entity.id,
                name=entity.name,
                entity_type="company",
                domains=entity.get_domains(),
                aliases=entity.get_aliases(),
            )

        # 3. Discover domains from source records
        discovered_domains: set[str] = set()
        if domain:
            discovered_domains.add(domain.lower())

        for record in session.query(SourceRecord).all():
            participants = json.loads(record.participants) if record.participants else []
            for p in participants:
                extracted_email = _extract_email_from_header(str(p))
                if extracted_email:
                    d = _extract_domain(extracted_email)
                    if d and company_name.lower() in d:
                        discovered_domains.add(d)
            # Also scan body for company name in signatures
            body = record.body or ""
            if company_name.lower() in body.lower():
                # Try to find domain in the same body
                email_matches = re.findall(r"[\w.+-]+@[\w-]+\.[\w.]+", body)
                for em in email_matches:
                    d = _extract_domain(em)
                    if d and company_name.lower() in d:
                        discovered_domains.add(d)

        # 4. Create new entity
        new_entity = EntityRecord(
            name=company_name,
            entity_type="company",
        )
        new_entity.set_domains(sorted(discovered_domains))
        new_entity.set_aliases([company_name.lower()])
        session.add(new_entity)
        session.commit()
        session.refresh(new_entity)

        return ResolvedEntity(
            entity_id=new_entity.id,
            name=company_name,
            entity_type="company",
            domains=sorted(discovered_domains),
        )
    finally:
        session.close()
