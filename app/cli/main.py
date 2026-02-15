"""CLI entry point for the Pre-Call Intelligence Briefing Engine.

Usage:
    brief --person "Jane Doe" --company "Acme Corp" --when "2026-02-15 14:00" --topic "Q1 Review"
    brief --person "Ben Titmus" --company "AnswerRocket" --strict
"""

from __future__ import annotations

import logging
import sys

import click
from rich.console import Console
from rich.panel import Panel

from app.brief.pipeline import run_pipeline
from app.config import settings
from app.store.database import init_db

console = Console()


def _setup_logging():
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


@click.command("brief")
@click.option("--person", "-p", default=None, help="Person name to brief on")
@click.option("--company", "-c", default=None, help="Company name to brief on")
@click.option("--when", "-w", "meeting_when", default=None, help="Meeting datetime (YYYY-MM-DD HH:MM)")
@click.option("--topic", "-t", default=None, help="Meeting topic")
@click.option(
    "--skip-ingestion",
    is_flag=True,
    default=False,
    help="Skip API calls; use only stored data",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Enforce quality gates: 95% evidence coverage, identity lock, genericness. Fail loudly.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging")
def cli(
    person: str | None,
    company: str | None,
    meeting_when: str | None,
    topic: str | None,
    skip_ingestion: bool,
    strict: bool,
    verbose: bool,
):
    """Generate a Pre-Call Intelligence Brief."""
    if verbose:
        settings.log_level = "DEBUG"
    _setup_logging()

    if not person and not company:
        console.print("[red]Error: At least --person or --company must be provided.[/red]")
        sys.exit(1)

    mode_label = "[bold red]STRICT[/bold red] " if strict else ""
    dash = "\u2014"
    console.print(
        Panel(
            f"[bold]{mode_label}Pre-Call Intelligence Brief[/bold]\n"
            f"Person: {person or dash}  |  Company: {company or dash}\n"
            f"Topic: {topic or dash}  |  Meeting: {meeting_when or dash}",
            title="Briefing Engine",
            border_style="red" if strict else "blue",
        )
    )

    init_db()

    with console.status("[bold green]Generating brief..."):
        result = run_pipeline(
            person=person,
            company=company,
            topic=topic,
            meeting_when=meeting_when,
            skip_ingestion=skip_ingestion,
            strict=strict,
        )

    # Print summary
    h = result.brief.header
    score = h.confidence_score
    score_color = "green" if score >= 0.5 else "yellow" if score >= 0.2 else "red"

    console.print()
    console.print(f"[bold]Confidence:[/bold] [{score_color}]{score:.0%}[/{score_color}]")
    source_count = len(result.brief.evidence_index) or len(result.brief.appendix_evidence)
    console.print(f"[bold]Sources used:[/bold] {source_count}")

    # Gate results
    if h.gate_status != "not_run":
        gate_color = {
            "passed": "green", "constrained": "yellow", "failed": "red"
        }.get(h.gate_status, "white")
        console.print(
            f"[bold]Gate Status:[/bold] [{gate_color}]{h.gate_status.upper()}[/{gate_color}]"
        )
        console.print(f"  Identity Lock: {h.identity_lock_score:.0f}/100")
        console.print(f"  Evidence Coverage: {h.evidence_coverage_pct:.0f}%")
        console.print(f"  Genericness: {h.genericness_score:.0f}%")

    if strict and h.gate_status == "failed":
        console.print()
        console.print(
            "[bold red]STRICT MODE FAILURE:[/bold red] "
            "Brief does not meet quality gates. Review evidence sources."
        )
    console.print()

    if result.md_path:
        console.print(f"[bold green]Markdown:[/bold green] {result.md_path}")
    if result.json_path:
        console.print(f"[bold green]JSON:[/bold green]     {result.json_path}")

    console.print()

    # Print the brief to terminal
    if result.markdown:
        from rich.markdown import Markdown
        console.print(Markdown(result.markdown))

    # Exit with error code in strict mode if gates failed
    if strict and h.gate_status == "failed":
        sys.exit(2)


def main():
    cli()


if __name__ == "__main__":
    main()
