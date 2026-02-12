"""CLI entry point for the Pre-Call Intelligence Briefing Engine.

Usage:
    brief --person "Jane Doe" --company "Acme Corp" --when "2026-02-15 14:00" --topic "Q1 Review"
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
@click.option("--verbose", "-v", is_flag=True, default=False, help="Verbose logging")
def cli(
    person: str | None,
    company: str | None,
    meeting_when: str | None,
    topic: str | None,
    skip_ingestion: bool,
    verbose: bool,
):
    """Generate a Pre-Call Intelligence Brief."""
    if verbose:
        settings.log_level = "DEBUG"
    _setup_logging()

    if not person and not company:
        console.print("[red]Error: At least --person or --company must be provided.[/red]")
        sys.exit(1)

    console.print(
        Panel(
            f"[bold]Pre-Call Intelligence Brief[/bold]\n"
            f"Person: {person or '—'}  |  Company: {company or '—'}\n"
            f"Topic: {topic or '—'}  |  Meeting: {meeting_when or '—'}",
            title="Briefing Engine",
            border_style="blue",
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
        )

    # Print summary
    score = result.brief.header.confidence_score
    score_color = "green" if score >= 0.5 else "yellow" if score >= 0.2 else "red"

    console.print()
    console.print(f"[bold]Confidence:[/bold] [{score_color}]{score:.0%}[/{score_color}]")
    console.print(f"[bold]Sources used:[/bold] {len(result.brief.appendix_evidence)}")
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


def main():
    cli()


if __name__ == "__main__":
    main()
