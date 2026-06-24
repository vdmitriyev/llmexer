"""Project group commands of the CLI interface."""

import os
from datetime import datetime, timezone
from enum import Enum

import typer
from rich.table import Table

from llmexer.base.experiment import (
    _get_generated_experiment_files,
    _is_experiment_initialized,
    generate_project_id,
)
from llmexer.common import ensure_directory_exists
from llmexer.configs import console, cprint, settings
from llmexer.constants import PROJECTS_PATH
from llmexer.exceptions import LLMExerException, ProjectAlreadyExistsException

app = typer.Typer(help="Manage projects.")


class SortBy(str, Enum):
    alpha = "alpha"
    date = "date"


@app.command()
def create(
    id: str = typer.Option(
        None,
        "--id",
        help="Custom project ID. If not provided, one is auto-generated.",
    )
) -> None:
    """Create a new project folder under .projects"""
    project_id = id if id else generate_project_id()
    project_path = os.path.join(PROJECTS_PATH, project_id)

    if os.path.exists(project_path):
        raise ProjectAlreadyExistsException(f"Project '{project_id}' already exists.")

    ensure_directory_exists(project_path)
    cprint(f"Created project: [bold yellow]{project_id}[/bold yellow]")


@app.command(name="list")
def list_projects(
    sort_by: SortBy = typer.Option(
        SortBy.alpha,
        "--sort-by",
        help="Sort projects by 'alpha' (alphabetical) or 'date' (creation date).",
    ),
    desc: bool = typer.Option(False, "--desc", help="Sort in descending order."),
) -> None:
    """List all projects in the projects folder"""
    if not os.path.exists(PROJECTS_PATH):
        cprint("No projects found.")
        return

    entries = [e for e in os.scandir(PROJECTS_PATH) if e.is_dir()]

    if not entries:
        cprint("No projects found.")
        return

    if sort_by == SortBy.date:
        entries.sort(key=lambda e: e.stat().st_ctime, reverse=desc)
    else:
        entries.sort(key=lambda e: e.name, reverse=desc)

    table = Table()
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="cyan", no_wrap=True)
    table.add_column("Initialized", justify="center", style="red", no_wrap=True)
    table.add_column("Experiments", style="white")

    current_pid = settings.project_id
    experiment_file_to_run = ""
    for i, entry in enumerate(entries, start=1):
        ctime = datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        is_initialized = _is_experiment_initialized(entry.path)
        init_display = "[green]Yes[/green]" if is_initialized else "[dim]No[/dim]"

        generated_files = _get_generated_experiment_files(entry.path)
        files_display = (
            "\n".join(generated_files) if generated_files else "[dim]-[/dim]"
        )
        files_display_plain = "\n".join(generated_files) if generated_files else "-"

        # Check if this is the current project
        is_current = current_pid and entry.name == current_pid

        if is_current:
            # Bold yellow for current project, underline only on counter
            table.add_row(
                f"[bold underline yellow]{i}[/bold underline yellow]",
                f"[bold yellow]{entry.name}[/bold yellow]",
                f"[bold yellow]{ctime}[/bold yellow]",
                f"[bold yellow]{'Yes' if is_initialized else 'No'}[/bold yellow]",
                f"[bold yellow]{files_display_plain}[/bold yellow]",
            )
            if len(generated_files) > 0:
                experiment_file_to_run = generated_files[-1]
        else:
            table.add_row(str(i), entry.name, ctime, init_display, files_display)

    console.print(table)
    cprint("\nExample to run an experiment:")
    cprint(
        f"[bold yellow]llmexer experiment run --file {experiment_file_to_run}[/bold yellow]"
    )


@app.command()
def rename(
    old_id: str = typer.Option(
        None,
        "--old-id",
        help="Current project ID to rename. If not provided, uses PROJECT_ID from .env.",
    ),
    new_id: str = typer.Option(
        ...,
        "--new-id",
        help="New project ID name.",
    ),
) -> None:
    """Rename an existing project"""

    # Use current project if old_id not provided
    if old_id is None:
        if settings.project_id:
            old_id = settings.project_id
        else:
            raise LLMExerException(
                "No project ID provided. Use --old-id or set PROJECT_ID in .env file."
            )

    old_path = os.path.join(PROJECTS_PATH, old_id)
    new_path = os.path.join(PROJECTS_PATH, new_id)

    if not os.path.exists(old_path):
        raise LLMExerException(f"Project '{old_id}' does not exist.")

    if os.path.exists(new_path):
        raise ProjectAlreadyExistsException(f"Project '{new_id}' already exists.")

    os.rename(old_path, new_path)
    cprint(
        f"Renamed project: [bold yellow]{old_id}[/bold yellow] → [bold yellow]{new_id}[/bold yellow]"
    )


@app.command()
def current() -> None:
    """Display the current project ID loaded from .env"""

    if settings.project_id:
        project_path = os.path.join(PROJECTS_PATH, settings.project_id)
        if os.path.exists(project_path):
            cprint(f"Current project: [bold yellow]{settings.project_id}[/bold yellow]")
        else:
            cprint(
                f"Current project: [bold yellow]{settings.project_id}[/bold yellow] "
                f"[bold red](not found in {PROJECTS_PATH})[/bold red]"
            )
    else:
        cprint(
            "[bold red]No current project set.[/bold red] Set PROJECT_ID in .env file."
        )
