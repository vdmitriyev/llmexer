"""Project group commands of the CLI interface."""

import os

import typer

from llmexer.base.experiment import generate_project_id
from llmexer.common import ensure_directory_exists
from llmexer.configs import cprint, settings
from llmexer.constants import PROJECTS_PATH
from llmexer.exceptions import LLMExerException, ProjectAlreadyExistsException

app = typer.Typer(help="Manage projects.")


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
