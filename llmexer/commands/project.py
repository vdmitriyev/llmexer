"""Project group commands of the CLI interface."""

import os

import typer
from rich.table import Table

from llmexer.base.experiment import DIR_EXPERIMENT, generate_project_id
from llmexer.base.project import (
    README_VERSION_PLACEHOLDER,
    SOURCE_GITIGNORE,
    SOURCE_README,
    SortBy,
    format_created,
    has_content,
    project_row,
    read_project_template,
    scan_projects,
)
from llmexer.common import ensure_directory_exists
from llmexer.configs import console, cprint, settings
from llmexer.constants import ANALYSIS_DIR, PAPERS_DIR, PROJECTS_PATH, SEARCHES_DIR
from llmexer.exceptions import LLMExerException, ProjectAlreadyExistsException
from llmexer.version import package_version

app = typer.Typer(help="Manage projects.")


def _print_current_project() -> None:
    """Print the active project, saying so when its folder is not there."""

    project_path = os.path.join(PROJECTS_PATH, settings.project_id)
    if os.path.exists(project_path):
        cprint(f"Current project: [bold yellow]{settings.project_id}[/bold yellow]")
    else:
        cprint(
            f"Current project: [bold yellow]{settings.project_id}[/bold yellow] "
            f"[bold red](not found in {PROJECTS_PATH})[/bold red]"
        )


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Manage projects."""

    # Bare `llmexer project` answers the question it looks like: which project
    # am I working on? The help is still one `--help` away.
    if ctx.invoked_subcommand is not None:
        return

    if settings.project_id:
        _print_current_project()
    else:
        cprint("No default project has been set.")


FILE_GITIGNORE = ".gitignore"
FILE_README = "README.md"


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

    gitignore_path = os.path.join(project_path, FILE_GITIGNORE)
    with open(gitignore_path, "w", encoding="utf-8") as f:
        f.write(read_project_template(SOURCE_GITIGNORE))

    readme = read_project_template(SOURCE_README).replace(README_VERSION_PLACEHOLDER, package_version())
    readme_path = os.path.join(project_path, FILE_README)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    cprint(f"Created project: [bold yellow]{project_id}[/bold yellow]")


# Columns of `project list`, each one a folder a project may or may not hold yet.
PROJECT_PARTS = [
    ("Search", SEARCHES_DIR),
    ("Experiment", DIR_EXPERIMENT),
    ("Analysis", ANALYSIS_DIR),
    ("Papers", PAPERS_DIR),
]


@app.command(name="list")
def list_projects(
    sort_by: SortBy = typer.Option(
        SortBy.alpha,
        "--sort-by",
        help="Sort projects by 'alpha' (alphabetical) or 'date' (creation date).",
    ),
    desc: bool = typer.Option(False, "--desc", help="Sort in descending order."),
) -> None:
    """List all projects under .projects with the parts they hold"""

    entries = scan_projects(PROJECTS_PATH, sort_by, desc)
    if not entries:
        cprint("No projects found.")
        return

    table = Table()
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="cyan", no_wrap=True)
    for label, _ in PROJECT_PARTS:
        table.add_column(label, justify="center", no_wrap=True)

    current_pid = settings.project_id
    for i, entry in enumerate(entries, start=1):
        present = [has_content(entry.path, part_dir) for _, part_dir in PROJECT_PARTS]

        plain_cells = [entry.name, format_created(entry)] + ["YES" if p else "NO" for p in present]
        display_cells = [entry.name, format_created(entry)] + [
            "[green]YES[/green]" if p else "[red]NO[/red]" for p in present
        ]

        is_current = bool(current_pid) and entry.name == current_pid
        table.add_row(*project_row(i, plain_cells, display_cells, is_current))

    console.print(table)


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
            raise LLMExerException("No project ID provided. Use --old-id or set PROJECT_ID in .env file.")

    old_path = os.path.join(PROJECTS_PATH, old_id)
    new_path = os.path.join(PROJECTS_PATH, new_id)

    if not os.path.exists(old_path):
        raise LLMExerException(f"Project '{old_id}' does not exist.")

    if os.path.exists(new_path):
        raise ProjectAlreadyExistsException(f"Project '{new_id}' already exists.")

    os.rename(old_path, new_path)
    cprint(f"Renamed project: [bold yellow]{old_id}[/bold yellow] → [bold yellow]{new_id}[/bold yellow]")


@app.command()
def current() -> None:
    """Display the current project ID loaded from .env"""

    if settings.project_id:
        _print_current_project()
    else:
        cprint("[bold red]No current project set.[/bold red] Set PROJECT_ID in .env file.")
