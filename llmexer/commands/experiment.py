"""Experiment group commands."""

import os
import uuid
from datetime import datetime, timezone
from enum import Enum

import typer
from rich.table import Table

from llmexer.common import ensure_directory_exists
from llmexer.configs import console, settings
from llmexer.constants import EXPERIMENTS_PATH
from llmexer.exceptions import ExperimentAlreadyExistsException, LLMExerException

app = typer.Typer(help="Manage LLM experiments.")


class SortBy(str, Enum):
    alpha = "alpha"
    date = "date"


def generate_experiment_id() -> str:
    """
    Generate a unique experiment ID formatted as 'YYYYMMDD-GUID'

    Returns:
      str: A string in the format 'YYYYMMDD-UUID'.
    """
    from datetime import date, datetime, timedelta, timezone

    now_utc = datetime.now(timezone.utc)
    formatted_datetime = now_utc.strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8]
    return f"{formatted_datetime}-{unique_id}"


@app.command()
def create(
    id: str = typer.Option(
        None,
        "--id",
        help="Custom experiment ID. If not provided, one is auto-generated.",
    )
) -> None:
    """Create a new experiment folder under .experiments."""
    experiment_id = id if id else generate_experiment_id()
    experiment_path = os.path.join(EXPERIMENTS_PATH, experiment_id)

    if os.path.exists(experiment_path):
        raise ExperimentAlreadyExistsException(
            f"Experiment '{experiment_id}' already exists."
        )

    ensure_directory_exists(experiment_path)
    console.print(f"Created experiment: [bold yellow]{experiment_id}[/bold yellow]")


@app.command(name="list")
def list_experiments(
    sort_by: SortBy = typer.Option(
        SortBy.alpha,
        "--sort-by",
        help="Sort experiments by 'alpha' (alphabetical) or 'date' (creation date).",
    ),
    desc: bool = typer.Option(False, "--desc", help="Sort in descending order."),
) -> None:
    """List all experiments in the experiments folder."""
    if not os.path.exists(EXPERIMENTS_PATH):
        console.print("No experiments found.")
        return

    entries = [e for e in os.scandir(EXPERIMENTS_PATH) if e.is_dir()]

    if not entries:
        console.print("No experiments found.")
        return

    if sort_by == SortBy.date:
        entries.sort(key=lambda e: e.stat().st_ctime, reverse=desc)
    else:
        entries.sort(key=lambda e: e.name, reverse=desc)

    table = Table()
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="yellow")
    table.add_column("Created", style="green", no_wrap=True)

    for i, entry in enumerate(entries, start=1):
        ctime = datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        table.add_row(str(i), entry.name, ctime)

    console.print(table)


@app.command()
def rename(
    old_id: str = typer.Option(
        None,
        "--old-id",
        help="Current experiment ID to rename. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    new_id: str = typer.Option(
        ...,
        "--new-id",
        help="New experiment ID name.",
    ),
) -> None:
    """Rename an existing experiment."""

    # Use current experiment if old_id not provided
    if old_id is None:
        if settings.experiment_id:
            old_id = settings.experiment_id
        else:
            raise LLMExerException(
                "No experiment ID provided. Use --old-id or set EXPERIMENT_ID in .env file."
            )

    old_path = os.path.join(EXPERIMENTS_PATH, old_id)
    new_path = os.path.join(EXPERIMENTS_PATH, new_id)

    if not os.path.exists(old_path):
        raise LLMExerException(f"Experiment '{old_id}' does not exist.")

    if os.path.exists(new_path):
        raise ExperimentAlreadyExistsException(f"Experiment '{new_id}' already exists.")

    os.rename(old_path, new_path)
    console.print(
        f"Renamed experiment: [bold yellow]{old_id}[/bold yellow] → [bold yellow]{new_id}[/bold yellow]"
    )


@app.command()
def current() -> None:
    """Display the current experiment ID loaded from .env."""

    if settings.experiment_id:
        experiment_path = os.path.join(EXPERIMENTS_PATH, settings.experiment_id)
        if os.path.exists(experiment_path):
            console.print(
                f"Current experiment: [bold yellow]{settings.experiment_id}[/bold yellow]"
            )
        else:
            console.print(
                f"Current experiment: [bold yellow]{settings.experiment_id}[/bold yellow] "
                f"[bold red](not found in {EXPERIMENTS_PATH})[/bold red]"
            )
    else:
        console.print(
            "[bold red]No current experiment set.[/bold red] Set EXPERIMENT_ID in .env file."
        )
