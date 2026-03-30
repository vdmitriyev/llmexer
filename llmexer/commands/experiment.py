"""Experiment group commands."""

import uuid
from datetime import datetime

import typer

from llmexer.common import ensure_directory_exists
from llmexer.configs import console
from llmexer.constants import EXPERIMENTS_PATH

app = typer.Typer(help="Commands for managing LLM experiments.")


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
def create() -> None:
    """Create a new experiment folder under .experiments."""
    experiment_id = generate_experiment_id()
    experiment_path = f"{EXPERIMENTS_PATH}/{experiment_id}"

    ensure_directory_exists(experiment_path)
    console.print(f"Created experiment: [bold yellow]{experiment_id}[/bold yellow]")
