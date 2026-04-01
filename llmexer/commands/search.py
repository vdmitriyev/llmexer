"""Search group commands."""

import os
import uuid

import typer

from llmexer.common import ensure_directory_exists
from llmexer.configs import console
from llmexer.constants import EXPERIMENTS_PATH, SEARCHES_DIR
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

app = typer.Typer(help="Search online digital libraries for papers and metadata.")


def generate_search_id() -> str:
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
def search(
    input: str = typer.Option(
        None,
        "--input",
        help="Input string to be used during the search",
    ),
    eid: str = typer.Option(
        None, "--eid", help="Experiment ID to be used to store search results"
    ),
) -> None:
    """Runs a new search and saves results"""

    if eid is None:
        raise ExperimentIDRequiredException()

    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")

    search_id = id if id else generate_search_id()
    searches_path = os.path.join(EXPERIMENTS_PATH, SEARCHES_DIR)
    ensure_directory_exists(searches_path)

    console.print(f"Created searches directory: [bold yellow]{eid}[/bold yellow]")
    console.print(f"Running search under id: [bold yellow]{search_id}[/bold yellow]")
    console.print(f"Running search with input: [bold yellow]{input}[/bold yellow]")
