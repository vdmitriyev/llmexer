"""Search group commands."""

import os

import typer

from llmexer.configs import settings
from llmexer.constants import EXPERIMENTS_PATH
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

app = typer.Typer(help="Work with papers.")


@app.command()
def rename(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to be used to store search results. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Renames papers of the given experiment"""

    # Use current experiment if eid not provided
    if eid is None:
        if settings.experiment_id:
            eid = settings.experiment_id
        else:
            raise ExperimentIDRequiredException(
                "No experiment ID provided. Use --eid or set EXPERIMENT_ID in .env file."
            )

    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")
