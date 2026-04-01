"""Search group commands."""

import os

import typer

from llmexer.constants import EXPERIMENTS_PATH
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)

app = typer.Typer(help="Work with papers.")


@app.command()
def rename(
    eid: str = typer.Option(
        None, "--eid", help="Experiment ID to be used to store search results"
    ),
) -> None:
    """Renames papers of the given experiment"""

    if eid is None:
        raise ExperimentIDRequiredException()

    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")
