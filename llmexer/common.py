import json
import os
from datetime import datetime

import requests

import llmexer.constants as _constants
from llmexer.configs import settings
from llmexer.constants import TEMP_PATH
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
)
from llmexer.logger import get_logger
from llmexer.version import package_version

logger = get_logger()


def get_user_agent():
    return f"{_constants.CLI_NAME}/{package_version()} (python-requests/{requests.__version__})"


def make_http_session() -> requests.Session:
    """Return a new requests.Session with the llmexer User-Agent header set."""

    ua_custom = get_user_agent()
    session = requests.Session()
    session.headers.update({"User-Agent": ua_custom})
    return session


def __save_json__(content: dict, filepath: str = None):
    """
    Saves the provided dictionary to a JSON file.

    If no filename is provided, a unique filename based on the current timestamp will be generated.

    Args:
        content (dict): The dictionary content to be saved as JSON.
        filepath (str, optional): The desired name for the JSON file (e.g., "data.json").
                                    If None, a timestamped filename is used. Defaults to None.
    """

    ensure_directory_exists(TEMP_PATH)

    if not filepath:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(TEMP_PATH, f"dump_{timestamp}.json")

    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=4)  # Pretty print JSON with indent=4
        logger.debug(f"API response successfully saved to: '{filepath}'")
        return filepath
    except IOError as e:
        logger.error(
            f"I/O Error: Could not save response to: '{filepath}'. Details: {e}"
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred while saving response to: '{filepath}'. Exception: {e}",
            exc_info=True,
        )


def ensure_directory_exists(path: str):
    """
    Ensures a directory exists using the os module.
    Creates the directory if it doesn't exist.
    Handles creation of parent directories if they don't exist.
    """

    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            logger.info(f"Directory has been created: {path}")
    except OSError as e:
        logger.error(f"Error creating directory '{path}': {e}")


def get_proper_eid(eid: str) -> str:
    """Use current experiment if eid not provided.

    Args:
        eid (str): experiment ID

    Raises:
        ExperimentIDRequiredException: _description_

    Returns:
        str: _description_
    """

    if eid is None:
        if settings.experiment_id:
            eid = settings.experiment_id
        else:
            raise ExperimentIDRequiredException(
                "No experiment ID provided. Use --eid or set EXPERIMENT_ID in .env file."
            )

    return eid


def get_experiment_directory_path(eid: str):
    """
    Ensures a directory exists using the os module.
    """

    experiment_path = os.path.join(_constants.EXPERIMENTS_PATH, eid)
    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' does not exist.")

    return experiment_path
