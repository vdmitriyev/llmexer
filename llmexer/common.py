import json
import os
from datetime import datetime

from llmexer.constants import TEMP_PATH
from llmexer.logger import get_logger

logger = get_logger()


class GlobalFlags:
    """Class to hold global configuration state."""

    dry_run: bool = False
    verbose: bool = False
    experiment_id: str = None


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
