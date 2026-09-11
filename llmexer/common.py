import json
import os
import re
from datetime import datetime, timezone

import requests

import llmexer.constants as _constants
from llmexer.configs import settings
from llmexer.constants import TEMP_PATH
from llmexer.exceptions import (
    LLMExerException,
    ProjectIDRequiredException,
    ProjectNotExistsException,
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
        logger.error(f"I/O Error: Could not save response to: '{filepath}'. Details: {e}")
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


def get_proper_pid(pid: str) -> str:
    """Use current project if pid not provided.

    Args:
        pid (str): project ID

    Raises:
        ProjectIDRequiredException: when no project ID is available.

    Returns:
        str: the resolved project ID.
    """

    if pid is None:
        if settings.project_id:
            pid = settings.project_id
        else:
            raise ProjectIDRequiredException("No project ID provided. Use --pid or set PROJECT_ID in .env file.")

    return pid


def get_project_directory_path(pid: str):
    """
    Return the project directory path, raising if it does not exist.
    """

    project_path = os.path.join(_constants.PROJECTS_PATH, pid)
    if not os.path.exists(project_path):
        raise ProjectNotExistsException(f"Project '{pid}' does not exist.")

    return project_path


def get_experiment_subdir_path(pid: str) -> str:
    """Return the ``experiment/`` subdir for a project, raising if missing.

    Raises ``LLMExerException`` if the project has not been initialised.
    """
    from llmexer.base.experiment import DIR_EXPERIMENT

    project_path = get_project_directory_path(pid)
    experiment_subdir_path = os.path.join(project_path, DIR_EXPERIMENT)
    if not os.path.exists(experiment_subdir_path):
        raise LLMExerException(
            f"Project '{pid}' has not been initialised. " f"Run `experiment init --pid {pid}` first."
        )
    return experiment_subdir_path


def strip_code_fence(text: str) -> str:
    """Drop a Markdown code fence wrapping a value, if there is one.

    Models routinely answer with their JSON inside ```` ```json ... ``` ````.
    Left in place that prefix makes every such answer unparseable, so the fence
    is peeled off before parsing - the text itself is never modified otherwise.

    NB: ``llmexer/base/analysis/transform.py`` carries a verbatim copy of this
    function on purpose. That module is copied into a project's ``analysis/``
    folder and imported as a top-level module, so it cannot import from
    ``llmexer``; ``tests/test_analysis_transform.py`` asserts the two stay in
    step.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # First line is the fence, optionally carrying a language tag ("```json").
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def next_backup_name(folder: str, stem: str, suffix: str = ".csv") -> str:
    """Return the next ``<stem>_backup_<YYYYMMDD>_<NN><suffix>`` name for ``folder``.

    ``<NN>`` is a zero-padded counter, one greater than the highest counter among
    today's existing ``<stem>_backup_<today>_*<suffix>`` files (starts at ``01``),
    so repeated runs on the same day never overwrite each other.
    """

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"{stem}_backup_{date}_"
    counter = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith(suffix):
                token = fname[len(prefix) : -len(suffix)]
                try:
                    counter = max(counter, int(token))
                except ValueError:
                    continue
    return f"{prefix}{counter + 1:02d}{suffix}"


def safe_filename_part(value: str, max_length: int = 40) -> str:
    """Return ``value`` reduced to characters that are safe in a file name.

    Letters, digits, ``.``, ``-`` and ``_`` are kept; every other character
    becomes ``-``. Runs of ``-`` and of ``_`` are collapsed to one, so a value
    never grows a ``__`` that reads as a separator, and leading or trailing
    ``-._`` are dropped. The result is cut to ``max_length`` and falls back to
    ``"unnamed"`` when nothing is left.
    """

    cleaned = re.sub(r"[^A-Za-z0-9._-]", "-", str(value))
    cleaned = re.sub(r"-+", "-", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned[:max_length].strip("-._")

    return cleaned or "unnamed"
