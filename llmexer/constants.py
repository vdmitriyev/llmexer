"""Provides module specific constants."""

import os
from pathlib import Path

CLI_NAME = "llmexer"
LOGGER_NAME = "llmexer"
LOG_FILE_NAME = "llmexer.log"
APP_LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
DEFAULT_PROJECTS_DIR = ".projects"
LLMEXER_HOME_DIR = ".llmexer"
INI_FILE_NAME = "llmexer.ini"
TEMP_DIR = "temp"
SEARCHES_DIR = "searches"
PAPERS_DIR = "papers"
SEARCHES_LOGS_DIR = "logs"
ANALYSIS_DIR = "analysis"
# Replaced notebooks/modules are kept here rather than beside the live ones.
ANALYSIS_BACKUP_DIR = ".backup"

# Package-relative paths. Unlike ``BASEDIR`` (set from `.llmexer/llmexer.ini`),
# these point inside the installed package, where bundled assets such as the
# Jinja2 HTML export template live.
PACKAGE_PATH = Path(__file__).resolve().parent
PACKAGE_DATA_DIR = "data"
PACKAGE_DATA_PATH = PACKAGE_PATH / PACKAGE_DATA_DIR

# Set at CLI start-up by `llmexer.paths.apply_paths()`, after `.env` has been loaded.
BASEDIR = None
PROJECTS_DIR = None
PROJECTS_PATH = None
TEMP_PATH = None
LOG_FILE_PATH = None

# Hard ceiling on the number of works processed from a single OpenAlex query.
# Overridable at runtime via the MAX_OPEN_ALEX_RESPONSES environment variable.
DEFAULT_MAX_OPENALEX_RESPONSES = 5000

# Default ceiling on what one `experiment run` may spend through OpenRouter, in USD.
# Overridable at runtime via the PROVIDER_OPENROUTER_MAX_SPEND environment variable.
# The tally lives in memory for the duration of a single run: when the command
# exits, the amount spent resets to zero and the next run starts from the full cap.
DEFAULT_OPENROUTER_MAX_SPEND_USD = 5.0

DEFAULT_DOCLING_URL = "http://localhost:5001/"
# DEFAULT_DOCLING_USER = "docling"
# DEFAULT_DOCLING_PASSWORD = "docling"
