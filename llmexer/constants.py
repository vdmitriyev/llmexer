"""Provides module specific constants."""

import os
from pathlib import Path

DEFAULT_DIR = Path.cwd()
BASEDIR = Path(os.getenv("LLMEXER_BASEDIR", DEFAULT_DIR))

CLI_NAME = "llmexer"
LOGGER_NAME = "llmexer"
LOG_FILE_NAME = "llmexer.log"
APP_LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
PROJECTS_DIR = ".projects"
TEMP_DIR = "temp"
SEARCHES_DIR = "searches"
PAPERS_DIR = "papers"
SEARCHES_LOGS_DIR = "logs"

LOG_FILE_PATH = os.path.join(BASEDIR, LOG_FILE_NAME)
TEMP_PATH = os.path.join(BASEDIR, TEMP_DIR)
PROJECTS_PATH = os.path.join(BASEDIR, PROJECTS_DIR)

# Hard ceiling on the number of works processed from a single OpenAlex query.
# Overridable at runtime via the MAX_OPEN_ALEX_RESPONSES environment variable.
DEFAULT_MAX_OPENALEX_RESPONSES = 5000

DEFAULT_DOCLING_URL = "http://localhost:5001/"
# DEFAULT_DOCLING_USER = "docling"
# DEFAULT_DOCLING_PASSWORD = "docling"
