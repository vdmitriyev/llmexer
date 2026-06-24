"""Provides module specific constants."""

import os
from pathlib import Path

# BASEDIR = os.path.abspath(os.path.dirname(__file__))
BASEDIR = os.path.join(Path(__file__).resolve().parent.parent)
CLI_NAME = "llmexer"
LOGGER_NAME = "llmexer"
LOG_FILE_NAME = "llmexer.log"
APP_LOG_LEVEL = os.environ.get("APP_LOG_LEVEL", "INFO").upper()
PROJECTS_DIR = ".projects"
TEMP_DIR = "temp"
SEARCHES_DIR = "searches"
PAPERS_DIR = "papers"

LOG_FILE_PATH = os.path.join(BASEDIR, LOG_FILE_NAME)

TEMP_PATH = os.path.join(BASEDIR, TEMP_DIR)
PROJECTS_PATH = os.path.join(BASEDIR, PROJECTS_DIR)

DEFAULT_DOCLING_URL = "http://localhost:5001/"
# DEFAULT_DOCLING_USER = "docling"
# DEFAULT_DOCLING_PASSWORD = "docling"
