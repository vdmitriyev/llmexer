f"""Base methods and feature to be used in experiment CLI command."""

import os
import uuid

DIR_EXPERIMENT = "experiment"
DIR_RESPONSES = "responses"
DIR_PROMPTS = "prompts"

# files describing an experiment
FILE_DATA = "data.csv"
FILE_MAPPING = "mapping.csv"
FILE_LLM_PARAMS = "llm-params.csv"
FILE_LLMS_FOR_EXPERIMENT = "llms-for-experiment.csv"

# Identity triple shared by llms-for-experiment.csv and llm-params.csv. A model
# row joins EXACTLY ONE profile row on these three columns; to run one model
# under two profiles, list it twice in llms-for-experiment.csv. The combination
# must be unique within each file.
CSV_JOIN_KEY_COLUMNS = [
    "provider",
    "model_name",
    "profile_name",
]

# Parameter columns copied from each ``llm-params.csv`` row into a generated
# row (``model_name``/``provider`` from that file are the join key / captured by
# llms-for-experiment.csv, so they are not duplicated here). The DAO routes
# these values into the ``params_<provider>`` table.
_PARAM_COLUMNS = [
    "profile_name",
    "temperature",
    "top_p",
    "max_tokens",
    "ollama_context_window",
    "ollama_repeat_penalty",
    "vllm_min_p",
    "vllm_best_of",
    "openai_seed",
    "gemini_thinking_level",
    "litellm_min_p",
    "litellm_best_of",
]

# --------------------------------------------------------------------- SQLite
# Schema partition for the per-provider SQLite tables. Each provider gets TWO
# tables, so a parameter set is stored once instead of being repeated on every
# generated row:
#
#   experiment_<provider> = COMMON_IDENTITY_COLUMNS + PARAMS_KEY_COLUMNS
#                           + RESULT_COLUMNS + HASH_COLUMNS
#   params_<provider>     = PARAMS_KEY_COLUMNS + COMMON_PARAM_COLUMNS
#                           + PROVIDER_PARAM_COLUMNS[provider]
#
# Each provider keeps its own parameters (the openai params table has no
# ollama_* columns), and ``ExperimentDAO.fetch_rows()`` joins the two tables
# back together so the rest of the codebase still sees one flat row dict.

# Identity / prompt columns shared by every provider table.
COMMON_IDENTITY_COLUMNS = [
    "ID",
    "code",
    "prompt",
    "tokens_estimate",
    "original_data",
    "model_name",
    "provider_name",
]

# Join key shared by BOTH tables: ``experiment_<provider>`` carries it so a row
# can find its parameters, ``params_<provider>`` uses it as its composite
# PRIMARY KEY. ``params_code`` is ``f"{model_name}_{provider}"``; one such pair
# can have several profiles in ``llm-params.csv``, hence the two-column key.
# NB: the column cannot be called ``code`` -- ``experiment_<provider>`` already
# has one and SQLite identifiers are case-insensitive.
PARAMS_KEY_COLUMNS = [
    "params_code",
    "profile_name",
]

# Parameter *values* shared by every provider, stored in ``params_<provider>``.
# ``profile_name`` is not repeated here: it is half of PARAMS_KEY_COLUMNS.
COMMON_PARAM_COLUMNS = [
    "temperature",
    "top_p",
    "max_tokens",
]

# Parameter columns specific to each provider (keyed by lower-cased provider).
# A provider not listed here gets no extra parameter columns.
PROVIDER_PARAM_COLUMNS = {
    "ollama": ["ollama_context_window", "ollama_repeat_penalty"],
    "vllm": ["vllm_min_p", "vllm_best_of"],
    "openai": ["openai_seed"],
    "gemini": ["gemini_thinking_level"],
    "litellm": ["litellm_min_p", "litellm_best_of"],
}

# Result columns written back once a row has been run. ``response_json`` stores
# the full per-call JSON payload (the same dict also exported to responses/).
RESULT_COLUMNS = [
    "response_text",
    "usage_tokens",
    "status",
    "state",
    "call_count",
    "total_tokens",
    "elapsed_seconds",
    "timestamp",
    "response_json",
]

# SHA-256 reproducibility hashes, kept as the trailing columns of every
# ``experiment_<provider>`` table.
HASH_COLUMNS = [
    "prompt_hash",
    "original_data_hash",
]


def generate_project_id() -> str:
    """
    Generate a unique project ID formatted as 'YYYYMMDD-GUID'

    Returns:
      str: A string in the format 'YYYYMMDD-UUID'.
    """
    from datetime import datetime, timezone

    now_utc = datetime.now(timezone.utc)
    formatted_datetime = now_utc.strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8]
    return f"{formatted_datetime}-{unique_id}"


def _is_experiment_initialized(experiment_path: str) -> bool:
    """Check if an experiment has been initialized with required CSV files."""
    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    required_files = [FILE_DATA, FILE_LLM_PARAMS, FILE_MAPPING, FILE_LLMS_FOR_EXPERIMENT]
    return all(os.path.exists(os.path.join(experiment_subdir_path, f)) for f in required_files)


def _get_generated_experiment_files(experiment_path: str) -> list[str]:
    """Get the sorted list of generated experiment databases (``experiment_*.db``)."""
    # Local import to avoid a circular import (``dao`` imports from this module).
    from llmexer.base.dao import list_db_files

    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    return list_db_files(experiment_subdir_path)
