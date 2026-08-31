"""Base methods and feature to be used in experiment CLI command."""

import os
import uuid

DIR_EXPERIMENT = "experiment"
DIR_RESPONSES = "responses"

# The experiment's model roster. Its identity columns (``provider``,
# ``model_name``) deliberately match ``llm-params.csv`` and the generated tables.
FILE_LLM_MODELS = "llms-for-experiment.csv"

_OUTPUT_COLUMNS = [
    "ID",
    "code",
    "prompt",
    "tokens_estimate",
    "original_data",
    "model_name",
    "provider_name",
    "prompt_hash",
    "original_data_hash",
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

# Parameter columns copied from each ``llm-params.csv`` row into a generated
# row (``model_name``/``provider`` from that file are the join key / captured by
# llms-for-experiment.csv, so they are not duplicated here).
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
# Schema partition for the per-provider SQLite tables. Each provider table is
# built from COMMON_IDENTITY_COLUMNS + COMMON_PARAM_COLUMNS + that provider's
# entry in PROVIDER_PARAM_COLUMNS + RESULT_COLUMNS + HASH_COLUMNS. This keeps
# every provider's parameters in its own table (e.g. the openai table has no
# ollama_* columns), with the reproducibility hashes trailing at the end.

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

# Parameter columns shared by every provider table. ``model_name`` and
# ``provider_name`` (identity columns) already capture the model/provider, so
# they are not duplicated here.
COMMON_PARAM_COLUMNS = [
    "profile_name",
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

# SHA-256 reproducibility hashes, kept as the trailing columns of every table.
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
    required_files = ["data.csv", "llm-params.csv", "mapping.csv", FILE_LLM_MODELS]
    return all(os.path.exists(os.path.join(experiment_subdir_path, f)) for f in required_files)


def _get_generated_experiment_files(experiment_path: str) -> list[str]:
    """Get the sorted list of generated experiment databases (``experiment_*.db``)."""
    # Local import to avoid a circular import (``dao`` imports from this module).
    from llmexer.base.dao import list_db_files

    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    return list_db_files(experiment_subdir_path)
