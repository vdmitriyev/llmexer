"""Shared helpers for the SQLite-backed experiment tests."""

import pandas as pd

from llmexer.base.dao import ExperimentDAO

# Canonical generated rows used across run/manager tests (pre-``run`` state).
OLLAMA_ROW = {
    "ID": 1,
    "code": "D01_prompt01_llama3.3:latest_ollama-default",
    "prompt": "Hello world",
    "tokens_estimate": 2,
    "original_data": '{"ID":"D01"}',
    "model_name": "llama3.3:latest",
    "provider_name": "ollama",
    "prompt_hash": "abc123",
    "original_data_hash": "def456",
    "profile_name": "ollama-default",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 512,
    "ollama_context_window": 4096,
    "ollama_repeat_penalty": 1.1,
}

OPENAI_ROW = {
    "ID": 2,
    "code": "D01_prompt01_gpt-4o_openai-default",
    "prompt": "Hello world",
    "tokens_estimate": 2,
    "original_data": '{"ID":"D01"}',
    "model_name": "gpt-4o",
    "provider_name": "openai",
    "prompt_hash": "abc123",
    "original_data_hash": "def456",
    "profile_name": "openai-default",
    "temperature": 0.7,
    "top_p": 1.0,
    "max_tokens": 512,
    "openai_seed": 42,
}


LITELLM_ROW = {
    "ID": 3,
    "code": "D01_prompt01_gpt-oss:120b_litellm-default",
    "prompt": "Hello world",
    "tokens_estimate": 2,
    "original_data": '{"ID":"D01"}',
    "model_name": "gpt-oss:120b",
    "provider_name": "litellm",
    "prompt_hash": "abc123",
    "original_data_hash": "def456",
    "profile_name": "litellm-default",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512,
    "litellm_min_p": 0.05,
    "litellm_best_of": 1,
}


def seed_db(db_path, rows_by_provider):
    """Create a database at ``db_path`` and insert rows per provider.

    ``rows_by_provider`` maps a provider name to a list of row dicts.
    """

    with ExperimentDAO(str(db_path), create=True) as dao:
        for provider, rows in rows_by_provider.items():
            dao.insert_rows(provider, rows)


def read_experiment_df(db_path):
    """Load all rows across provider tables into one DataFrame (ordered by ID)."""

    with ExperimentDAO(str(db_path)) as dao:
        rows = dao.fetch_rows()
    df = pd.DataFrame(rows)
    if "_provider" in df.columns:
        df = df.drop(columns=["_provider"])
    return df


def table_columns(db_path, provider):
    """Return the ordered column names of a provider's table."""

    with ExperimentDAO(str(db_path)) as dao:
        return list(dao.provider_tables()[provider].c.keys())


def params_table_columns(db_path, provider):
    """Return the ordered column names of a provider's params table."""

    with ExperimentDAO(str(db_path)) as dao:
        return list(dao.params_tables()[provider].c.keys())


def read_params_rows(db_path, provider):
    """Return all ``params_<provider>`` rows as dicts, ordered by the key pair."""

    with ExperimentDAO(str(db_path)) as dao:
        return dao.fetch_params_rows(provider)


def find_db(exp_subdir):
    """Return the single ``experiment_*.db`` path in ``exp_subdir``."""

    return sorted(exp_subdir.glob("experiment_*.db"))[0]


def list_dbs(exp_subdir):
    """Return sorted ``experiment_*.db`` file names in ``exp_subdir``."""

    return sorted(p.name for p in exp_subdir.glob("experiment_*.db"))
