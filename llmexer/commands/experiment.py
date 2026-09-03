"""Experiment group commands of the CLI interface."""

import hashlib
import json
import math
import os
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from jinja2 import BaseLoader, DebugUndefined, Environment
from rich.table import Table

from llmexer.base.dao import (
    ExperimentDAO,
    _clean_value,
    latest_db,
    list_db_files,
    next_db_filename,
    params_code_for,
    try_params_table_name_for,
    try_table_name_for,
)
from llmexer.base.experiment import (
    _PARAM_COLUMNS,
    CSV_JOIN_KEY_COLUMNS,
    DIR_EXPERIMENT,
    DIR_PROMPTS,
    DIR_RESPONSES,
    FILE_DATA,
    FILE_LLM_PARAMS,
    FILE_LLMS_FOR_EXPERIMENT,
    FILE_MAPPING,
    PARAMS_KEY_COLUMNS,
    _get_generated_experiment_files,
    _is_experiment_initialized,
)
from llmexer.common import (
    ensure_directory_exists,
    get_experiment_subdir_path,
    get_project_directory_path,
    get_proper_pid,
)
from llmexer.configs import console, cprint, settings
from llmexer.constants import PAPERS_DIR, PROJECTS_PATH, SEARCHES_DIR
from llmexer.exceptions import LLMExerException


class SortBy(str, Enum):
    alpha = "alpha"
    date = "date"


app = typer.Typer(help="Manage LLM experiments.")


def _key_part(value: Any) -> str:
    """Normalise one join-key cell: a missing value becomes ``""``, else stripped.

    ``pd.read_csv`` yields ``NaN`` for an empty cell and ``str(nan)`` is the
    string ``"nan"``, which would join against a profile literally named ``nan``
    and print ``nan`` in every warning. Collapse it to ``""`` instead.

    Args:
        value (Any): a raw CSV cell.

    Returns:
        str: the stripped value, or ``""`` when the cell is empty.
    """

    return "" if pd.isna(value) else str(value).strip()


def _join_key(provider: Any, model_name: Any, profile_name: Any) -> tuple[str, str, str]:
    """Identity key shared by ``llms-for-experiment.csv`` and ``llm-params.csv``.

    The join is on all THREE columns, so a model row matches exactly one profile
    row; running one model under two profiles means listing it twice in
    ``llms-for-experiment.csv``. Surrounding whitespace is stripped so a stray
    space in a CSV cell does not silently drop a model, but the comparison stays
    case-sensitive: model names are sent verbatim to the provider, where casing
    can be significant. An empty ``profile_name`` normalises to ``""`` and so
    matches nothing.

    Args:
        provider (Any): the ``provider`` cell.
        model_name (Any): the ``model_name`` cell.
        profile_name (Any): the ``profile_name`` cell.

    Returns:
        tuple[str, str, str]: ``(provider, model_name, profile_name)``, stripped.
    """

    return (_key_part(provider), _key_part(model_name), _key_part(profile_name))


def _row_join_key(row: Any) -> tuple[str, str, str]:
    """Build the join key from a CSV row of either file."""

    return _join_key(row["provider"], row["model_name"], row["profile_name"])


def _require_join_columns(df: pd.DataFrame, filename: str, pid: str, expected_header: str) -> None:
    """Fail fast when a CSV predates the three-column join key.

    Without this a ``llms-for-experiment.csv`` written before ``profile_name``
    existed would die with a bare ``KeyError`` deep inside the join.
    """

    missing = [column for column in CSV_JOIN_KEY_COLUMNS if column not in df.columns]
    if missing:
        raise LLMExerException(
            f"'{filename}' of project '{pid}' is missing required column(s): "
            f"{', '.join(missing)}. Expected header: '{expected_header}'. Models and profiles "
            f"are matched on {', '.join(CSV_JOIN_KEY_COLUMNS)}; list a model once per profile "
            "it should run under."
        )


def _check_unique_join_keys(df: pd.DataFrame, filename: str, pid: str) -> None:
    """Abort when the same (model_name, provider, profile_name) appears twice.

    The check runs on the NORMALISED key, not on the raw cells: ``duplicated()``
    would let ``' ollama '`` and ``'ollama'`` through, yet the join collapses
    them and the two rows would then fight over one parameter set.
    """

    counts = Counter(_row_join_key(row) for _, row in df.iterrows())
    duplicates = [key for key, count in counts.items() if count > 1]
    if not duplicates:
        return

    for provider, model_name, profile_name in duplicates:
        cprint(
            f"[bold red]Error:[/bold red] duplicated row in {filename}: "
            f"model_name='{model_name}', provider='{provider}', "
            f"profile_name='{profile_name}' (x{counts[(provider, model_name, profile_name)]})."
        )
    raise LLMExerException(
        f"'{filename}' of project '{pid}' has {len(duplicates)} duplicated row(s): the "
        "combination of 'model_name', 'provider' and 'profile_name' must be unique. "
        "Nothing was generated."
    )


def _find_db_files(experiment_subdir_path: str) -> list[str]:
    """List experiment SQLite databases (``experiment*.db``) in the subdir."""

    return list_db_files(experiment_subdir_path)


def _format_hms(seconds: float) -> str:
    """Format a duration in seconds as ``HH:MM:SS`` (hours not capped at 24)."""

    total = int(seconds or 0)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _resolve_experiment_db(pid: str, file: str) -> tuple[str, str]:
    """Resolve and validate a generated experiment database path for a project.

    Returns ``(db_path, experiment_subdir_path)``. When ``file`` is omitted the
    newest ``experiment*.db`` (highest counter) is used. Raises
    ``LLMExerException`` if the project is not initialised or no database exists.
    """

    experiment_subdir_path = get_experiment_subdir_path(pid)

    if file is None:
        db_path = latest_db(experiment_subdir_path)
        if db_path is None:
            raise LLMExerException(
                f"No experiment database found for project '{pid}'. " f"Run `experiment generate --pid {pid}` first."
            )
        return db_path, experiment_subdir_path

    db_path = file if os.path.isabs(file) else os.path.join(experiment_subdir_path, file)

    if not os.path.exists(db_path):
        raise LLMExerException(f"Experiment database not found: '{db_path}'.")

    return db_path, experiment_subdir_path


def _next_backup_name(folder: str, stem: str) -> str:
    """Return the next ``<stem>_backup_<YYYYMMDD>_<NN>.csv`` name for ``folder``.

    ``<NN>`` is a zero-padded counter, one greater than the highest counter among
    today's existing ``<stem>_backup_<today>_*.csv`` files (starts at ``01``).
    """

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"{stem}_backup_{date}_"
    counter = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith(".csv"):
                token = fname[len(prefix) : -len(".csv")]
                try:
                    counter = max(counter, int(token))
                except ValueError:
                    continue
    return f"{prefix}{counter + 1:02d}.csv"


def _write_csv_with_backup(folder: str, filename: str, df: pd.DataFrame) -> tuple[str, str]:
    """Write ``df`` to ``folder/filename``, backing up any existing file first.

    The backup is named ``<stem>_backup_<YYYYMMDD>_<NN>.csv`` so repeated runs on
    the same day never overwrite each other.

    Returns ``(path, backup_name)`` where ``backup_name`` is ``""`` when no prior
    file existed.
    """

    path = os.path.join(folder, filename)
    backup_name = ""
    if os.path.exists(path):
        backup_name = _next_backup_name(folder, os.path.splitext(filename)[0])
        shutil.copy2(path, os.path.join(folder, backup_name))
    df.to_csv(path, index=False, sep=";", encoding="utf-8")
    return path, backup_name


def _write_data_csv(experiment_subdir_path: str, df: pd.DataFrame) -> tuple[str, str]:
    """Write ``df`` to ``experiment/data.csv``, backing up any existing file first.

    Returns ``(data_path, backup_name)`` where ``backup_name`` is ``""`` when no
    prior ``data.csv`` existed.
    """

    return _write_csv_with_backup(experiment_subdir_path, FILE_DATA, df)


def _resolve_prompt_ids(prompts_subdir: str, prompt: list[str] | None) -> list[str]:
    """Resolve ``--prompt`` values into existing prompt IDs from ``prompts/``.

    Each value may itself be a comma-separated list, and the ``.txt`` extension is
    optional: both ``prompt01`` and ``prompt01.txt`` resolve to the ID
    ``prompt01`` — the extension-less form ``experiment generate`` expects in
    ``mapping.csv``. Duplicates are dropped, keeping the order given.

    When no value is passed, every ``*.txt`` in ``prompts/`` is used, sorted by
    filename.

    Raises:
        LLMExerException: if a name escapes ``prompts/``, if any named prompt file
            is missing (all missing names are reported at once), or if no prompt
            templates exist at all.
    """

    names: list[str] = []
    for value in prompt or []:
        names.extend(part.strip() for part in str(value).split(","))
    names = [name for name in names if name]

    prompt_ids: list[str] = []

    if not names:
        prompt_ids = sorted(
            os.path.splitext(fname)[0] for fname in os.listdir(prompts_subdir) if fname.lower().endswith(".txt")
        )
        if not prompt_ids:
            raise LLMExerException(
                f"No prompt templates found in '{prompts_subdir}'. Add at least one '<name>.txt' file first."
            )
        return prompt_ids

    for name in names:
        if os.path.sep in name or (os.path.altsep and os.path.altsep in name) or ".." in name:
            raise LLMExerException(f"Invalid prompt name '{name}': it must be a file name inside 'prompts/'.")
        prompt_id = name[: -len(".txt")] if name.lower().endswith(".txt") else name
        if prompt_id not in prompt_ids:
            prompt_ids.append(prompt_id)

    missing = [
        f"{prompt_id}.txt"
        for prompt_id in prompt_ids
        if not os.path.exists(os.path.join(prompts_subdir, f"{prompt_id}.txt"))
    ]
    if missing:
        raise LLMExerException(f"Prompt file(s) not found in '{prompts_subdir}': {', '.join(missing)}.")

    return prompt_ids


@app.command()
def init(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to initialise. If not provided, uses PROJECT_ID from .env.",
    )
) -> None:
    """Initialise a project with a standard folder structure and template files"""

    pid = get_proper_pid(pid)
    project_path = get_project_directory_path(pid)

    # Raise if already initialised
    experiment_subdir_path = os.path.join(project_path, DIR_EXPERIMENT)
    if os.path.exists(experiment_subdir_path):
        raise LLMExerException(f"Project '{pid}' has already been initialised.")

    # Create subfolders
    prompts_subdir = os.path.join(experiment_subdir_path, DIR_PROMPTS)
    ensure_directory_exists(experiment_subdir_path)
    ensure_directory_exists(prompts_subdir)

    # llms-for-experiment.csv
    models_path = os.path.join(experiment_subdir_path, FILE_LLMS_FOR_EXPERIMENT)
    with open(models_path, "w", encoding="utf-8") as f:
        f.write("provider;model_name;profile_name;notes\n")
        f.write("ollama;gemma4:31b;ollama-gemma4-default;local model\n")
        f.write("ollama;phi4:14b;ollama-phi4-creative;local model\n")
        f.write("litellm;gemma4:31b;litellm-gemma4-default;local model\n")

    # data.csv
    data_path = os.path.join(experiment_subdir_path, FILE_DATA)
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("ID;Title;Abstract\n")
        f.write("D01;Sample Paper Title One;This is the abstract of the first sample paper.\n")
        f.write("D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n")

    # mapping.csv
    mapping_path = os.path.join(experiment_subdir_path, FILE_MAPPING)
    with open(mapping_path, "w", encoding="utf-8") as f:
        f.write("data_id;prompt_id\n")
        f.write("D01;prompt01\n")
        f.write("D02;prompt01\n")

    # prompts/prompt01.txt
    prompt_path = os.path.join(prompts_subdir, "prompt01.txt")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write(
            "Here is the title: {{title}}.\n\n"
            "Here is the abstract: {{abstract}}.\n\n"
            "Count number of words in the both title and abstract."
        )

    # llm-params.csv
    llm_params_path = os.path.join(experiment_subdir_path, FILE_LLM_PARAMS)
    with open(llm_params_path, "w", encoding="utf-8") as f:
        f.write(
            "provider;model_name;profile_name;temperature;top_p;max_tokens;"
            "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level;"
            "litellm_min_p;litellm_best_of\n"
        )
        f.write("ollama;gemma4:31b;ollama-gemma4-default;0.7;1.0;512;4096;1.1;;;;;;\n")
        f.write("ollama;phi4:14b;ollama-phi4-creative;1.2;0.95;512;4096;1.0;;;;;;\n")
        f.write("openai;gpt-4o;openai-default;0.7;1.0;512;;;;;42;;;\n")
        f.write("vllm;meta-llama/Llama-3-8b;vllm-default;0.7;0.9;512;;;0.05;1;;;;\n")
        f.write("gemini;gemini-2.0-flash;gemini-default;0.7;1.0;512;;;;;;standard;;\n")
        f.write("litellm;minimax-m2.7:229b;litellm-minimax-m2-default;0.7;0.9;4096;;;;;;;0.05;1\n")
        f.write("litellm;gemma4:31b;litellm-gemma4-default;0.7;0.9;512;;;;;;;;1\n")

    cprint(f"Init project [bold yellow]{pid}[/bold yellow] with standard structure.")


@app.command(name="copy-papers")
def copy_papers(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Copy parsed papers (.md/.txt) from the project's papers/ folder into data.csv

    Each parsed paper becomes a row ``ID;filename;content`` with IDs ``P01``,
    ``P02``, … ordered alphabetically by filename. When a paper has both a ``.md``
    and a ``.txt`` the Markdown is preferred. An existing ``data.csv`` is backed
    up first.
    """

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)
    papers_path = os.path.join(get_project_directory_path(pid), PAPERS_DIR)

    if not os.path.isdir(papers_path):
        raise LLMExerException(f"No papers folder found for project '{pid}': '{papers_path}'.")

    # Group parsed files by stem, preferring .md over .txt.
    chosen: dict[str, str] = {}
    for fname in os.listdir(papers_path):
        stem, ext = os.path.splitext(fname)
        ext = ext.lower()
        if ext not in (".md", ".txt"):
            continue
        if stem not in chosen or (ext == ".md" and chosen[stem].endswith(".txt")):
            chosen[stem] = fname

    filenames = sorted(chosen.values())
    if not filenames:
        cprint(
            "[bold yellow]Warning:[/bold yellow] no parsed papers (.md/.txt) found in "
            f"'{papers_path}' — nothing to copy. Run `papers extract` first."
        )
        return

    rows = []
    for index, fname in enumerate(filenames, start=1):
        content = Path(papers_path, fname).read_text(encoding="utf-8")
        rows.append({"ID": f"P{index:02d}", "filename": fname, "content": content})

    df = pd.DataFrame(rows, columns=["ID", "filename", "content"])
    _, backup_name = _write_data_csv(experiment_subdir_path, df)

    backup_note = f" (backed up previous {FILE_DATA} → {backup_name})" if backup_name else ""
    cprint(
        f"Copied [bold green]{len(rows)}[/bold green] paper(s) → "
        f"[bold yellow]{FILE_DATA}[/bold yellow]{backup_note}"
    )


@app.command(name="copy-search")
def copy_search(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    file: str = typer.Option(
        ...,
        "--file",
        help="Search results CSV to copy from (absolute, or relative to the " "project's searches/ folder).",
    ),
) -> None:
    """Copy a search results file into data.csv

    Writes rows ``ID;Title;Abstract;doi;authors`` with IDs ``S01``, ``S02``, …
    preserving the source file's row order. An existing ``data.csv`` is backed
    up first.
    """

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)

    search_path = file if os.path.isabs(file) else os.path.join(get_project_directory_path(pid), SEARCHES_DIR, file)
    if not os.path.exists(search_path):
        raise LLMExerException(f"Search file not found: '{search_path}'.")

    search_df = pd.read_csv(search_path, sep=";", encoding="utf-8")
    required = ["title", "abstract", "doi", "authors"]
    missing = [c for c in required if c not in search_df.columns]
    if missing:
        raise LLMExerException(f"Search file '{search_path}' is missing required column(s): " f"{', '.join(missing)}.")

    source = search_df[required].fillna("")
    df = pd.DataFrame(
        {
            "ID": [f"S{i:02d}" for i in range(1, len(source) + 1)],
            "Title": source["title"].values,
            "Abstract": source["abstract"].values,
            "doi": source["doi"].values,
            "authors": source["authors"].values,
        }
    )
    _, backup_name = _write_data_csv(experiment_subdir_path, df)

    backup_note = f" (backed up previous {FILE_DATA} → {backup_name})" if backup_name else ""
    cprint(
        f"Copied [bold green]{len(df)}[/bold green] search result(s) → "
        f"[bold yellow]{FILE_DATA}[/bold yellow]{backup_note}"
    )


@app.command(name="map")
def map_data(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    prompt: list[str] = typer.Option(
        None,
        "--prompt",
        help="Prompt file(s) from prompts/ to pair with every data row. Repeatable, and "
        "each value may itself be a comma-separated list. The '.txt' extension is "
        "optional. If omitted, every prompt in prompts/ is used.",
    ),
) -> None:
    """Build mapping.csv by pairing every row of data.csv with the selected prompt(s)

    Every data row is paired with every selected prompt (a cross join), written
    prompt by prompt: all data rows for the first prompt, then the second, and so
    on. An existing ``mapping.csv`` is backed up first.
    """

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)

    data_path = os.path.join(experiment_subdir_path, FILE_DATA)
    prompts_subdir = os.path.join(experiment_subdir_path, DIR_PROMPTS)

    for label, path in [(FILE_DATA, data_path), (f"{DIR_PROMPTS}/", prompts_subdir)]:
        if not os.path.exists(path):
            raise LLMExerException(f"Required file or directory not found for project '{pid}': {label}")

    data_df = pd.read_csv(data_path, sep=";", encoding="utf-8")
    if "ID" not in data_df.columns:
        raise LLMExerException(f"'{FILE_DATA}' of project '{pid}' is missing the required 'ID' column.")

    # Resolved before anything is written, so a typo leaves mapping.csv untouched.
    prompt_ids = _resolve_prompt_ids(prompts_subdir, prompt)

    data_ids = [str(value).strip() for value in data_df["ID"]]
    if not data_ids:
        cprint(
            f"[bold yellow]Warning:[/bold yellow] {FILE_DATA} has no rows — {FILE_MAPPING} left unchanged. "
            "Fill it in, or run `experiment copy-papers` / `experiment copy-search` first."
        )
        return

    rows = [{"data_id": data_id, "prompt_id": prompt_id} for prompt_id in prompt_ids for data_id in data_ids]
    df = pd.DataFrame(rows, columns=["data_id", "prompt_id"])

    mapping_path = os.path.join(experiment_subdir_path, FILE_MAPPING)
    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write {len(rows)} row(s) to '{mapping_path}'")
        return

    _, backup_name = _write_csv_with_backup(experiment_subdir_path, FILE_MAPPING, df)

    backup_note = f" (backed up previous {FILE_MAPPING} \u2192 {backup_name})" if backup_name else ""
    cprint(
        f"Mapped [bold green]{len(data_ids)}[/bold green] data row(s) \u00d7 "
        f"[bold green]{len(prompt_ids)}[/bold green] prompt(s) = "
        f"[bold green]{len(rows)}[/bold green] row(s) \u2192 "
        f"[bold yellow]{FILE_MAPPING}[/bold yellow]{backup_note}"
    )


def _load_experiment_inputs(
    pid: str, experiment_subdir_path: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Read and validate the four CSVs describing an experiment.

    Shared by ``generate`` (which turns them into a new database) and ``update``
    (which diffs them against an existing one), so both see the same inputs and
    fail on the same structural problems.

    Args:
        pid (str): project ID, used in the error messages.
        experiment_subdir_path (str): the project's ``experiment/`` folder.

    Returns:
        tuple: ``(models_df, data_df, mapping_df, params_df, prompts_subdir)``.
    """

    models_path = os.path.join(experiment_subdir_path, FILE_LLMS_FOR_EXPERIMENT)
    data_path = os.path.join(experiment_subdir_path, FILE_DATA)
    mapping_path = os.path.join(experiment_subdir_path, FILE_MAPPING)
    llm_params_path = os.path.join(experiment_subdir_path, FILE_LLM_PARAMS)
    prompts_subdir = os.path.join(experiment_subdir_path, DIR_PROMPTS)

    for label, path in [
        (FILE_LLMS_FOR_EXPERIMENT, models_path),
        (FILE_DATA, data_path),
        (FILE_MAPPING, mapping_path),
        (FILE_LLM_PARAMS, llm_params_path),
        (f"{DIR_PROMPTS}/", prompts_subdir),
    ]:
        if not os.path.exists(path):
            raise LLMExerException(f"Required file or directory not found for project '{pid}': {label}")

    models_df = pd.read_csv(models_path, sep=";", encoding="utf-8")
    data_df = pd.read_csv(data_path, sep=";", encoding="utf-8")
    mapping_df = pd.read_csv(mapping_path, sep=";", encoding="utf-8")
    params_df = pd.read_csv(llm_params_path, sep=";", encoding="utf-8")

    # Structural problems win over "the file is empty": a missing column or a
    # duplicated combination is reported before anything else happens.
    _require_join_columns(models_df, FILE_LLMS_FOR_EXPERIMENT, pid, "provider;model_name;profile_name;notes")
    _require_join_columns(params_df, FILE_LLM_PARAMS, pid, "provider;model_name;profile_name;temperature;...")
    _check_unique_join_keys(models_df, FILE_LLMS_FOR_EXPERIMENT, pid)
    _check_unique_join_keys(params_df, FILE_LLM_PARAMS, pid)

    return models_df, data_df, mapping_df, params_df, prompts_subdir


def _build_params_by_key(params_df: pd.DataFrame) -> dict:
    """Index ``llm-params.csv`` by its three-column join key.

    Profiles are matched to models on ALL THREE identity columns, so neither a
    profile written for another provider nor another profile of the same model
    can attach itself. Values are stripped but compared case-sensitively.
    Uniqueness was enforced by the caller, so there is exactly one row per key.
    """

    params_by_key: dict = {}
    for _, param_row in params_df.iterrows():
        key = _row_join_key(param_row)
        if not key[2]:
            # An unnamed profile would join a models row that is *also* blank,
            # which is never what the user meant. Reject it on both sides.
            cprint(
                f"[bold yellow]Warning:[/bold yellow] a row of {FILE_LLM_PARAMS} for model "
                f"'{key[1]}' has an empty 'profile_name' — ignored."
            )
            continue
        params_by_key[key] = param_row
    return params_by_key


def _report_unmatched_models(models_df: pd.DataFrame, params_by_key: dict) -> None:
    """Warn once per model row about an unknown provider or a missing profile."""

    # Surface unknown provider names now rather than at run time, where they
    # would otherwise create a table named after the typo. This is a warning,
    # not an error: a custom endpoint only needs PROVIDER_<UPPER>_URL to be set
    # by the time `run` executes. Local import keeps `generate` LLM-dep free.
    from llmexer.base.llm_provider import is_known_provider

    for _, model_row in models_df.iterrows():
        provider_name, model_name, profile_name = _row_join_key(model_row)
        if not is_known_provider(provider_name):
            cprint(
                f"[bold yellow]Warning:[/bold yellow] unknown provider "
                f"'{provider_name}' for model '{model_name}'. The "
                f"'provider' column of {FILE_LLMS_FOR_EXPERIMENT} takes a provider "
                "name (e.g. "
                "'litellm'), not a profile name (e.g. 'litellm-default') — a profile "
                "belongs in the 'profile_name' column. "
                "'experiment run' will fail unless PROVIDER_"
                f"{provider_name.upper()}_URL is set."
            )
        # Warned once per model here, rather than once per data row.
        if (provider_name, model_name, profile_name) not in params_by_key:
            empty_hint = (
                f" The 'profile_name' cell is empty; it must name a profile from {FILE_LLM_PARAMS}."
                if not profile_name
                else ""
            )
            cprint(
                f"[bold yellow]Warning:[/bold yellow] no profile in {FILE_LLM_PARAMS} "
                f"matches model '{model_name}' with provider "
                f"'{provider_name}' and profile '{profile_name}' — skipping. Profiles are "
                "matched on 'provider', 'model_name' and 'profile_name'." + empty_hint
            )


def _prompt_environment() -> Environment:
    """Build the Jinja2 environment prompt templates are rendered with.

    Autoescape stays off deliberately: these templates render LLM prompts, not
    HTML, and escaping would corrupt the text sent to the provider.
    ``DebugUndefined`` leaves an unknown placeholder verbatim instead of
    blanking it, so a typo in a template is visible in the rendered prompt.
    """

    return Environment(loader=BaseLoader(), undefined=DebugUndefined)


def _render_prompt(
    template_str: str,
    data_id: str,
    data_row: dict,
    env: Environment | None = None,
) -> tuple[str, str, str, str]:
    """Render one prompt template against one data row.

    Returns ``(rendered_prompt, prompt_hash, original_data, original_data_hash)``
    — the four reproducibility values every generated row carries. Shared by
    ``generate``/``update`` and ``try``, so a try can never be rendered or hashed
    differently from the row ``generate`` would have produced for it.

    Args:
        template_str (str): the raw Jinja2 template.
        data_id (str): the ``ID`` of the ``data.csv`` row.
        data_row (dict): that row's remaining columns.
        env (Environment | None): a shared Jinja2 environment; one is built when
            omitted (callers rendering many rows pass their own).
    """

    context = {k.lower(): v for k, v in data_row.items()}
    context["id"] = data_id

    original_data_dict = {"ID": data_id, **data_row}
    original_data_str = json.dumps(original_data_dict, ensure_ascii=False)
    original_data_hash = hashlib.sha256(original_data_str.encode("utf-8")).hexdigest()

    env = env or _prompt_environment()
    rendered_prompt = env.from_string(template_str).render(**context)
    prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()

    return rendered_prompt, prompt_hash, original_data_str, original_data_hash


def _combination_row(
    data_id: str,
    prompt_id: str,
    key: tuple[str, str, str],
    param_row: Any,
    rendered: tuple[str, str, str, str],
) -> dict:
    """Assemble one flat experiment row from a rendered prompt and a profile.

    ``key`` is the ``(provider, model_name, profile_name)`` join key. No ``ID``
    is assigned: ``generate`` numbers rows from 1, ``update`` continues the
    database's sequence and ``try`` lets SQLite assign one.
    """

    provider_name, model_name, profile_name = key
    rendered_prompt, prompt_hash, original_data_str, original_data_hash = rendered

    return {
        "code": f"{data_id}_{prompt_id}_{model_name}_{profile_name}",
        "prompt": rendered_prompt,
        "tokens_estimate": len(rendered_prompt) // 4,
        "original_data": original_data_str,
        "model_name": model_name,
        "provider_name": provider_name,
        "prompt_hash": prompt_hash,
        "original_data_hash": original_data_hash,
        **{k: param_row.get(k) for k in _PARAM_COLUMNS},
        # After the unpack: store the stripped profile so `code` and the params
        # table's primary key can never disagree.
        "profile_name": profile_name,
    }


def _build_generated_rows(
    models_df: pd.DataFrame,
    data_df: pd.DataFrame,
    mapping_df: pd.DataFrame,
    params_by_key: dict,
    prompts_subdir: str,
) -> list[dict]:
    """Render every (data row × prompt × model × parameters) combination.

    The returned dicts carry no ``ID``: ``generate`` numbers them from 1 while
    ``update`` continues the database's existing sequence.
    """

    data_lookup = data_df.set_index("ID").to_dict(orient="index")
    env = _prompt_environment()

    rows = []

    for _, mapping_row in mapping_df.iterrows():
        data_id = str(mapping_row["data_id"]).strip()
        prompt_id = str(mapping_row["prompt_id"]).strip()

        if data_id not in data_lookup:
            cprint(f"[bold yellow]Warning:[/bold yellow] data_id '{data_id}' not found in {FILE_DATA} — skipping.")
            continue

        prompt_file_path = os.path.join(prompts_subdir, f"{prompt_id}.txt")
        if not os.path.exists(prompt_file_path):
            cprint(f"[bold yellow]Warning:[/bold yellow] prompt file '{prompt_id}.txt' not found — skipping.")
            continue

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        rendered = _render_prompt(template_str, data_id, data_lookup[data_id], env=env)

        for _, model_row in models_df.iterrows():
            key = _row_join_key(model_row)
            param_row = params_by_key.get(key)
            if param_row is None:
                continue  # already reported once per model row
            rows.append(_combination_row(data_id, prompt_id, key, param_row, rendered))

    return rows


def _sort_rows_by_model_order(rows: list[dict], models_df: pd.DataFrame) -> list[dict]:
    """Sort generated rows by model order from ``llms-for-experiment.csv``.

    The sort is stable, so the mapping order is preserved within a model.
    """

    model_order = [str(name).strip() for name in models_df["model_name"]]

    def _model_rank(row: dict) -> int:
        name = str(row["model_name"])
        return model_order.index(name) if name in model_order else len(model_order)

    rows.sort(key=_model_rank)
    return rows


def _params_equal(db_value: Any, csv_value: Any) -> bool:
    """Compare one stored parameter with the value now in ``llm-params.csv``.

    SQLite returns what it stored (an ``INTEGER`` ``4096``) while pandas may hand
    over the same cell as a float (``4096.0``), so numbers are compared as
    floats. An empty cell is ``NaN`` on the CSV side and ``NULL`` on the database
    side; both normalise to ``None`` and compare equal.
    """

    left = _clean_value(db_value)
    right = _clean_value(csv_value)
    if left is None or right is None:
        return left is None and right is None
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def _collect_param_drift(dao: ExperimentDAO, rows: list[dict]) -> list[dict]:
    """Find profiles whose parameters differ from the ones already stored.

    Only profiles the database already knows are compared: one it has never seen
    is new, not drifted, and is simply inserted. The compared columns come from
    the *reflected* params table, so a database written with a different set of
    provider parameters is compared on the columns it actually has.

    Returns:
        list[dict]: one entry per drifted profile, with ``provider``,
        ``model_name``, ``profile_name`` and ``changes`` — a list of
        ``(column, db_value, csv_value)`` triples.
    """

    params_tables = dao.params_tables()
    stored_by_provider: dict = {}
    drift: list[dict] = []
    seen: set = set()

    for row in rows:
        provider = str(row["provider_name"]).lower()
        params_table = params_tables.get(provider)
        if params_table is None:
            continue  # a provider new to this database: nothing to drift from

        code = params_code_for(row.get("model_name"), provider)
        profile = "" if row.get("profile_name") is None else str(row["profile_name"])
        key = (provider, code, profile)
        if key in seen:
            continue  # the same parameter set recurs on every row of the cross join
        seen.add(key)

        if provider not in stored_by_provider:
            stored_by_provider[provider] = {
                (stored_row["params_code"], stored_row["profile_name"]): stored_row
                for stored_row in dao.fetch_params_rows(provider)
            }
        stored = stored_by_provider[provider].get((code, profile))
        if stored is None:
            continue  # a new parameter set, inserted as it is

        changes = [
            (column, stored.get(column), row.get(column))
            for column in params_table.c.keys()
            if column not in PARAMS_KEY_COLUMNS and not _params_equal(stored.get(column), row.get(column))
        ]
        if changes:
            drift.append(
                {
                    "provider": provider,
                    "model_name": row.get("model_name"),
                    "profile_name": profile,
                    "changes": changes,
                }
            )
    return drift


def _report_param_drift(drift: list[dict], db_name: str) -> None:
    """Print every drifted profile, then abort — a stored profile is immutable.

    Rewriting the parameters of a profile the database already used would make
    rows generated before and after the change indistinguishable, so the changed
    parameters have to be published under a new ``profile_name`` instead.
    """

    for entry in drift:
        cprint(
            f"[bold red]Error:[/bold red] parameter drift for "
            f"{entry['provider']} / {entry['model_name']} / {entry['profile_name']}"
        )
        for column, db_value, csv_value in entry["changes"]:
            cprint(f"  {column}: db={db_value}  csv={csv_value}")

    raise LLMExerException(
        f"{len(drift)} profile(s) of {FILE_LLM_PARAMS} differ from the parameters stored in "
        f"'{db_name}'. Nothing was added: a stored parameter set is immutable, otherwise rows "
        "generated before and after the change would be indistinguishable. Give the changed "
        f"parameters a new 'profile_name' in {FILE_LLM_PARAMS} (e.g. 'ollama-default-v2'), point "
        f"the matching {FILE_LLMS_FOR_EXPERIMENT} row at it, then run `experiment update` again."
    )


def _report_changed_source_text(codes: list[str]) -> None:
    """Warn about stored rows whose prompt or data text has since changed."""

    preview = ", ".join(codes[:5])
    more = f" (and {len(codes) - 5} more)" if len(codes) > 5 else ""
    cprint(
        f"[bold yellow]Warning:[/bold yellow] {len(codes)} stored row(s) no longer match the current "
        f"prompt template or {FILE_DATA}: {preview}{more}. They are left untouched — run "
        "`experiment generate` for a fresh database if the new text belongs to the experiment."
    )


@app.command()
def generate(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to generate prompts for. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Generate rendered prompts for all data-model combinations defined in the project"""

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)

    models_df, data_df, mapping_df, params_df, prompts_subdir = _load_experiment_inputs(pid, experiment_subdir_path)

    if params_df.empty:
        cprint(f"[bold yellow]Warning:[/bold yellow] {FILE_LLM_PARAMS} is empty — no rows will be generated.")
        return

    params_by_key = _build_params_by_key(params_df)
    _report_unmatched_models(models_df, params_by_key)

    rows = _build_generated_rows(models_df, data_df, mapping_df, params_by_key, prompts_subdir)

    if not rows:
        cprint(
            f"[bold yellow]Warning:[/bold yellow] No rows were generated. "
            f"Check your {FILE_MAPPING} and {FILE_DATA}."
        )
        return

    # Sort by model order (stable -> preserves mapping order within a model),
    # then number IDs 1..N across the whole generation.
    rows = _sort_rows_by_model_order(rows, models_df)
    for new_id, row in enumerate(rows, start=1):
        row["ID"] = new_id

    output_filename = next_db_filename(experiment_subdir_path)
    output_path = os.path.join(experiment_subdir_path, output_filename)

    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write {len(rows)} row(s) to '{output_path}'")
        return

    # Each provider gets its own pair of tables: experiment_<provider> for the
    # rows and params_<provider> for the parameter sets they point at.
    by_provider: dict = defaultdict(list)
    for row in rows:
        by_provider[str(row["provider_name"]).lower()].append(row)

    with ExperimentDAO(output_path, create=True) as dao:
        for provider, provider_rows in by_provider.items():
            dao.insert_rows(provider, provider_rows)

    cprint(
        f"Generated [bold green]{len(rows)}[/bold green] row(s) across "
        f"[bold green]{len(by_provider)}[/bold green] provider table(s) → "
        f"[bold yellow]{output_filename}[/bold yellow]"
    )


@app.command()
def update(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Path to a specific experiment_*.db to update in place. Defaults to the newest one.",
    ),
) -> None:
    """Updates an already generated database by adding new combinations from the input CSVs

    The input CSVs are re-read and cross-joined exactly as ``experiment generate``
    does, then compared with the database: combinations it does not hold yet are
    appended after the highest existing ID, while everything already stored —
    including collected results — is left untouched. Changed hyperparameters are
    never applied to a stored profile; they abort the update instead.
    """

    pid = get_proper_pid(pid)
    db_path, experiment_subdir_path = _resolve_experiment_db(pid, file)
    db_name = os.path.basename(db_path)

    models_df, data_df, mapping_df, params_df, prompts_subdir = _load_experiment_inputs(pid, experiment_subdir_path)

    if params_df.empty:
        cprint(f"[bold yellow]Warning:[/bold yellow] {FILE_LLM_PARAMS} is empty — nothing to add.")
        return

    params_by_key = _build_params_by_key(params_df)
    _report_unmatched_models(models_df, params_by_key)

    rows = _build_generated_rows(models_df, data_df, mapping_df, params_by_key, prompts_subdir)

    if not rows:
        cprint(
            f"[bold yellow]Warning:[/bold yellow] No combinations could be built. "
            f"Check your {FILE_MAPPING} and {FILE_DATA}."
        )
        return

    rows = _sort_rows_by_model_order(rows, models_df)

    # create=False reflects what the database already holds and keeps the
    # legacy-layout guard, which create=True would silently bypass.
    with ExperimentDAO(db_path) as dao:
        # Checked before anything is written: a changed profile aborts the whole
        # update, so the database can never end up half-updated.
        drift = _collect_param_drift(dao, rows)
        if drift:
            _report_param_drift(drift, db_name)

        stored_index = dao.fetch_row_index()

        new_rows = []
        changed_codes = []
        for row in rows:
            provider = str(row["provider_name"]).lower()
            stored = stored_index.get(provider, {}).get(str(row["code"]))
            if stored is None:
                new_rows.append(row)
            elif stored.get("prompt_hash") != row.get("prompt_hash") or stored.get("original_data_hash") != row.get(
                "original_data_hash"
            ):
                changed_codes.append(str(row["code"]))

        if changed_codes:
            _report_changed_source_text(changed_codes)

        if not new_rows:
            cprint(f"Nothing to add — [bold yellow]{db_name}[/bold yellow] is already up to date.")
            return

        if settings.dry_run:
            cprint(f"[bold yellow]Dry run:[/bold yellow] would add {len(new_rows)} row(s) to '{db_path}'")
            return

        # Continue the existing ID sequence: stored rows keep their IDs, and
        # with them the results `run` has already written against those IDs.
        first_id = dao.max_row_id() + 1
        for offset, row in enumerate(new_rows):
            row["ID"] = first_id + offset

        by_provider: dict = defaultdict(list)
        for row in new_rows:
            by_provider[str(row["provider_name"]).lower()].append(row)

        for provider, provider_rows in by_provider.items():
            dao.insert_rows(provider, provider_rows)

        cprint(
            f"Added [bold green]{len(new_rows)}[/bold green] row(s) across "
            f"[bold green]{len(by_provider)}[/bold green] provider table(s) → "
            f"[bold yellow]{db_name}[/bold yellow]"
        )


def _report_no_rows_to_run(active_filters: list[str], code: str | None) -> None:
    """Explain an empty row set for ``run``.

    A filter that matches nothing is a normal outcome and only warns, while a
    ``--code`` that matches nothing names a row that does not exist and raises.
    """

    if active_filters:
        cprint(
            f"[bold yellow]Warning:[/bold yellow] No rows found for " f"{', '.join(active_filters)} — nothing to run."
        )
        return
    if code is not None:
        raise LLMExerException(f"No experiment row found with code '{code}'.")
    cprint("[bold yellow]Warning:[/bold yellow] Experiment database is empty — " "nothing to run.")


@app.command()
def run(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Path to a specific experiment_*.db database. Defaults to the newest one. "
        "Results are written back into the same database.",
    ),
    filter_provider: str = typer.Option(
        None,
        "--filter-provider",
        help="Only run rows whose provider matches this value (case-insensitive). " "E.g. --filter-provider ollama",
    ),
    filter_model: str = typer.Option(
        None,
        "--filter-model",
        help="Only run rows whose model_name matches this value in full (case-sensitive). "
        "E.g. --filter-model gemma4:31b",
    ),
    filter_profile: str = typer.Option(
        None,
        "--filter-profile",
        help="Only run rows whose profile_name matches this value in full (case-sensitive). "
        "E.g. --filter-profile ollama-default",
    ),
    code: str = typer.Option(
        None,
        "--code",
        help="Run only a single combination by its `code` (or numeric ID) instead of all rows.",
    ),
) -> None:
    """Run rows from the generated experiment database and save results"""

    pid = get_proper_pid(pid)
    db_path, experiment_subdir_path = _resolve_experiment_db(pid, file)

    # Stored model/profile values are stripped by `generate`, so a stray space
    # around a shell argument must not turn into a silent no-match.
    filter_model = filter_model.strip() if filter_model is not None else None
    filter_profile = filter_profile.strip() if filter_profile is not None else None
    active_filters = [
        f"{label} '{value}'"
        for label, value in (
            ("provider", filter_provider),
            ("model", filter_model),
            ("profile", filter_profile),
        )
        if value is not None
    ]

    # Lazy import to keep openai optional
    try:
        import llmexer.base.llm_provider  # noqa: F401  (validates LLM deps importable)
        from llmexer.base.llm_manager import (
            build_response_payload,
            result_values,
            run_experiment_row,
        )
    except ImportError as exc:
        raise LLMExerException(
            "Missing required packages for LLM calls. " "Install them with: pip install openai pydantic"
        ) from exc

    cprint(f"Using experiment database: [bold yellow]{os.path.basename(db_path)}[/bold yellow]")

    with ExperimentDAO(db_path) as dao:
        rows = dao.fetch_rows(
            provider=filter_provider,
            id_experiment=code,
            model_name=filter_model,
            profile_name=filter_profile,
        )

        if not rows:
            _report_no_rows_to_run(active_filters, code)
            return

        responses_dir = os.path.join(experiment_subdir_path, DIR_RESPONSES)
        if not settings.dry_run:
            ensure_directory_exists(responses_dir)
            cprint(f"JSON responses will be saved into: [bold yellow]{responses_dir}[/bold yellow]")

        if active_filters:
            cprint(f"Active filter(s): [bold yellow]{', '.join(active_filters)}[/bold yellow]")

        total_runs = len(rows)
        cprint(f"Total experiments to run: [bold green]{total_runs}[/bold green]")

        ran = 0
        for index, row in enumerate(rows):
            current_run_info = f"[[green]{index+1}[/green]/[cyan]{total_runs}[/cyan]]"
            experiment_info = f"[[yellow]{row['code']}[/yellow]]"
            run_info_prefix = f"{current_run_info}{experiment_info}"

            # Skip rows that already completed successfully on an earlier run.
            if row.get("status") == "success":
                cprint(f"{run_info_prefix} [bold yellow]skipped[/bold yellow] (already success)")
                continue

            cprint(f"{run_info_prefix} running")
            provider = row.get("_provider") or str(row["provider_name"]).lower()

            if settings.dry_run:
                continue

            experiment = run_experiment_row(row)
            status = experiment.status

            json_payload = build_response_payload(experiment, provider)

            # Save individual JSON response (kept alongside the DB row).
            file_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            safe_model = str(row["model_name"]).replace("/", "-").replace(":", "-")
            json_path = os.path.join(responses_dir, f"{file_ts}_{safe_model}_{provider}.json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_payload, f, indent=2, ensure_ascii=False)

            dao.update_result(provider, row["ID"], result_values(experiment, provider))
            ran += 1

            status_color = "green" if status == "success" else "red"
            run_status_info = f"[bold {status_color}]{status} [/bold {status_color}]"
            cprint(f"{run_info_prefix} finished {run_status_info}")

        if not settings.dry_run:
            cprint(
                f"Saved [bold green]{ran}[/bold green] result(s) → "
                f"[bold yellow]{os.path.basename(db_path)}[/bold yellow]"
            )


def _resolve_try_data_row(data_df: pd.DataFrame, data_id: str, pid: str) -> dict:
    """Look up one row of ``data.csv`` by its ``ID``.

    Raises:
        LLMExerException: if the file has no ``ID`` column, or if no row carries
            this ID -- the available IDs are named so the typo is obvious.
    """

    if "ID" not in data_df.columns:
        raise LLMExerException(f"'{FILE_DATA}' of project '{pid}' is missing the required 'ID' column.")

    lookup = {_key_part(key): value for key, value in data_df.set_index("ID").to_dict(orient="index").items()}
    if data_id in lookup:
        return lookup[data_id]

    known = list(lookup)
    preview = ", ".join(known[:10]) + (f" (and {len(known) - 10} more)" if len(known) > 10 else "")
    raise LLMExerException(
        f"No row with ID '{data_id}' in {FILE_DATA} of project '{pid}'."
        + (f" Available ID(s): {preview}." if known else f" {FILE_DATA} has no rows.")
    )


def _resolve_try_profile(params_df: pd.DataFrame, profile: str, model: str | None, provider: str | None) -> tuple:
    """Resolve a profile name to exactly one row of ``llm-params.csv``.

    A profile name alone is not an identity: ``llm-params.csv`` is keyed by
    ``(provider, model_name, profile_name)``, so the same profile name may cover
    several models. ``--model`` and ``--provider`` narrow the candidates; an
    ambiguous profile is reported with its candidates rather than guessed at.

    Returns:
        tuple: ``((provider, model_name, profile_name), param_row)``.
    """

    candidates = []
    for _, param_row in params_df.iterrows():
        key = _row_join_key(param_row)
        row_provider, row_model, row_profile = key
        if row_profile != profile:
            continue
        if model is not None and row_model != model:
            continue
        if provider is not None and row_provider != provider:
            continue
        candidates.append((key, param_row))

    if len(candidates) == 1:
        return candidates[0]

    if not candidates:
        known = sorted({_key_part(value) for value in params_df["profile_name"]} - {""})
        narrowed = [f"{label} '{value}'" for label, value in (("model", model), ("provider", provider)) if value]
        raise LLMExerException(
            f"No profile '{profile}'"
            + (f" with {' and '.join(narrowed)}" if narrowed else "")
            + f" in {FILE_LLM_PARAMS}. Profiles are matched in full and case-sensitively."
            + (f" Available profile(s): {', '.join(known)}." if known else "")
        )

    for (row_provider, row_model, _), _param_row in candidates:
        cprint(f"[bold red]Error:[/bold red] candidate: {row_provider} / {row_model}")
    raise LLMExerException(
        f"Profile '{profile}' matches {len(candidates)} rows of {FILE_LLM_PARAMS}. "
        "Narrow it with --model (and --provider if the same model is served by several providers)."
    )


def _print_try_header(model_name: str, provider: str, usage_tokens: Any = None) -> None:
    """Print the ``model`` / ``provider`` / ``usage_tokens`` header of a try."""

    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="bold white", no_wrap=True)
    header.add_column()
    header.add_row("Model:", f"[bold yellow]{model_name}[/bold yellow]")
    header.add_row("Provider:", f"[bold yellow]{provider}[/bold yellow]")
    header.add_row("Usage tokens:", f"[bold green]{usage_tokens if usage_tokens is not None else '-'}[/bold green]")
    console.print(header)


@app.command(name="try")
def try_one(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    prompt: str = typer.Option(
        ...,
        "--prompt",
        help="Prompt template from prompts/ to render (the '.txt' extension is optional).",
    ),
    profile: str = typer.Option(
        ...,
        "--profile",
        help="Profile name from llm-params.csv, matched in full (case-sensitive).",
    ),
    data_id: str = typer.Option(
        ...,
        "--data-id",
        help="ID of the data.csv row to render the prompt with.",
    ),
    model: str = typer.Option(
        None,
        "--model",
        help="Model name, needed only when one profile name covers several models.",
    ),
    provider: str = typer.Option(
        None,
        "--provider",
        help="Provider, needed only when one profile and model are served by several providers.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Experiment database to record the try in. Defaults to the newest one.",
    ),
) -> None:
    """Performs a single experiment `try` run, where custom data x prompt x profile combination is tested once

    A try is `generate` and `run` for a single combination: the three names are
    validated first, the combination is then put together, executed against its
    provider, and then appended to the ``try_experiment_<provider>`` and
    ``try_param_<provider>`` tables of the database.
    """

    pid = get_proper_pid(pid)
    db_path, experiment_subdir_path = _resolve_experiment_db(pid, file)

    prompt = prompt.strip()
    profile = profile.strip()
    data_id = data_id.strip()
    model = model.strip() if model is not None else None
    provider = provider.strip() if provider is not None else None

    _models_df, data_df, _mapping_df, params_df, prompts_subdir = _load_experiment_inputs(pid, experiment_subdir_path)

    # Everything is resolved before a single token is sent: a typo in any of the
    # three names must fail immediately, not after an LLM call.
    prompt_id = _resolve_prompt_ids(prompts_subdir, [prompt])[0]
    data_row = _resolve_try_data_row(data_df, data_id, pid)
    key, param_row = _resolve_try_profile(params_df, profile, model, provider)
    provider_name, model_name, _profile_name = key

    with open(os.path.join(prompts_subdir, f"{prompt_id}.txt"), "r", encoding="utf-8") as f:
        template_str = f.read()

    rendered = _render_prompt(template_str, data_id, data_row)
    row = _combination_row(data_id, prompt_id, key, param_row, rendered)

    if settings.dry_run:
        _print_try_header(model_name, provider_name)
        cprint(
            f"[bold yellow]Dry run:[/bold yellow] would run [bold yellow]{row['code']}[/bold yellow] "
            f"and append it to '{os.path.basename(db_path)}'. \n\nRendered prompt:\n"
        )
        cprint(f'[green]{row["prompt"]}[/green]')
        return

    # Lazy import to keep openai optional
    try:
        import llmexer.base.llm_provider  # noqa: F401  (validates LLM deps importable)
        from llmexer.base.llm_manager import (
            build_response_payload,
            result_values,
            run_experiment_row,
        )
    except ImportError as exc:
        raise LLMExerException(
            "Missing required packages for LLM calls. " "Install them with: pip install openai pydantic"
        ) from exc

    # Opened before the call so a database this version cannot write to aborts
    # the try instead of throwing the response away afterwards.
    with ExperimentDAO(db_path) as dao:
        cprint(f"Trying [bold yellow]{row['code']}[/bold yellow] against [bold yellow]{provider_name}[/bold yellow]")
        experiment = run_experiment_row(row)
        responses_dir = os.path.join(experiment_subdir_path, DIR_RESPONSES)
        ensure_directory_exists(responses_dir)
        file_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_model = str(model_name).replace("/", "-").replace(":", "-")
        json_path = os.path.join(responses_dir, f"{file_ts}_{safe_model}_{provider_name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(build_response_payload(experiment, provider_name), f, indent=2, ensure_ascii=False)

        # A failed try is still a try: it is appended with its error status, so
        # the table stays the full history of what was attempted.
        try_id = dao.append_try_row(provider_name, {**row, **result_values(experiment, provider_name)})

    _print_try_header(model_name, provider_name, experiment.usage_tokens)

    # Printed for a failed try as well: the prompt that was sent is the first
    # thing to look at when the answer is missing.
    cprint("\nRendered prompt:\n")
    cprint(f"[italic yellow]{row['prompt']}[/italic yellow]")

    if experiment.status == "success":
        cprint("\nResponse text:\n")
        cprint(f"[green]{experiment.response_text}[/green]")
    else:
        cprint(f"[bold red]{experiment.status}[/bold red]")

    cprint(
        f"\nTry [bold green]{try_id}[/bold green] appended to "
        f"[bold yellow]{os.path.basename(db_path)}[/bold yellow] "
        f"({', '.join((try_table_name_for(provider_name), try_params_table_name_for(provider_name)))}); "
        f"response saved to [bold yellow]{os.path.basename(json_path)}[/bold yellow]"
    )


@app.command()
def stats(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Experiment database to read stats from. Defaults to the single experiment_*.db.",
    ),
) -> None:
    """Show aggregate statistics for a project's experiment results

    Defaults to the single ``experiment_*.db`` produced by ``experiment
    generate``/``run``; pass ``--file`` to inspect a specific database instead.
    """

    pid = get_proper_pid(pid)
    if file is not None:
        db_path, _ = _resolve_experiment_db(pid, file)
    else:
        experiment_subdir_path = get_experiment_subdir_path(pid)
        db_files = _find_db_files(experiment_subdir_path)
        if not db_files:
            raise LLMExerException(
                f"No experiment database found for project '{pid}'. " f"Run `experiment generate --pid {pid}` first."
            )
        if len(db_files) > 1:
            joined = ", ".join(db_files)
            raise LLMExerException(
                f"Multiple experiment databases found for project '{pid}': {joined}. " f"Pass --file to choose one."
            )
        db_path = os.path.join(experiment_subdir_path, db_files[0])

    with ExperimentDAO(db_path) as dao:
        data = dao.stats()

    summary = Table(title=f"Experiment stats — {pid}")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")
    for key in ("total", "finished", "running", "errors", "total_tokens"):
        summary.add_row(key, str(data[key]))
    console.print(summary)

    providers = data["providers"]
    if providers:
        table = Table(title="Providers")
        table.add_column("Provider", style="cyan")
        table.add_column("requests", justify="right", style="green")
        for name, count in providers.items():
            table.add_row(name, str(count))
        console.print(table)

    models = data["models"]
    if models:
        table = Table(title="Models")
        table.add_column("Model", style="cyan")
        table.add_column("requests", justify="right", style="green")
        table.add_column("finished", justify="right", style="green")
        table.add_column("open", justify="right", style="green")
        table.add_column("time total", justify="right", style="green")
        table.add_column("average time", justify="right", style="green")
        table.add_column("tokens", justify="right", style="green")
        for name, agg in models.items():
            table.add_row(
                name,
                str(agg["requests"]),
                str(agg["finished"]),
                str(agg["open"]),
                _format_hms(agg["elapsed_seconds"]),
                _format_hms(agg["avg_elapsed_seconds"]),
                str(agg["tokens"]),
            )
        console.print(table)


@app.command(name="list")
def list_experiments(
    sort_by: SortBy = typer.Option(
        SortBy.alpha,
        "--sort-by",
        help="Sort projects by 'alpha' (alphabetical) or 'date' (creation date).",
    ),
    desc: bool = typer.Option(False, "--desc", help="Sort in descending order."),
) -> None:
    """List all projects with their initialization state and generated experiments"""
    if not os.path.exists(PROJECTS_PATH):
        cprint("No projects found.")
        return

    entries = [e for e in os.scandir(PROJECTS_PATH) if e.is_dir()]

    if not entries:
        cprint("No projects found.")
        return

    if sort_by == SortBy.date:
        entries.sort(key=lambda e: e.stat().st_ctime, reverse=desc)
    else:
        entries.sort(key=lambda e: e.name, reverse=desc)

    table = Table()
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Name", style="cyan")
    table.add_column("Created", style="cyan", no_wrap=True)
    table.add_column("Initialized", justify="center", style="red", no_wrap=True)
    table.add_column("Experiments", style="white")

    current_pid = settings.project_id
    experiment_file_to_run = ""
    for i, entry in enumerate(entries, start=1):
        ctime = datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        is_initialized = _is_experiment_initialized(entry.path)
        init_display = "[green]Yes[/green]" if is_initialized else "[dim]No[/dim]"

        generated_files = _get_generated_experiment_files(entry.path)
        files_display = "\n".join(generated_files) if generated_files else "[dim]-[/dim]"
        files_display_plain = "\n".join(generated_files) if generated_files else "-"

        # Check if this is the current project
        is_current = current_pid and entry.name == current_pid

        if is_current:
            # Bold yellow for current project, underline only on counter
            table.add_row(
                f"[bold underline yellow]{i}[/bold underline yellow]",
                f"[bold yellow]{entry.name}[/bold yellow]",
                f"[bold yellow]{ctime}[/bold yellow]",
                f"[bold yellow]{'Yes' if is_initialized else 'No'}[/bold yellow]",
                f"[bold yellow]{files_display_plain}[/bold yellow]",
            )
            if len(generated_files) > 0:
                experiment_file_to_run = generated_files[-1]
        else:
            table.add_row(str(i), entry.name, ctime, init_display, files_display)

    console.print(table)
    cprint("\nExample to run an experiment:")
    cprint(f"[bold yellow]llmexer experiment run --file {experiment_file_to_run}[/bold yellow]")
