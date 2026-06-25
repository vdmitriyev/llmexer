"""Experiment group commands of the CLI interface."""

import hashlib
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd
import typer
from jinja2 import BaseLoader, DebugUndefined, Environment
from rich.table import Table

from llmexer.base.dao import ExperimentDAO, latest_db, list_db_files, next_db_filename
from llmexer.base.experiment import (
    _PARAM_COLUMNS,
    DIR_EXPERIMENT,
    DIR_RESPONSES,
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


def _find_db_files(experiment_subdir_path: str) -> list[str]:
    """List experiment SQLite databases (``experiment*.db``) in the subdir."""

    return list_db_files(experiment_subdir_path)


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
                f"No experiment database found for project '{pid}'. "
                f"Run `experiment generate --pid {pid}` first."
            )
        return db_path, experiment_subdir_path

    db_path = (
        file if os.path.isabs(file) else os.path.join(experiment_subdir_path, file)
    )

    if not os.path.exists(db_path):
        raise LLMExerException(f"Experiment database not found: '{db_path}'.")

    return db_path, experiment_subdir_path


def _next_data_backup_name(folder: str) -> str:
    """Return the next ``data_backup_<YYYYMMDD>_<NN>.csv`` name for ``folder``.

    ``<NN>`` is a zero-padded counter, one greater than the highest counter among
    today's existing ``data_backup_<today>_*.csv`` files (starts at ``01``).
    """

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"data_backup_{date}_"
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


def _write_data_csv(experiment_subdir_path: str, df: pd.DataFrame) -> tuple[str, str]:
    """Write ``df`` to ``experiment/data.csv``, backing up any existing file first.

    Returns ``(data_path, backup_name)`` where ``backup_name`` is ``""`` when no
    prior ``data.csv`` existed.
    """

    data_path = os.path.join(experiment_subdir_path, "data.csv")
    backup_name = ""
    if os.path.exists(data_path):
        backup_name = _next_data_backup_name(experiment_subdir_path)
        shutil.copy2(data_path, os.path.join(experiment_subdir_path, backup_name))
    df.to_csv(data_path, index=False, sep=";", encoding="utf-8")
    return data_path, backup_name


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
    prompts_subdir = os.path.join(experiment_subdir_path, "prompts")
    ensure_directory_exists(experiment_subdir_path)
    ensure_directory_exists(prompts_subdir)

    # llm-models.csv
    models_path = os.path.join(experiment_subdir_path, "llm-models.csv")
    with open(models_path, "w", encoding="utf-8") as f:
        f.write("name;provider;notes\n")
        f.write("gemma4:31b;ollama;local model\n")
        f.write("phi4:14b;ollama;local model\n")

    # data.csv
    data_path = os.path.join(experiment_subdir_path, "data.csv")
    with open(data_path, "w", encoding="utf-8") as f:
        f.write("ID;Title;Abstract\n")
        f.write(
            "D01;Sample Paper Title One;This is the abstract of the first sample paper.\n"
        )
        f.write(
            "D02;Sample Paper Title Two;This is the abstract of the second sample paper.\n"
        )

    # mapping.csv
    mapping_path = os.path.join(experiment_subdir_path, "mapping.csv")
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
    llm_params_path = os.path.join(experiment_subdir_path, "llm-params.csv")
    with open(llm_params_path, "w", encoding="utf-8") as f:
        f.write(
            "provider;model_name;profile_name;temperature;top_p;max_tokens;"
            "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
        )
        f.write("ollama;gemma4:31b;ollama-default;0.7;1.0;512;4096;1.1;;;;\n")
        f.write("ollama;phi4:14b;ollama-creative;1.2;0.95;512;4096;1.0;;;;\n")
        f.write("openai;gpt-4o;openai-default;0.7;1.0;512;;;;42;\n")
        f.write("vllm;meta-llama/Llama-3-8b;vllm-default;0.7;0.9;512;;;0.05;1;;\n")
        f.write("gemini;gemini-2.0-flash;gemini-default;0.7;1.0;512;;;;;standard\n")

    cprint(f"Init project [bold yellow]{pid}[/bold yellow] with standard structure.")


@app.command(name="copy-papers")
def copy_papers(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Copy parsed papers (.md/.txt) from the project's papers/ folder into data.csv.

    Each parsed paper becomes a row ``ID;filename;content`` with IDs ``P01``,
    ``P02``, … ordered alphabetically by filename. When a paper has both a ``.md``
    and a ``.txt`` the Markdown is preferred. An existing ``data.csv`` is backed
    up first.
    """

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)
    papers_path = os.path.join(get_project_directory_path(pid), PAPERS_DIR)

    if not os.path.isdir(papers_path):
        raise LLMExerException(
            f"No papers folder found for project '{pid}': '{papers_path}'."
        )

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

    backup_note = (
        f" (backed up previous data.csv → {backup_name})" if backup_name else ""
    )
    cprint(
        f"Copied [bold green]{len(rows)}[/bold green] paper(s) → "
        f"[bold yellow]data.csv[/bold yellow]{backup_note}"
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
        help="Search results CSV to copy from (absolute, or relative to the "
        "project's searches/ folder).",
    ),
) -> None:
    """Copy a search results file into data.csv.

    Writes rows ``ID;Title;Abstract;doi;authors`` with IDs ``S01``, ``S02``, …
    preserving the source file's row order. An existing ``data.csv`` is backed
    up first.
    """

    pid = get_proper_pid(pid)
    experiment_subdir_path = get_experiment_subdir_path(pid)

    search_path = (
        file
        if os.path.isabs(file)
        else os.path.join(get_project_directory_path(pid), SEARCHES_DIR, file)
    )
    if not os.path.exists(search_path):
        raise LLMExerException(f"Search file not found: '{search_path}'.")

    search_df = pd.read_csv(search_path, sep=";", encoding="utf-8")
    required = ["title", "abstract", "doi", "authors"]
    missing = [c for c in required if c not in search_df.columns]
    if missing:
        raise LLMExerException(
            f"Search file '{search_path}' is missing required column(s): "
            f"{', '.join(missing)}."
        )

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

    backup_note = (
        f" (backed up previous data.csv → {backup_name})" if backup_name else ""
    )
    cprint(
        f"Copied [bold green]{len(df)}[/bold green] search result(s) → "
        f"[bold yellow]data.csv[/bold yellow]{backup_note}"
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

    models_path = os.path.join(experiment_subdir_path, "llm-models.csv")
    data_path = os.path.join(experiment_subdir_path, "data.csv")
    mapping_path = os.path.join(experiment_subdir_path, "mapping.csv")
    llm_params_path = os.path.join(experiment_subdir_path, "llm-params.csv")
    prompts_subdir = os.path.join(experiment_subdir_path, "prompts")

    for label, path in [
        ("llm-models.csv", models_path),
        ("data.csv", data_path),
        ("mapping.csv", mapping_path),
        ("llm-params.csv", llm_params_path),
        ("prompts/", prompts_subdir),
    ]:
        if not os.path.exists(path):
            raise LLMExerException(
                f"Required file or directory not found for project '{pid}': {label}"
            )

    models_df = pd.read_csv(models_path, sep=";", encoding="utf-8")
    data_df = pd.read_csv(data_path, sep=";", encoding="utf-8")
    mapping_df = pd.read_csv(mapping_path, sep=";", encoding="utf-8")
    params_df = pd.read_csv(llm_params_path, sep=";", encoding="utf-8")

    if params_df.empty:
        cprint(
            "[bold yellow]Warning:[/bold yellow] llm-params.csv is empty — no rows will be generated."
        )
        return

    data_lookup = data_df.set_index("ID").to_dict(orient="index")
    env = Environment(loader=BaseLoader(), undefined=DebugUndefined)

    rows = []
    row_counter = 1

    for _, mapping_row in mapping_df.iterrows():
        data_id = str(mapping_row["data_id"]).strip()
        prompt_id = str(mapping_row["prompt_id"]).strip()

        if data_id not in data_lookup:
            cprint(
                f"[bold yellow]Warning:[/bold yellow] data_id '{data_id}' not found in data.csv — skipping."
            )
            continue

        data_row = data_lookup[data_id]
        context = {k.lower(): v for k, v in data_row.items()}
        context["id"] = data_id

        original_data_dict = {"ID": data_id, **data_row}
        original_data_str = json.dumps(original_data_dict, ensure_ascii=False)
        original_data_hash = hashlib.sha256(
            original_data_str.encode("utf-8")
        ).hexdigest()

        prompt_file_path = os.path.join(prompts_subdir, f"{prompt_id}.txt")
        if not os.path.exists(prompt_file_path):
            cprint(
                f"[bold yellow]Warning:[/bold yellow] prompt file '{prompt_id}.txt' not found — skipping."
            )
            continue

        with open(prompt_file_path, "r", encoding="utf-8") as f:
            template_str = f.read()

        rendered_prompt = env.from_string(template_str).render(**context)
        prompt_hash = hashlib.sha256(rendered_prompt.encode("utf-8")).hexdigest()

        for _, model_row in models_df.iterrows():
            for _, param_row in params_df.iterrows():
                if model_row["name"] == param_row["model_name"]:
                    rows.append(
                        {
                            "ID": row_counter,
                            "code": f"{data_id}_{prompt_id}_{str(model_row['name'])}_{param_row['profile_name']}",
                            "prompt": rendered_prompt,
                            "tokens_estimate": len(rendered_prompt) // 4,
                            "original_data": original_data_str,
                            "model_name": str(model_row["name"]),
                            "provider_name": str(model_row["provider"]),
                            "prompt_hash": prompt_hash,
                            "original_data_hash": original_data_hash,
                            **{k: param_row.get(k) for k in _PARAM_COLUMNS},
                        }
                    )
                    row_counter += 1

    if not rows:
        cprint(
            "[bold yellow]Warning:[/bold yellow] No rows were generated. Check your mapping.csv and data.csv."
        )
        return

    # Sort by model order (stable -> preserves mapping order within a model),
    # then renumber IDs 1..N across the whole generation.
    model_order = list(models_df["name"].astype(str))

    def _model_rank(row: dict) -> int:
        name = str(row["model_name"])
        return model_order.index(name) if name in model_order else len(model_order)

    rows.sort(key=_model_rank)
    for new_id, row in enumerate(rows, start=1):
        row["ID"] = new_id

    output_filename = next_db_filename(experiment_subdir_path)
    output_path = os.path.join(experiment_subdir_path, output_filename)

    if settings.dry_run:
        cprint(
            f"[bold yellow]Dry run:[/bold yellow] would write {len(rows)} row(s) to '{output_path}'"
        )
        return

    # Each provider gets its own table holding only its parameters.
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
        help="Only run rows whose provider matches this value (case-insensitive). "
        "E.g. --filter-provider ollama",
    ),
    id: str = typer.Option(
        None,
        "--id",
        help="Run only a single combination by its ID (or code) instead of all rows.",
    ),
) -> None:
    """Run rows from the generated experiment database and save results"""

    pid = get_proper_pid(pid)
    db_path, experiment_subdir_path = _resolve_experiment_db(pid, file)

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
            "Missing required packages for LLM calls. "
            "Install them with: pip install openai pydantic"
        ) from exc

    cprint(
        f"Using experiment database: [bold yellow]{os.path.basename(db_path)}[/bold yellow]"
    )

    with ExperimentDAO(db_path) as dao:
        rows = dao.fetch_rows(provider=filter_provider, id_experiment=id)

        if not rows:
            if filter_provider is not None:
                cprint(
                    f"[bold yellow]Warning:[/bold yellow] No rows found for provider "
                    f"'{filter_provider}' — nothing to run."
                )
                return
            if id is not None:
                raise LLMExerException(f"No experiment row found with id '{id}'.")
            cprint(
                "[bold yellow]Warning:[/bold yellow] Experiment database is empty — "
                "nothing to run."
            )
            return

        responses_dir = os.path.join(experiment_subdir_path, DIR_RESPONSES)
        if not settings.dry_run:
            ensure_directory_exists(responses_dir)
            cprint(
                f"JSON responses will be saved into: [bold yellow]{responses_dir}[/bold yellow]"
            )

        total_runs = len(rows)
        cprint(f"Total experiments to run: [bold green]{total_runs}[/bold green]")

        ran = 0
        for index, row in enumerate(rows):
            current_run_info = f"[[green]{index+1}[/green]/[cyan]{total_runs}[/cyan]]"
            experiment_info = f"[[yellow]{row['code']}[/yellow]]"
            run_info_prefix = f"{current_run_info}{experiment_info}"

            # Skip rows that already completed successfully on an earlier run.
            if row.get("status") == "success":
                cprint(
                    f"{run_info_prefix} [bold yellow]skipped[/bold yellow] (already success)"
                )
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
            json_path = os.path.join(
                responses_dir, f"{file_ts}_{safe_model}_{provider}.json"
            )
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
    """Show aggregate statistics for a project's experiment results.

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
                f"No experiment database found for project '{pid}'. "
                f"Run `experiment generate --pid {pid}` first."
            )
        if len(db_files) > 1:
            joined = ", ".join(db_files)
            raise LLMExerException(
                f"Multiple experiment databases found for project '{pid}': {joined}. "
                f"Pass --file to choose one."
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

    for label, key in (("Providers", "providers"), ("Models", "models")):
        breakdown = data[key]
        if not breakdown:
            continue
        table = Table(title=label)
        table.add_column(label.rstrip("s"), style="cyan")
        table.add_column("Count", justify="right", style="green")
        for name, count in breakdown.items():
            table.add_row(name, str(count))
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
        ctime = datetime.fromtimestamp(entry.stat().st_ctime, tz=timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        is_initialized = _is_experiment_initialized(entry.path)
        init_display = "[green]Yes[/green]" if is_initialized else "[dim]No[/dim]"

        generated_files = _get_generated_experiment_files(entry.path)
        files_display = (
            "\n".join(generated_files) if generated_files else "[dim]-[/dim]"
        )
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
    cprint(
        f"[bold yellow]llmexer experiment run --file {experiment_file_to_run}[/bold yellow]"
    )
