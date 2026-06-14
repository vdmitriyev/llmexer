"""Experiment group commands of the CLI interface."""

import hashlib
import json
import os
from datetime import datetime, timezone
from enum import Enum

import pandas as pd
import typer
from jinja2 import BaseLoader, DebugUndefined, Environment
from rich.table import Table

from llmexer.base.experiment import (
    _OUTPUT_COLUMNS,
    _PARAM_COLUMNS,
    DIR_EXPERIMENT,
    DIR_RESPONSES,
    _get_generated_experiment_files,
    _is_experiment_initialized,
    generate_experiment_id,
)
from llmexer.common import (
    ensure_directory_exists,
    get_experiment_directory_path,
    get_proper_eid,
)
from llmexer.configs import console, cprint, settings
from llmexer.constants import EXPERIMENTS_PATH
from llmexer.exceptions import ExperimentAlreadyExistsException, LLMExerException

app = typer.Typer(help="Manage LLM experiments.")


class SortBy(str, Enum):
    alpha = "alpha"
    date = "date"


def _resolve_experiment_csv(eid: str, file: str) -> tuple[str, str]:
    """Resolve and validate a generated experiment CSV path for an experiment.

    Returns ``(file_path, experiment_subdir_path)``. Raises ``LLMExerException``
    if the experiment is not initialised, no file is given, or the file is
    missing.
    """

    experiment_path = get_experiment_directory_path(eid)
    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    if not os.path.exists(experiment_subdir_path):
        raise LLMExerException(
            f"Experiment '{eid}' has not been initialised. "
            f"Run `experiment init --eid {eid}` first."
        )

    if file is None:
        raise LLMExerException(
            f"No experiment CSV provided. Run `experiment generate --eid {eid}` first."
        )
    file_path = (
        file if os.path.isabs(file) else os.path.join(experiment_subdir_path, file)
    )

    if not os.path.exists(file_path):
        raise LLMExerException(f"Experiment CSV not found: '{file_path}'.")

    return file_path, experiment_subdir_path


@app.command()
def create(
    id: str = typer.Option(
        None,
        "--id",
        help="Custom experiment ID. If not provided, one is auto-generated.",
    )
) -> None:
    """Create a new experiment folder under .experiments"""
    experiment_id = id if id else generate_experiment_id()
    experiment_path = os.path.join(EXPERIMENTS_PATH, experiment_id)

    if os.path.exists(experiment_path):
        raise ExperimentAlreadyExistsException(
            f"Experiment '{experiment_id}' already exists."
        )

    ensure_directory_exists(experiment_path)
    cprint(f"Created experiment: [bold yellow]{experiment_id}[/bold yellow]")


@app.command(name="list")
def list_experiments(
    sort_by: SortBy = typer.Option(
        SortBy.alpha,
        "--sort-by",
        help="Sort experiments by 'alpha' (alphabetical) or 'date' (creation date).",
    ),
    desc: bool = typer.Option(False, "--desc", help="Sort in descending order."),
) -> None:
    """List all experiments in the experiments folder"""
    if not os.path.exists(EXPERIMENTS_PATH):
        cprint("No experiments found.")
        return

    entries = [e for e in os.scandir(EXPERIMENTS_PATH) if e.is_dir()]

    if not entries:
        cprint("No experiments found.")
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

    current_eid = settings.experiment_id
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

        # Check if this is the current experiment
        is_current = current_eid and entry.name == current_eid

        if is_current:
            # Bold yellow for current experiment, underline only on counter
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


@app.command()
def rename(
    old_id: str = typer.Option(
        None,
        "--old-id",
        help="Current experiment ID to rename. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    new_id: str = typer.Option(
        ...,
        "--new-id",
        help="New experiment ID name.",
    ),
) -> None:
    """Rename an existing experiment"""

    # Use current experiment if old_id not provided
    if old_id is None:
        if settings.experiment_id:
            old_id = settings.experiment_id
        else:
            raise LLMExerException(
                "No experiment ID provided. Use --old-id or set EXPERIMENT_ID in .env file."
            )

    old_path = os.path.join(EXPERIMENTS_PATH, old_id)
    new_path = os.path.join(EXPERIMENTS_PATH, new_id)

    if not os.path.exists(old_path):
        raise LLMExerException(f"Experiment '{old_id}' does not exist.")

    if os.path.exists(new_path):
        raise ExperimentAlreadyExistsException(f"Experiment '{new_id}' already exists.")

    os.rename(old_path, new_path)
    cprint(
        f"Renamed experiment: [bold yellow]{old_id}[/bold yellow] → [bold yellow]{new_id}[/bold yellow]"
    )


@app.command()
def init(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to initialise. If not provided, uses EXPERIMENT_ID from .env.",
    )
) -> None:
    """Initialise an experiment with a standard folder structure and template files"""

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    # Raise if already initialised
    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    if os.path.exists(experiment_subdir_path):
        raise LLMExerException(f"Experiment '{eid}' has already been initialised.")

    # Create subfolders
    prompts_subdir = os.path.join(experiment_subdir_path, "prompts")
    ensure_directory_exists(experiment_subdir_path)
    ensure_directory_exists(prompts_subdir)

    # models.csv
    models_path = os.path.join(experiment_subdir_path, "models.csv")
    with open(models_path, "w", encoding="utf-8") as f:
        f.write("name;provider;notes\n")
        f.write("llama3.3:latest;ollama;local model\n")
        f.write("phi4:14b;ollama;local model\n")
        f.write("gemma3:12b;ollama;local model\n")
        f.write("gemma3:27b;ollama;local model\n")

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
            "profile_name;model_name;provider;temperature;top_p;max_tokens;"
            "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
        )
        f.write("ollama-default;llama3.3:latest;ollama;0.7;1.0;512;4096;1.1;;;;\n")
        f.write("ollama-creative;llama3.3:latest;ollama;1.2;0.95;512;4096;1.0;;;;\n")
        f.write("openai-default;gpt-4o;openai;0.7;1.0;512;;;;42;\n")
        f.write("vllm-default;meta-llama/Llama-3-8b;vllm;0.7;0.9;512;;;0.05;1;;\n")
        f.write("gemini-default;gemini-2.0-flash;gemini;0.7;1.0;512;;;;;standard\n")

    cprint(f"Init experiment [bold yellow]{eid}[/bold yellow] with standard structure.")


@app.command()
def generate(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to generate prompts for. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Generate rendered prompts for all data-model combinations defined in the experiment"""

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    experiment_subdir_path = os.path.join(experiment_path, DIR_EXPERIMENT)
    if not os.path.exists(experiment_subdir_path):
        raise LLMExerException(
            f"Experiment '{eid}' has not been initialised. Run `experiment init --eid {eid}` first."
        )

    models_path = os.path.join(experiment_subdir_path, "models.csv")
    data_path = os.path.join(experiment_subdir_path, "data.csv")
    mapping_path = os.path.join(experiment_subdir_path, "mapping.csv")
    llm_params_path = os.path.join(experiment_subdir_path, "llm-params.csv")
    prompts_subdir = os.path.join(experiment_subdir_path, "prompts")

    for label, path in [
        ("models.csv", models_path),
        ("data.csv", data_path),
        ("mapping.csv", mapping_path),
        ("llm-params.csv", llm_params_path),
        ("prompts/", prompts_subdir),
    ]:
        if not os.path.exists(path):
            raise LLMExerException(
                f"Required file or directory not found for experiment '{eid}': {label}"
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

    params_df = params_df.rename(
        columns={"model_name": "param_model_name", "provider": "param_provider"}
    )

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
                if model_row["name"] == param_row["param_model_name"]:
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

    result_df = pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
    model_order = list(models_df["name"].astype(str))
    result_df["model_name"] = pd.Categorical(
        result_df["model_name"], categories=model_order, ordered=True
    )
    result_df = result_df.sort_values("model_name").reset_index(drop=True)
    result_df["model_name"] = result_df["model_name"].astype(str)
    result_df["ID"] = range(1, len(result_df) + 1)
    output_filename = f"experiment_{generate_experiment_id()}.csv"
    output_path = os.path.join(experiment_subdir_path, output_filename)

    if settings.dry_run:
        cprint(
            f"[bold yellow]Dry run:[/bold yellow] would write {len(result_df)} row(s) to '{output_path}'"
        )
        cprint(f"Columns: {_OUTPUT_COLUMNS}")
        return

    result_df.to_csv(output_path, index=False, encoding="utf-8", sep=";")
    cprint(
        f"Generated [bold green]{len(result_df)}[/bold green] row(s) → "
        f"[bold yellow]{output_filename}[/bold yellow]"
    )


@app.command()
def current() -> None:
    """Display the current experiment ID loaded from .env"""

    if settings.experiment_id:
        experiment_path = os.path.join(EXPERIMENTS_PATH, settings.experiment_id)
        if os.path.exists(experiment_path):
            cprint(
                f"Current experiment: [bold yellow]{settings.experiment_id}[/bold yellow]"
            )
        else:
            cprint(
                f"Current experiment: [bold yellow]{settings.experiment_id}[/bold yellow] "
                f"[bold red](not found in {EXPERIMENTS_PATH})[/bold red]"
            )
    else:
        cprint(
            "[bold red]No current experiment set.[/bold red] Set EXPERIMENT_ID in .env file."
        )


@app.command()
def run(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Path to a specific experiment_NAME.csv file. Results are written into a single experiment_NAME_results.csv next to it.",
    ),
    filter_provider: str = typer.Option(
        None,
        "--filter-provider",
        help="Only run rows whose param_provider matches this value (case-insensitive). "
        "E.g. --filter-provider ollama",
    ),
    id: str = typer.Option(
        None,
        "--id",
        help="Run only a single combination by its ID (or code) instead of all rows.",
    ),
) -> None:
    """Run all rows in the generated experiment CSV and save results"""

    eid = get_proper_eid(eid)
    file_path, experiment_subdir_path = _resolve_experiment_csv(eid, file)

    # Lazy import to keep openai optional
    try:
        import llmexer.base.llm_provider  # noqa: F401  (validates LLM deps importable)
        from llmexer.base.llm_manager import ExperimentsManager
    except ImportError as exc:
        raise LLMExerException(
            "Missing required packages for LLM calls. "
            "Install them with: pip install openai pydantic"
        ) from exc

    manager = ExperimentsManager()
    prompts_df = manager.load(file_path)

    if prompts_df.empty:
        cprint(
            "[bold yellow]Warning:[/bold yellow] Experiment CSV is empty — nothing to run."
        )
        return

    # Build the run view (which rows to execute) without shrinking manager.df,
    # so the full row set is always persisted to the single results file.
    mask = pd.Series(True, index=prompts_df.index)
    if filter_provider is not None:
        mask &= prompts_df["param_provider"].str.lower() == filter_provider.lower()
        if not mask.any():
            cprint(
                f"[bold yellow]Warning:[/bold yellow] No rows found for provider "
                f"'{filter_provider}' — nothing to run."
            )
            return

    if id is not None:
        id_mask = prompts_df["ID"].astype("string") == str(id)
        if "code" in prompts_df.columns:
            id_mask = id_mask | (prompts_df["code"].astype("string") == str(id))
        mask &= id_mask
        if not mask.any():
            raise LLMExerException(f"No experiment row found with id '{id}'.")

    run_df = prompts_df[mask].reset_index(drop=True)

    # A single, stable results file per experiment (no per-run timestamp).
    results_path = os.path.join(experiment_subdir_path, f"experiment_{eid}_results.csv")
    cprint(
        f"Output results will be saved into: [bold yellow]{results_path}[/bold yellow]"
    )

    if not settings.dry_run:
        responses_dir = os.path.join(experiment_subdir_path, DIR_RESPONSES)
        ensure_directory_exists(responses_dir)
        cprint(
            f"JSON responses will be saved into: [bold yellow]{responses_dir}[/bold yellow]"
        )
        # Retain results from earlier runs for rows not executed this time.
        manager.merge_results(results_path)

    total_runs = len(run_df)
    cprint(f"Total experiments to run: [bold green]{total_runs}[/bold green]")

    for index, p_row in run_df.iterrows():
        current_run_info = f"[[green]{index+1}[/green]/[cyan]{total_runs}[/cyan]]"
        experiment_info = f"[[yellow]{p_row['code']}[/yellow]]"
        run_info_prefix = f"{current_run_info}{experiment_info}"
        cprint(f"{run_info_prefix} running")
        provider = str(p_row["param_provider"]).lower()

        if settings.dry_run:
            continue

        experiment = manager.run(p_row["ID"])
        status = experiment.status

        json_payload: dict = {
            "model": experiment.param_model_name or experiment.model_name,
            "provider": provider,
            "prompt": experiment.prompt,
            "profile": experiment.profile_name,
            "response_text": experiment.response_text,
            "usage_tokens": experiment.usage_tokens,
            "status": status,
            "timestamp": experiment.timestamp,
        }

        # Save individual JSON response
        file_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        safe_model = str(p_row["param_model_name"]).replace("/", "-").replace(":", "-")
        json_path = os.path.join(
            responses_dir, f"{file_ts}_{safe_model}_{provider}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, indent=2, ensure_ascii=False)

        status_color = "green" if status == "success" else "red"
        run_status_info = f"[bold {status_color}]{status} [/bold {status_color}]"
        cprint(f"{run_info_prefix} finished {run_status_info}")

    if not settings.dry_run:
        manager.save_results(results_path)
        output_filename = os.path.basename(results_path)
        cprint(
            f"Saved [bold green]{total_runs}[/bold green] result(s) → "
            f"[bold yellow]{output_filename}[/bold yellow]"
        )


@app.command()
def stats(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Path to a generated experiment_NAME.csv (or its results CSV).",
    ),
) -> None:
    """Show aggregate statistics for a generated experiment CSV."""

    eid = get_proper_eid(eid)
    file_path, _ = _resolve_experiment_csv(eid, file)

    from llmexer.base.llm_manager import ExperimentsManager

    manager = ExperimentsManager()
    manager.load(file_path)
    data = manager.stats()

    summary = Table(title=f"Experiment stats — {eid}")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right", style="green")
    for key in ("total", "completed", "running", "errors", "pending", "total_tokens"):
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
