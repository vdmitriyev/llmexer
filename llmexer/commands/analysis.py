"""Analysis group commands of the CLI interface."""

import os
import shutil
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import typer
from rich.table import Table

from llmexer.base.analysis import COPIED_MODULES, prisma
from llmexer.base.analysis.notebook import (
    ADDON_NOTEBOOKS,
    NOTEBOOKS,
    copy_analysis_modules,
    write_notebook,
)
from llmexer.base.dao import latest_db
from llmexer.base.experiment_layout import (
    ExperimentPaths,
    experiments_dir_name,
    resolve_experiment,
)
from llmexer.common import (
    ensure_directory_exists,
    get_project_directory_path,
    get_proper_pid,
    next_backup_name,
)
from llmexer.configs import console, cprint, settings
from llmexer.constants import ANALYSIS_BACKUP_DIR, ANALYSIS_DIR, SEARCHES_DIR
from llmexer.exceptions import UnexpectedCLIParamsException
from llmexer.version import package_version

app = typer.Typer(help="Helps with analyse of the data by generating ready-to-run Jupyter notebooks.")


def _latest_db_name(paths: ExperimentPaths):
    """Filename of the (sub-)experiment's newest ``experiment_*.db``, or ``None``.

    Only the name travels into the notebook: the notebook joins it onto its own
    ``DIR_EXPERIMENT``, so the project folder stays relocatable.
    """

    newest = latest_db(paths.work_dir)

    return os.path.basename(newest) if newest else None


def _exp_prefix(paths: ExperimentPaths) -> str:
    """``<name>__`` for a sub-experiment, so two of them never share an output file."""

    return "" if paths.name is None else f"{paths.name}__"


def _analysis_paths(pid: str, exp: str | None) -> ExperimentPaths:
    """The (sub-)experiment to analyse; a project without ``experiments/`` yet is fine."""

    paths = resolve_experiment(pid, exp, must_exist=False)
    if paths.name is not None:
        cprint(f"Experiment: [bold yellow]{paths.label()}[/bold yellow]")
    return paths


def _exp_option():
    return typer.Option(
        None,
        "--exp",
        help="Sub-experiment to analyse in a project with several. Defaults to the default one.",
    )


def _search_files(project_path: str) -> list:
    """One entry per search, naming its YAML and the CSVs written from it.

    Each entry is ``{"search": <id>.yaml, "results": ... | None, "filter": ... | None}``:
    a search that has not been run yet has no CSVs, and one that has never been
    filtered has no ``__filtered.csv``. The project's merged CSVs have no YAML of
    their own and are deliberately left out - they carry an extra column per
    search and are a different shape.
    """

    searches_path = Path(os.path.join(project_path, SEARCHES_DIR))
    if not searches_path.is_dir():
        return []

    entries = []
    for yaml_path in sorted(searches_path.glob("*.yaml")):
        search_id = yaml_path.stem
        results = searches_path / f"{search_id}__results.csv"
        filtered = searches_path / f"{search_id}__filtered.csv"
        entries.append(
            {
                "search": yaml_path.name,
                "results": results.name if results.is_file() else None,
                "filter": filtered.name if filtered.is_file() else None,
            }
        )

    return entries


def _plan(analysis_path: str, rewrite: bool, filenames=None) -> list:
    """Decide, per output file, whether it is written, kept or backed up first.

    Notebooks and modules are treated the same way: both may carry a
    researcher's edits, so neither is overwritten without ``--rewrite``, and
    ``--rewrite`` backs the old file up before replacing it. Backups go into
    ``.backup/`` rather than sitting next to the notebooks, so the analysis
    folder keeps showing only the files meant to be opened.
    """

    backup_path = os.path.join(analysis_path, ANALYSIS_BACKUP_DIR)
    names = list(filenames) if filenames is not None else list(NOTEBOOKS.values()) + list(COPIED_MODULES)
    planned = []
    for filename in names:
        path = os.path.join(analysis_path, filename)
        if not os.path.exists(path):
            planned.append((filename, path, "write", ""))
        elif rewrite:
            stem, suffix = os.path.splitext(filename)
            planned.append((filename, path, "rewrite", next_backup_name(backup_path, stem, suffix)))
        else:
            planned.append((filename, path, "keep", ""))

    return planned


def _render_plan_table(planned: list, analysis_path: str) -> Table:
    table = Table(title=f"Analysis workspace: {analysis_path}", border_style="bright_blue")
    table.add_column("File", style="white", no_wrap=True)
    table.add_column("Action", no_wrap=True)
    table.add_column("Note", style="dim")

    styles = {"write": "green", "rewrite": "yellow", "keep": "dim"}
    notes = {
        "write": "created",
        "rewrite": "backed up, then replaced",
        "keep": "already exists - use --rewrite to replace",
    }
    for filename, _, action, backup in planned:
        note = f"backed up to {ANALYSIS_BACKUP_DIR}/{backup}" if backup else notes[action]
        table.add_row(filename, f"[{styles[action]}]{action}[/{styles[action]}]", note)

    return table


@app.command()
def init(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help="Replace existing notebooks and analysis modules, backing each up first.",
    ),
    exp: str = _exp_option(),
) -> None:
    """Scaffold Jupyter notebooks for analysing a project

    Creates ``analysis/`` next to the project's ``experiments/``, ``papers/`` and
    ``searches/`` folders, holding two ready-to-run notebooks and the Python
    modules they call. The modules are copies, so they can be tweaked per
    project; ``--rewrite`` restores the shipped versions and backs up the old
    ones first.

    An experiment does not have to be generated first: the notebooks report an
    empty project rather than failing.
    """

    pid = get_proper_pid(pid)
    project_path = get_project_directory_path(pid)
    analysis_path = os.path.join(project_path, ANALYSIS_DIR)
    paths = _analysis_paths(pid, exp)
    notebooks = _notebook_names(paths)

    planned = _plan(analysis_path, rewrite, list(notebooks.values()) + list(COPIED_MODULES))
    console.print(_render_plan_table(planned, analysis_path))

    to_write = [item for item in planned if item[2] != "keep"]
    if not to_write:
        cprint(
            "[bold yellow]Warning:[/bold yellow] the analysis workspace already exists. "
            "Use --rewrite to replace the notebooks and modules."
        )
        return

    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write {len(to_write)} file(s) into '{analysis_path}'")
        return

    ensure_directory_exists(analysis_path)

    # Back up whatever is being replaced before anything is overwritten.
    backup_path = os.path.join(analysis_path, ANALYSIS_BACKUP_DIR)
    if any(action == "rewrite" for _, _, action, _ in to_write):
        ensure_directory_exists(backup_path)
    for _, path, action, backup in to_write:
        if action == "rewrite":
            shutil.copy2(path, os.path.join(backup_path, backup))

    context = {
        "project_id": pid,
        "db_file": _latest_db_name(paths),
        "exp_name": paths.name,
        "experiments_dir": experiments_dir_name(),
        "csv_prefix": _exp_prefix(paths),
        "searches": _search_files(project_path),
        "package_version": package_version(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    written = []
    pending = {filename for filename, _, action, _ in to_write if action != "keep"}
    for kind, filename in notebooks.items():
        if filename in pending:
            written.append(write_notebook(kind, context, os.path.join(analysis_path, filename)))
    if pending & set(COPIED_MODULES):
        written.extend(copy_analysis_modules(analysis_path))

    cprint(f"Analysis workspace ready — {len(written)} file(s) written:\n  {analysis_path}")

    notebook = os.path.join(analysis_path, notebooks["experiment"])
    cprint("\nNext step is to open Jupyter Notebook file and run all cells.")
    cprint(f"Jupyter Notebook: [bold yellow]{notebook}[/bold yellow]")


def _resolve_fields(fields) -> list:
    """Flatten the repeatable ``--fields`` option into ordered, unique names.

    Each value may itself be a comma-separated list, as ``experiment map
    --prompt`` accepts. Blanks are dropped and duplicates removed, keeping the
    order given.
    """

    names = []
    for value in fields or []:
        names.extend(part.strip() for part in str(value).split(","))

    unique = []
    for name in names:
        if name and name not in unique:
            unique.append(name)

    return unique


def _notebook_names(paths: ExperimentPaths) -> dict:
    """``NOTEBOOKS``, with the experiment notebook prefixed for a sub-experiment.

    The searches notebook covers the whole project, so it keeps its name.
    """

    names = dict(NOTEBOOKS)
    names["experiment"] = f"{_exp_prefix(paths)}{NOTEBOOKS['experiment']}"
    return names


def _flattened_csv_name(paths: ExperimentPaths):
    """Name of the flattened CSV the experiment notebook writes, or ``None``."""

    db_file = _latest_db_name(paths)

    return f"{_exp_prefix(paths)}{Path(db_file).stem}_flattened.csv" if db_file else None


def _print_agreement_inputs(fields: list, autodetect: bool, csv_file, notebook: str) -> None:
    """Show what the notebook is being written with, before anything is written."""

    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="bold white", no_wrap=True)
    header.add_column()

    fields_value = ", ".join(fields) if fields else "(none - detected in the notebook)"
    autodetect_note = " (explicit --fields given)" if fields else ""

    for label, value in (
        ("fields:", fields_value),
        ("autodetect-fields:", f"{autodetect}{autodetect_note}"),
        ("flattened CSV:", csv_file or "(no experiment database yet)"),
        ("notebook:", notebook),
    ):
        header.add_row(label, f"[bold blue]{value}[/bold blue]")
    console.print(header)


@app.command(name="add-agreement")
def add_agreement(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    fields: list[str] = typer.Option(
        None,
        "--fields",
        help="Yes/no field(s) of the flattened CSV to compare. Repeatable, and each value "
        "may itself be a comma-separated list. If omitted, the notebook detects them.",
    ),
    autodetect_fields: bool = typer.Option(
        True,
        "--autodetect-fields/--no-autodetect-fields",
        help="Let the notebook find the yes/no fields when --fields is not given.",
    ),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help="Replace an existing agreements notebook, backing it up first.",
    ),
    exp: str = _exp_option(),
) -> None:
    """Add the agreements notebook: pairwise Cohen's kappa between configurations

    Reads the flattened CSV written by ``analyse_experiment.ipynb`` and reports
    Cohen's kappa for every pair of (provider, model, profile), one row per
    yes/no field. The modules the notebook calls are copied alongside it, so it
    also works on a project where ``analysis init`` never ran.
    """

    selected_fields = _resolve_fields(fields)
    if not selected_fields and not autodetect_fields:
        raise UnexpectedCLIParamsException("Nothing to compare: pass --fields, or leave --autodetect-fields on.")

    pid = get_proper_pid(pid)
    project_path = get_project_directory_path(pid)
    analysis_path = os.path.join(project_path, ANALYSIS_DIR)

    paths = _analysis_paths(pid, exp)
    notebook_name = f"{_exp_prefix(paths)}{ADDON_NOTEBOOKS['agreements']}"
    csv_file = _flattened_csv_name(paths)
    _print_agreement_inputs(selected_fields, autodetect_fields, csv_file, notebook_name)

    planned = _plan(analysis_path, rewrite, [notebook_name] + list(COPIED_MODULES))
    console.print(_render_plan_table(planned, analysis_path))

    to_write = [item for item in planned if item[2] != "keep"]
    if not to_write:
        cprint(
            "[bold yellow]Warning:[/bold yellow] the agreements notebook already exists. "
            "Use --rewrite to replace it."
        )
        return

    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write {len(to_write)} file(s) into '{analysis_path}'")
        return

    ensure_directory_exists(analysis_path)

    # Back up whatever is being replaced before anything is overwritten.
    backup_path = os.path.join(analysis_path, ANALYSIS_BACKUP_DIR)
    if any(action == "rewrite" for _, _, action, _ in to_write):
        ensure_directory_exists(backup_path)
    for _, path, action, backup in to_write:
        if action == "rewrite":
            shutil.copy2(path, os.path.join(backup_path, backup))

    context = {
        "project_id": pid,
        "db_file": _latest_db_name(paths),
        "exp_name": paths.name,
        "experiments_dir": experiments_dir_name(),
        "csv_file": csv_file,
        "fields": selected_fields,
        "autodetect_fields": autodetect_fields,
        "package_version": package_version(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    written = []
    pending = {filename for filename, _, action, _ in to_write if action != "keep"}
    if notebook_name in pending:
        written.append(write_notebook("agreements", context, os.path.join(analysis_path, notebook_name)))
    if pending & set(COPIED_MODULES):
        written.extend(copy_analysis_modules(analysis_path))

    cprint(f"Agreements analysis ready — {len(written)} file(s) written:\n  {analysis_path}")

    notebook = os.path.join(analysis_path, notebook_name)
    cprint("\nNext step is to open Jupyter Notebook file and run all cells.")
    cprint(f"Jupyter Notebook: [bold yellow]{notebook}[/bold yellow]")


# ---------------------------------------------------------------------------
# PRISMA 2020 flow diagram
# ---------------------------------------------------------------------------

PRISMA_SVG = "prisma.svg"
PRISMA_COUNTS = "prisma_counts.yaml"
PRISMA_DECISIONS = "prisma_decisions.csv"

# Settings key -> the option that sets it, for messages.
PRISMA_SETTING_OPTIONS = {
    "searches": "--search",
    "field": "--field",
    "llm_vote_rule": "--llm-vote-rule",
    "prompt": "--prompt",
    "filter_provider": "--filter-provider",
    "filter_model": "--filter-model",
    "filter_profile": "--filter-profile",
    "fulltext_exp": "--fulltext-exp",
    "fulltext_field": "--fulltext-field",
    "fulltext_prompt": "--fulltext-prompt",
    "reason_field": "--reason-field",
    "title": "--title",
}
PRISMA_LIST_SETTINGS = ("searches", "prompt", "filter_provider", "filter_model", "filter_profile", "fulltext_prompt")


class PrismaRule(str, Enum):
    """How the answers of several models for one record become one decision."""

    majority = "majority"
    any = "any"
    all = "all"


def _prisma_experiment(pid: str, exp: str | None, label: str):
    """``(paths, data_csv, newest_db)`` of a (sub-)experiment; ``data_csv`` is ``None`` when it has no data yet."""

    paths = resolve_experiment(pid, exp, must_exist=False)
    if not os.path.isfile(paths.data):
        return paths, None, None
    db_path = latest_db(paths.work_dir)
    if db_path is None:
        cprint(f"[bold yellow]Warning:[/bold yellow] {label} '{paths.label()}' has no database yet.")
    return paths, paths.data, db_path


def _relative(path, project_path: str):
    """``path`` relative to the project, with ``/`` on every OS (it is saved in files)."""

    return None if path is None else os.path.relpath(path, project_path).replace(os.sep, "/")


def _split_values(values) -> list[str]:
    """Repeatable, comma-separated option values (or a saved list / string) as one list."""

    if isinstance(values, str):
        values = [values]
    return [value.strip() for item in (values or []) for value in str(item).split(",") if value.strip()]


def _is_unset(value) -> bool:
    return value is None or value == "" or value == [] or value == ()


def _prisma_settings(counts_path: str, given: dict, ignore_settings: bool) -> tuple[dict, list[str]]:
    """The options given, with the settings saved in the counts file filling the gaps.

    Returns the settings and the keys taken from the file.
    """

    try:
        saved = {} if ignore_settings else prisma.load_settings(counts_path)
    except ValueError as e:
        raise UnexpectedCLIParamsException(str(e)) from e

    settings, reused = {}, []
    for key in PRISMA_SETTING_OPTIONS:
        value = given.get(key)
        if _is_unset(value) and not _is_unset(saved.get(key)):
            value = saved[key]
            reused.append(key)
        settings[key] = _split_values(value) if key in PRISMA_LIST_SETTINGS else value

    if isinstance(settings["llm_vote_rule"], PrismaRule):
        settings["llm_vote_rule"] = settings["llm_vote_rule"].value
    settings["llm_vote_rule"] = settings["llm_vote_rule"] or PrismaRule.majority.value
    if settings["llm_vote_rule"] not in prisma.RULES:
        raise UnexpectedCLIParamsException(
            f"llm_vote_rule '{settings['llm_vote_rule']}' in '{counts_path}' must be one of {', '.join(prisma.RULES)}."
        )
    return settings, reused


def _prisma_inputs(pid: str, given: dict, ignore_settings: bool) -> dict:
    """Everything `derive` needs, from the CLI options and the settings saved by the last run."""

    project_path = get_project_directory_path(pid)
    paths, screening_data, screening_db = _prisma_experiment(pid, given.get("exp"), "Screening experiment")
    analysis_path = os.path.join(project_path, ANALYSIS_DIR)
    prefix = _exp_prefix(paths)
    counts_path = os.path.join(analysis_path, f"{prefix}{PRISMA_COUNTS}")

    settings, reused = _prisma_settings(counts_path, given, ignore_settings)
    settings["exp"] = paths.name

    fulltext_data = fulltext_db = None
    if settings["fulltext_exp"]:
        if not settings["fulltext_field"]:
            raise UnexpectedCLIParamsException("--fulltext-exp needs --fulltext-field.")
        _, fulltext_data, fulltext_db = _prisma_experiment(pid, settings["fulltext_exp"], "Full-text experiment")
    elif settings["fulltext_field"] or settings["reason_field"] or settings["fulltext_prompt"]:
        raise UnexpectedCLIParamsException(
            "--fulltext-field, --fulltext-prompt and --reason-field need --fulltext-exp."
        )

    known = prisma.list_searches(project_path)
    unknown = [search_id for search_id in settings["searches"] if search_id not in known]
    if unknown:
        raise UnexpectedCLIParamsException(
            f"Search(es) without a __results.csv: {', '.join(unknown)}. Known: {', '.join(known) or 'none'}."
        )

    return {
        "project_path": project_path,
        "analysis_path": analysis_path,
        "paths": paths,
        "prefix": prefix,
        "counts_path": counts_path,
        "svg_path": os.path.join(analysis_path, f"{prefix}{PRISMA_SVG}"),
        "decisions_path": os.path.join(analysis_path, f"{prefix}{PRISMA_DECISIONS}"),
        "settings": settings,
        "reused": reused,
        "screening_data": screening_data,
        "screening_db": screening_db,
        "fulltext_data": fulltext_data,
        "fulltext_db": fulltext_db,
    }


def _prisma_files(inputs: dict) -> dict:
    """The files a run read, relative to the project (saved as `inputs` in the counts file)."""

    project_path = inputs["project_path"]
    files = {"llmexer_version": package_version()}
    for key in ("screening_data", "screening_db", "fulltext_data", "fulltext_db"):
        if inputs[key] is not None:
            files[key] = _relative(inputs[key], project_path)
    return files


def _print_reused_settings(inputs: dict) -> None:
    if not inputs["reused"]:
        return
    shown = []
    for key in inputs["reused"]:
        value = inputs["settings"][key]
        value = ",".join(value) if isinstance(value, list) else value
        shown.append(f"{PRISMA_SETTING_OPTIONS[key]} {value}")
    cprint(
        f"Settings from {os.path.basename(inputs['counts_path'])} (pass an option to change it, "
        f"`--ignore-settings` to drop them): [cyan]{' '.join(shown)}[/cyan]"
    )


def _derive_prisma(inputs: dict) -> tuple[dict, object]:
    settings = inputs["settings"]
    try:
        manual = prisma.load_decisions_file(inputs["decisions_path"])
        return prisma.derive(
            inputs["project_path"],
            searches=settings["searches"],
            screening_data=inputs["screening_data"],
            screening_db=inputs["screening_db"],
            field=settings["field"],
            rule=settings["llm_vote_rule"],
            fulltext_data=inputs["fulltext_data"],
            fulltext_db=inputs["fulltext_db"],
            fulltext_field=settings["fulltext_field"],
            reason_field=settings["reason_field"],
            screening_filters=prisma.answer_filters(settings, prisma.STAGE_SCREENING),
            fulltext_filters=prisma.answer_filters(settings, prisma.STAGE_FULLTEXT),
            manual=manual,
        )
    except ValueError as e:
        raise UnexpectedCLIParamsException(str(e)) from e


def _format_count(value) -> str:
    if value is None:
        return "[bold red]?[/bold red]"
    if isinstance(value, dict):
        return ", ".join(f"{name}: {count}" for name, count in value.items()) or "-"
    return str(value)


def _print_prisma_table(derived: dict, overrides: dict, values: dict) -> None:
    table = Table(title="PRISMA 2020 counts", border_style="bright_blue", show_lines=True)
    table.add_column("Box", style="white")
    table.add_column("Used", style="bold green")
    table.add_column("Derived")
    table.add_column("Override", style="yellow")
    table.add_column("Source", style="dim")
    for key, label in prisma.BOXES.items():
        box = derived.get(key) or {}
        override = _format_count(overrides[key]) if key in overrides else ""
        table.add_row(
            label, _format_count(values.get(key)), _format_count(box.get("value")), override, box.get("source", "")
        )
    console.print(table)


_PRISMA_OPTIONS_HELP = {
    "search": "Search id(s) to count (repeatable, or comma-separated). Default: every search with results.",
    "field": "Yes/no field of the screening answers that keeps (yes) or excludes (no) a record.",
    "llm_vote_rule": "How the LLM answers (votes) for one record are combined: 'majority' (tie keeps), "
    "'any' (one yes keeps), 'all' (one no excludes). Default: majority.",
    "prompt": "Count only the screening answers of these prompt(s) (repeatable, or comma-separated).",
    "filter_provider": "Count only the answers of these provider(s), at both stages.",
    "filter_model": "Count only the answers of these model(s), at both stages.",
    "filter_profile": "Count only the answers of these profile(s), at both stages.",
    "fulltext_exp": "Sub-experiment that assessed the full texts (fills the eligibility boxes).",
    "fulltext_field": "Yes/no field of the full-text answers.",
    "fulltext_prompt": "Count only the full-text answers of these prompt(s).",
    "reason_field": "Field of the full-text answers holding the exclusion reason.",
    "title": "Title drawn above the diagram.",
    "ignore_settings": "Do not re-use the settings saved in the counts file by the last run.",
}


@app.command(name="draw-prisma")
def draw_prisma(
    pid: str = typer.Option(None, "--pid", help="Project ID. If not provided, uses PROJECT_ID from .env."),
    exp: str = _exp_option(),
    search: list[str] = typer.Option(None, "--search", help=_PRISMA_OPTIONS_HELP["search"]),
    field: str = typer.Option(None, "--field", help=_PRISMA_OPTIONS_HELP["field"]),
    llm_vote_rule: PrismaRule = typer.Option(None, "--llm-vote-rule", help=_PRISMA_OPTIONS_HELP["llm_vote_rule"]),
    prompt: list[str] = typer.Option(None, "--prompt", help=_PRISMA_OPTIONS_HELP["prompt"]),
    filter_provider: list[str] = typer.Option(None, "--filter-provider", help=_PRISMA_OPTIONS_HELP["filter_provider"]),
    filter_model: list[str] = typer.Option(None, "--filter-model", help=_PRISMA_OPTIONS_HELP["filter_model"]),
    filter_profile: list[str] = typer.Option(None, "--filter-profile", help=_PRISMA_OPTIONS_HELP["filter_profile"]),
    fulltext_exp: str = typer.Option(None, "--fulltext-exp", help=_PRISMA_OPTIONS_HELP["fulltext_exp"]),
    fulltext_field: str = typer.Option(None, "--fulltext-field", help=_PRISMA_OPTIONS_HELP["fulltext_field"]),
    fulltext_prompt: list[str] = typer.Option(None, "--fulltext-prompt", help=_PRISMA_OPTIONS_HELP["fulltext_prompt"]),
    reason_field: str = typer.Option(None, "--reason-field", help=_PRISMA_OPTIONS_HELP["reason_field"]),
    title: str = typer.Option(None, "--title", help=_PRISMA_OPTIONS_HELP["title"]),
    ignore_settings: bool = typer.Option(False, "--ignore-settings", help=_PRISMA_OPTIONS_HELP["ignore_settings"]),
) -> None:
    """Draw a PRISMA 2020 flow diagram (SVG) from the project's searches and screening

    Writes analysis/prisma_counts.yaml (the settings, every number with its source),
    analysis/prisma.svg and, with --field, analysis/prisma_decisions.csv (one row per
    record). Options not given are taken from the settings of the last run. To change
    a number, set it under `overrides` in the YAML; to overrule the LLMs for a record,
    write yes/no in the `manual` column of the decision file. Both are kept on every
    run. A file whose content changes is first copied to analysis/.backup/.
    """

    pid = get_proper_pid(pid)
    given = dict(
        exp=exp,
        searches=search,
        field=field,
        llm_vote_rule=llm_vote_rule,
        prompt=prompt,
        filter_provider=filter_provider,
        filter_model=filter_model,
        filter_profile=filter_profile,
        fulltext_exp=fulltext_exp,
        fulltext_field=fulltext_field,
        fulltext_prompt=fulltext_prompt,
        reason_field=reason_field,
        title=title,
    )
    inputs = _prisma_inputs(pid, given, ignore_settings)
    _print_reused_settings(inputs)

    derived, decisions = _derive_prisma(inputs)
    try:
        _, overrides = prisma.load_counts_file(inputs["counts_path"])
    except ValueError as e:
        raise UnexpectedCLIParamsException(str(e)) from e
    values = prisma.effective_counts(derived, overrides)

    _print_prisma_table(derived, overrides, values)
    for warning in prisma.check_counts(values):
        cprint(f"[bold yellow]Warning:[/bold yellow] {warning}")

    outputs = prisma.output_texts(
        inputs["counts_path"],
        inputs["svg_path"],
        derived,
        overrides,
        values,
        inputs["settings"]["title"],
        settings=inputs["settings"],
        inputs=_prisma_files(inputs),
        decisions_path=inputs["decisions_path"],
        decisions=decisions,
    )
    planned = prisma.plan_outputs(outputs)

    if settings.dry_run:
        for path, _, backup in planned:
            note = f", backing up the old one to {ANALYSIS_BACKUP_DIR}/{backup.name}" if backup else ""
            cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{path}'{note}")
        return

    for backup in prisma.write_outputs(planned):
        cprint(f"Backed up the previous version to {ANALYSIS_BACKUP_DIR}/{backup.name}")
    cprint(f"Counts (edit `overrides` to change a number):\n  [bold yellow]{inputs['counts_path']}[/bold yellow]")
    if any(str(path) == inputs["decisions_path"] for path, _, _ in planned):
        cprint(
            "Decisions per record (write yes/no in `manual` to overrule the LLMs):\n"
            f"  [bold yellow]{inputs['decisions_path']}[/bold yellow]"
        )
    cprint(f"PRISMA diagram:\n  [bold yellow]{inputs['svg_path']}[/bold yellow]")


@app.command(name="add-prisma")
def add_prisma(
    pid: str = typer.Option(None, "--pid", help="Project ID. If not provided, uses PROJECT_ID from .env."),
    exp: str = _exp_option(),
    search: list[str] = typer.Option(None, "--search", help=_PRISMA_OPTIONS_HELP["search"]),
    field: str = typer.Option(None, "--field", help=_PRISMA_OPTIONS_HELP["field"]),
    llm_vote_rule: PrismaRule = typer.Option(None, "--llm-vote-rule", help=_PRISMA_OPTIONS_HELP["llm_vote_rule"]),
    prompt: list[str] = typer.Option(None, "--prompt", help=_PRISMA_OPTIONS_HELP["prompt"]),
    filter_provider: list[str] = typer.Option(None, "--filter-provider", help=_PRISMA_OPTIONS_HELP["filter_provider"]),
    filter_model: list[str] = typer.Option(None, "--filter-model", help=_PRISMA_OPTIONS_HELP["filter_model"]),
    filter_profile: list[str] = typer.Option(None, "--filter-profile", help=_PRISMA_OPTIONS_HELP["filter_profile"]),
    fulltext_exp: str = typer.Option(None, "--fulltext-exp", help=_PRISMA_OPTIONS_HELP["fulltext_exp"]),
    fulltext_field: str = typer.Option(None, "--fulltext-field", help=_PRISMA_OPTIONS_HELP["fulltext_field"]),
    fulltext_prompt: list[str] = typer.Option(None, "--fulltext-prompt", help=_PRISMA_OPTIONS_HELP["fulltext_prompt"]),
    reason_field: str = typer.Option(None, "--reason-field", help=_PRISMA_OPTIONS_HELP["reason_field"]),
    title: str = typer.Option(None, "--title", help=_PRISMA_OPTIONS_HELP["title"]),
    ignore_settings: bool = typer.Option(False, "--ignore-settings", help=_PRISMA_OPTIONS_HELP["ignore_settings"]),
    rewrite: bool = typer.Option(False, "--rewrite", help="Replace an existing PRISMA notebook, backing it up first."),
) -> None:
    """Add the PRISMA notebook: derive every count step by step and draw the SVG

    Same counts as `analysis draw-prisma`, shown with the rows behind them (e.g. the
    records the models disagree on), and the same counts and decision files.
    Options not given are taken from the settings saved by the last run.
    """

    pid = get_proper_pid(pid)
    given = dict(
        exp=exp,
        searches=search,
        field=field,
        llm_vote_rule=llm_vote_rule,
        prompt=prompt,
        filter_provider=filter_provider,
        filter_model=filter_model,
        filter_profile=filter_profile,
        fulltext_exp=fulltext_exp,
        fulltext_field=fulltext_field,
        fulltext_prompt=fulltext_prompt,
        reason_field=reason_field,
        title=title,
    )
    inputs = _prisma_inputs(pid, given, ignore_settings)
    _print_reused_settings(inputs)
    project_path = inputs["project_path"]
    analysis_path = inputs["analysis_path"]
    prefix = inputs["prefix"]
    notebook_name = f"{prefix}{ADDON_NOTEBOOKS['prisma']}"

    planned = _plan(analysis_path, rewrite, [notebook_name] + list(COPIED_MODULES))
    console.print(_render_plan_table(planned, analysis_path))

    to_write = [item for item in planned if item[2] != "keep"]
    if not to_write:
        cprint("[bold yellow]Warning:[/bold yellow] the PRISMA notebook already exists. Use --rewrite to replace it.")
        return

    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write {len(to_write)} file(s) into '{analysis_path}'")
        return

    ensure_directory_exists(analysis_path)
    backup_path = os.path.join(analysis_path, ANALYSIS_BACKUP_DIR)
    if any(action == "rewrite" for _, _, action, _ in to_write):
        ensure_directory_exists(backup_path)
    for _, path, action, backup in to_write:
        if action == "rewrite":
            shutil.copy2(path, os.path.join(backup_path, backup))

    run_settings = inputs["settings"]
    context = {
        "project_id": pid,
        "package_version": package_version(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "exp_name": run_settings["exp"],
        "searches": run_settings["searches"],
        "screening_data": _relative(inputs["screening_data"], project_path),
        "screening_db": _relative(inputs["screening_db"], project_path),
        "field": run_settings["field"],
        "llm_vote_rule": run_settings["llm_vote_rule"],
        "prompts": run_settings["prompt"],
        "filter_providers": run_settings["filter_provider"],
        "filter_models": run_settings["filter_model"],
        "filter_profiles": run_settings["filter_profile"],
        "fulltext_exp": run_settings["fulltext_exp"],
        "fulltext_data": _relative(inputs["fulltext_data"], project_path),
        "fulltext_db": _relative(inputs["fulltext_db"], project_path),
        "fulltext_field": run_settings["fulltext_field"],
        "fulltext_prompts": run_settings["fulltext_prompt"],
        "reason_field": run_settings["reason_field"],
        "title": run_settings["title"],
        "counts_file": f"{prefix}{PRISMA_COUNTS}",
        "svg_file": f"{prefix}{PRISMA_SVG}",
        "decisions_file": f"{prefix}{PRISMA_DECISIONS}",
    }

    written = []
    pending = {filename for filename, _, action, _ in to_write if action != "keep"}
    if notebook_name in pending:
        written.append(write_notebook("prisma", context, os.path.join(analysis_path, notebook_name)))
    if pending & set(COPIED_MODULES):
        written.extend(copy_analysis_modules(analysis_path))

    cprint(f"PRISMA notebook ready — {len(written)} file(s) written:\n  {analysis_path}")
    cprint(f"Jupyter Notebook: [bold yellow]{os.path.join(analysis_path, notebook_name)}[/bold yellow]")
