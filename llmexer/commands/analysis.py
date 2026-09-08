"""Analysis group commands of the CLI interface."""

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from llmexer.base.analysis import COPIED_MODULES
from llmexer.base.analysis.notebook import (
    NOTEBOOKS,
    copy_analysis_modules,
    write_notebook,
)
from llmexer.base.dao import latest_db
from llmexer.common import (
    ensure_directory_exists,
    get_project_directory_path,
    get_proper_pid,
    next_backup_name,
)
from llmexer.configs import console, cprint, settings
from llmexer.constants import ANALYSIS_BACKUP_DIR, ANALYSIS_DIR, SEARCHES_DIR
from llmexer.version import package_version

app = typer.Typer(help="Helps with analyse of the data by generating ready-to-run Jupyter notebooks.")


def _latest_db_name(project_path: str):
    """Filename of the project's newest ``experiment_*.db``, or ``None``.

    Only the name travels into the notebook: the notebook joins it onto its own
    ``DIR_EXPERIMENT``, so the project folder stays relocatable.
    """

    from llmexer.base.experiment import DIR_EXPERIMENT

    newest = latest_db(os.path.join(project_path, DIR_EXPERIMENT))

    return os.path.basename(newest) if newest else None


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


def _plan(analysis_path: str, rewrite: bool) -> list:
    """Decide, per output file, whether it is written, kept or backed up first.

    Notebooks and modules are treated the same way: both may carry a
    researcher's edits, so neither is overwritten without ``--rewrite``, and
    ``--rewrite`` backs the old file up before replacing it. Backups go into
    ``.backup/`` rather than sitting next to the notebooks, so the analysis
    folder keeps showing only the files meant to be opened.
    """

    backup_path = os.path.join(analysis_path, ANALYSIS_BACKUP_DIR)
    planned = []
    for filename in list(NOTEBOOKS.values()) + list(COPIED_MODULES):
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
) -> None:
    """Scaffold Jupyter notebooks for analysing a project

    Creates ``analysis/`` next to the project's ``experiment/``, ``papers/`` and
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

    planned = _plan(analysis_path, rewrite)
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
        "db_file": _latest_db_name(project_path),
        "searches": _search_files(project_path),
        "package_version": package_version(),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    written = []
    pending = {filename for filename, _, action, _ in to_write if action != "keep"}
    for kind, filename in NOTEBOOKS.items():
        if filename in pending:
            written.append(write_notebook(kind, context, os.path.join(analysis_path, filename)))
    if pending & set(COPIED_MODULES):
        written.extend(copy_analysis_modules(analysis_path))

    cprint(f"Analysis workspace ready — {len(written)} file(s) written:\n  {analysis_path}")

    notebook = os.path.join(analysis_path, NOTEBOOKS["experiment"])
    cprint("\nNext step is to open Jupyter Notebook file and run all cells.")
    cprint(f"Jupyter Notebook: [bold yellow]{notebook}[/bold yellow]")
