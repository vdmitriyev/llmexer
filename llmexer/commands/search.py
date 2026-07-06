"""Search group commands of the CLI interface."""

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import typer
import yaml
from rich.table import Table

from llmexer.base.search import (
    _PAPER_CSV_COLUMNS,
    DEFAULT_OPEN_ACCESS_PARAM,
    DEFAULT_SEARCH_YEAR_PARAM,
    gather_search_csvs,
    generate_search_id,
    merge_search_csvs,
    read_search_params,
    run_semantic_scholar_search,
    save_search_query,
    synf_df_runs_of_search_and_papers,
)
from llmexer.common import (
    ensure_directory_exists,
    get_project_directory_path,
    get_proper_pid,
)
from llmexer.configs import console, cprint, settings
from llmexer.constants import PAPERS_DIR, SEARCHES_DIR, SEARCHES_LOGS_DIR
from llmexer.exceptions import (
    LLMExerException,
    SearchResultsAlreadyExistException,
    UnexpectedCLIParamsException,
)
from llmexer.logger import get_logger

logger = get_logger()

app = typer.Typer(help="Search online digital libraries for papers and metadata.")

# Default values
DEFAULT_QUERY_PARAM = "influence of machine learning on computer science"

# Suffixes for the project-wide merged files: `<pid>__merged_results.csv` and
# `<pid>__merged_filtered.csv`.
MERGED_RESULTS_SUFFIX = "__merged_results.csv"
MERGED_FILTERED_SUFFIX = "__merged_filtered.csv"

# Subdirectory (within `searches/`) holding the raw JSON search responses.
SEARCH_JSONS_DIR = "jsons"

# Shared audit log (within `searches/logs/`) recording every applied filter.
FILTERS_LOG_FILENAME = "filters-applied.log"

# Raw JSON responses live in the `jsons/` subdir; the download-failed CSV lives in the
# `logs/` subdir; the rest sit directly in `searches/`.
_SEARCH_JSON_SUFFIX = "__results_raw.json"
_SEARCH_DOWNLOAD_FAILED_SUFFIX = "__results_download_failed.csv"
_SEARCH_FILE_SUFFIXES = [
    ".yaml",
    "__results.csv",
    "__filtered.csv",
]


def _build_stats_grid(df):
    """Build the 2-column stats grid (Publications per Year + Open Access) for a dataframe."""
    year_counts = (
        df["year"].dropna().astype(int).value_counts().sort_index(ascending=False)
    )
    year_df = year_counts.rename("count").to_frame()
    year_df["pct"] = (year_df["count"] / year_df["count"].sum() * 100).round(1)
    table1 = Table(title="Papers by Year", expand=True)
    table1.add_column("Year", style="cyan", no_wrap=True, min_width=6)
    table1.add_column("Count", style="green", justify="right", min_width=5)
    table1.add_column("%", style="yellow", justify="right")
    for year, row in year_df.iterrows():
        table1.add_row(str(year), str(int(row["count"])), f"{row['pct']}%")

    total = len(df)

    def _pct(n):
        return f"{round(n / total * 100, 1)}%" if total else "0%"

    table2 = Table(title="Stats Breakdown", expand=True)
    table2.add_column("Stat", no_wrap=True)
    table2.add_column("Count", style="cyan", justify="right")
    table2.add_column("%", style="cyan", justify="right")

    table2.add_row(
        "[white]Total:[/white] [bold white]papers[/bold white]",
        str(total),
        "100%",
    )

    oa_true = int(df["isOpenAccess"].sum())
    table2.add_row(
        f"[white]Open Access:[/white] [magenta]True[/magenta]",
        f"[magenta]{oa_true}[/magenta]",
        f"[magenta]{_pct(oa_true)}[/magenta]",
    )

    if "entry_source" in df.columns:
        for src, cnt in df["entry_source"].fillna("").value_counts().items():
            table2.add_row(
                f"[white]Entry Source:[/white] [magenta]{src}[/magenta]",
                f"[magenta]{int(cnt)}[/magenta]",
                f"[magenta]{_pct(int(cnt))}[/magenta]",
            )

    for lang, cnt in df["language"].value_counts().items():
        table2.add_row(
            f"[white]Language:[/white] [magenta]{lang}[/magenta]",
            f"[magenta]{int(cnt)}[/magenta]",
            f"[magenta]{_pct(int(cnt))}[/magenta]",
        )

    if "pdf_downloaded" in df.columns:
        dl_existing = int(df["pdf_downloaded"].fillna(False).astype(bool).sum())
        dl_missing = total - dl_existing
        table2.add_row(
            "[white]PDF:[/white] [bold green]existing[/bold green]",
            f"[bold green]{dl_existing}[/bold green]",
            f"[bold green]{_pct(dl_existing)}[/bold green]",
        )
        table2.add_row(
            "[white]PDF:[/white] [bold red]missing[/bold red]",
            f"[bold red]{dl_missing}[/bold red]",
            f"[bold red]{_pct(dl_missing)}[/bold red]",
        )

    if "txt_filename" in df.columns:
        txt_existing = int(
            df["txt_filename"].fillna("").astype(str).str.strip().ne("").sum()
        )
        txt_missing = total - txt_existing
        table2.add_row(
            "[white]TXT:[/white] [bold green]existing[/bold green]",
            f"[bold green]{txt_existing}[/bold green]",
            f"[bold green]{_pct(txt_existing)}[/bold green]",
        )
        table2.add_row(
            "[white]TXT:[/white] [bold red]missing[/bold red]",
            f"[bold red]{txt_missing}[/bold red]",
            f"[bold red]{_pct(txt_missing)}[/bold red]",
        )

    if "markdown_filename" in df.columns:
        md_existing = int(
            df["markdown_filename"].fillna("").astype(str).str.strip().ne("").sum()
        )
        md_missing = total - md_existing
        table2.add_row(
            "[white]Markdown:[/white] [bold green]existing[/bold green]",
            f"[bold green]{md_existing}[/bold green]",
            f"[bold green]{_pct(md_existing)}[/bold green]",
        )
        table2.add_row(
            "[white]Markdown:[/white] [bold red]missing[/bold red]",
            f"[bold red]{md_missing}[/bold red]",
            f"[bold red]{_pct(md_missing)}[/bold red]",
        )

    layout = Table.grid(padding=(0, 2))
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(table1, table2)

    return layout


def print_search_header(pid, search_id, query, year, only_open_access):
    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="bold white", no_wrap=True)
    header.add_column()
    header.add_row("Project:", f"[bold yellow]{pid}[/bold yellow]")
    header.add_row("Search ID:", f"[bold yellow]{search_id}[/bold yellow]")
    header.add_row("Query:", f"[bold green]{query}[/bold green]")
    header.add_row("Year:", f"[bold cyan]{year}[/bold cyan]")
    header.add_row(
        "Only Open Access:", f"[bold magenta]{only_open_access}[/bold magenta]"
    )
    console.print(header)


@app.command()
def create(
    query: str = typer.Option(
        DEFAULT_QUERY_PARAM,
        "--query",
        help="Query string for the search",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to store the search config. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Create a new search configuration YAML file"""

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    search_id, yaml_filename = save_search_query(
        experiment_path, query, DEFAULT_SEARCH_YEAR_PARAM, DEFAULT_OPEN_ACCESS_PARAM
    )

    cprint(f"Created search config: [bold yellow]{yaml_filename}[/bold yellow]")
    cprint(f"Query: [bold green]{query}[/bold green]")


@app.command(name="list")
def list_searches(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """List all search YAML files for a project"""
    from datetime import datetime, timezone

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    if not os.path.isdir(searches_path):
        cprint("No searches found.")
        return

    yaml_files = sorted(Path(searches_path).glob("*.yaml"))
    if not yaml_files:
        cprint("No searches found.")
        return

    table = Table()
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Search file", style="cyan", no_wrap=True)
    table.add_column("Query", style="white")
    table.add_column("Year", style="cyan", no_wrap=True)
    table.add_column("Created", style="cyan", no_wrap=True)
    table.add_column("Results", justify="center", no_wrap=True)

    for i, yaml_path in enumerate(yaml_files, start=1):
        search_id = yaml_path.stem
        with open(yaml_path, "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        query = params.get("query", "")
        year = params.get("year", DEFAULT_SEARCH_YEAR_PARAM)
        ctime = datetime.fromtimestamp(
            yaml_path.stat().st_ctime, tz=timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")
        results_csv = os.path.join(searches_path, f"{search_id}__results.csv")
        results_display = (
            "[bold green]Yes[/bold green]"
            if os.path.exists(results_csv)
            else "[dim]No[/dim]"
        )
        table.add_row(str(i), yaml_path.name, query, year, ctime, results_display)

    console.print(table)
    latest_name = yaml_files[-1].name
    cprint("\nExample to view search stats:")
    cprint(f"[bold yellow]llmexer search stats --file {latest_name}[/bold yellow]")


def print_info_not_search_file(file, results_csv):
    cprint(
        f"\n[bold yellow]Warning![/bold yellow] Results file not found:\n[cyan]{results_csv}[/cyan]"
    )
    cprint(f"\nRun search command first:")
    cprint(f"[bold yellow]search run --file {file}[/bold yellow]")


@app.command(name="rename")
def rename_search(
    old_id: str = typer.Option(
        ...,
        "--old-id",
        help="Current search ID (or full YAML filename) to rename.",
    ),
    new_id: str = typer.Option(
        ...,
        "--new-id",
        help="New search ID.",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Rename a search and all its associated files"""

    # Strip .yaml extension if passed as full filename
    old_id = os.path.splitext(old_id)[0]
    new_id = os.path.splitext(new_id)[0]

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    old_yaml = os.path.join(searches_path, f"{old_id}.yaml")
    new_yaml = os.path.join(searches_path, f"{new_id}.yaml")

    if not os.path.exists(old_yaml):
        raise LLMExerException(f"Search '{old_id}' does not exist.")

    if os.path.exists(new_yaml):
        raise LLMExerException(f"Search '{new_id}' already exists.")

    for suffix in _SEARCH_FILE_SUFFIXES:
        src = os.path.join(searches_path, f"{old_id}{suffix}")
        dst = os.path.join(searches_path, f"{new_id}{suffix}")
        if os.path.exists(src):
            os.rename(src, dst)

    # The raw JSON responses live in the `jsons/` subdirectory.
    jsons_path = os.path.join(searches_path, SEARCH_JSONS_DIR)
    json_src = os.path.join(jsons_path, f"{old_id}{_SEARCH_JSON_SUFFIX}")
    json_dst = os.path.join(jsons_path, f"{new_id}{_SEARCH_JSON_SUFFIX}")
    if os.path.exists(json_src):
        os.rename(json_src, json_dst)

    # The download-failed CSV lives in the `logs/` subdirectory.
    logs_path = os.path.join(searches_path, SEARCHES_LOGS_DIR)
    failed_src = os.path.join(logs_path, f"{old_id}{_SEARCH_DOWNLOAD_FAILED_SUFFIX}")
    failed_dst = os.path.join(logs_path, f"{new_id}{_SEARCH_DOWNLOAD_FAILED_SUFFIX}")
    if os.path.exists(failed_src):
        os.rename(failed_src, failed_dst)

    cprint(
        f"Renamed search: [bold yellow]{old_id}[/bold yellow] → [bold yellow]{new_id}[/bold yellow]"
    )


@app.command()
def run(
    query: str = typer.Option(
        None,
        "--query",
        help="Query string to be used during the search",
    ),
    file: str = typer.Option(
        None,
        "--file",
        help="Name of the YAML file with search parameters",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to be used to store search results. If not provided, uses PROJECT_ID from .env.",
    ),
    limit: int = typer.Option(
        None,
        "--limit",
        help="Maximum number of papers to retrieve (fetched in batches of 100)",
    ),
    batch: int = typer.Option(
        1000,
        "--batch",
        help="A number of papers to retrieve at once",
    ),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help="Overwrite existing result files if they already exist.",
    ),
) -> None:
    """Runs a new search and saves results"""

    if query is not None and file is not None:
        raise UnexpectedCLIParamsException("Only one can be set: query or file")

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    if query:
        year = DEFAULT_SEARCH_YEAR_PARAM
        only_open_access = DEFAULT_OPEN_ACCESS_PARAM
        search_id, yaml_filename = save_search_query(
            experiment_path, query, DEFAULT_SEARCH_YEAR_PARAM, DEFAULT_OPEN_ACCESS_PARAM
        )
        cprint(f"Query saved to: [bold blue]{yaml_filename}[/bold blue]")
    if file:
        search_id, query, year, only_open_access = read_search_params(
            file, experiment_path, query
        )
        cprint(f"Loaded config from: [bold blue]{file}[/bold blue]")
    elif not query:
        raise LLMExerException(
            "No query provided. Use --query or --file with a YAML file."
        )

    cprint(f"Limit: [bold blue]{limit}[/bold blue]")

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    ensure_directory_exists(searches_path)
    jsons_path = os.path.join(searches_path, SEARCH_JSONS_DIR)
    ensure_directory_exists(jsons_path)

    json_path = os.path.join(jsons_path, f"{search_id}{_SEARCH_JSON_SUFFIX}")
    csv_path = os.path.join(searches_path, f"{search_id}__results.csv")

    if not rewrite:
        existing = [p for p in [json_path, csv_path] if os.path.exists(p)]
        if existing:
            raise SearchResultsAlreadyExistException(
                f"Result files already exist: {existing}. Use --rewrite to overwrite."
            )

    if settings.dry_run:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{json_path}'")
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{csv_path}'")
        return

    print_search_header(pid, search_id, query, year, only_open_access)

    records = run_semantic_scholar_search(
        query, year, only_open_access, batch, limit, json_path, csv_path
    )

    cprint(f"[bold green]Total retrieved:[/bold green] {len(records)} paper(s)")
    cprint(f"File with raw responses ([magenta]JSON[/magenta]):\n  {json_path}")
    cprint(f"File with results ([magenta]CSV[/magenta]):\n  {csv_path}")


@app.command()
def merge(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID whose searches to merge. If not provided, uses PROJECT_ID from .env.",
    ),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help="Overwrite the merged file if it already exists.",
    ),
) -> None:
    """Merge a project's search results into two deduplicated CSVs.

    Produces ``<pid>__merged_results.csv`` from every ``*__results.csv`` and
    ``<pid>__merged_filtered.csv`` from every ``*__filtered.csv``. Publications are deduplicated
    by DOI (falling back to title). Each source search becomes a binary column named after its
    YAML file (the search id, without ``.yaml``), plus a ``duplicates_counter`` column holding
    the number of duplicate occurrences (one less than the number of searches each publication
    was found in).
    """

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    results_csvs, filtered_csvs = gather_search_csvs(searches_path)
    if not results_csvs and not filtered_csvs:
        raise LLMExerException(
            f"No search result files found in '{searches_path}'. Run `search run` first."
        )

    # (source csvs, stem suffix to strip, output path, label)
    jobs = []
    if results_csvs:
        jobs.append(
            (
                results_csvs,
                "__results",
                os.path.join(searches_path, f"{pid}{MERGED_RESULTS_SUFFIX}"),
                "results",
            )
        )
    if filtered_csvs:
        jobs.append(
            (
                filtered_csvs,
                "__filtered",
                os.path.join(searches_path, f"{pid}{MERGED_FILTERED_SUFFIX}"),
                "filtered",
            )
        )

    existing = [out for _, _, out, _ in jobs if os.path.exists(out)]
    if existing and not rewrite:
        raise SearchResultsAlreadyExistException(
            f"Merged file(s) already exist: {existing}. Use --rewrite to overwrite."
        )

    ensure_directory_exists(searches_path)

    for csvs, stem_suffix, output_path, label in jobs:
        merged_df, run_columns = merge_search_csvs(csvs, stem_suffix)

        if settings.dry_run:
            cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{output_path}'")
            cprint(
                f"[bold yellow]Dry run:[/bold yellow] {label}: {len(run_columns)} "
                f"search(es), {len(merged_df)} unique publication(s)"
            )
            continue

        merged_df.to_csv(output_path, index=False, encoding="utf-8", sep=";")
        cprint(
            f"[bold green]Merged[/bold green] {len(run_columns)} {label} search(es) into "
            f"{len(merged_df)} unique publication(s)."
        )
        cprint(f"File with merged {label} ([magenta]CSV[/magenta]):\n  {output_path}")


@app.command()
def stats(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID.",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to look up results for. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Display statistics for a completed search result"""

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    if file is None:
        merged_results_path = os.path.join(
            searches_path, f"{pid}{MERGED_RESULTS_SUFFIX}"
        )
        merged_filtered_path = os.path.join(
            searches_path, f"{pid}{MERGED_FILTERED_SUFFIX}"
        )
        merged_paths = [
            (merged_results_path, "Merged results"),
            (merged_filtered_path, "Merged filtered"),
        ]
        if not any(os.path.exists(p) for p, _ in merged_paths):
            raise UnexpectedCLIParamsException(
                "--file is required (or run `search merge` first)."
            )
        cprint(f"[bold]Project:[/bold] [cyan]{pid}[/cyan]")
        for merged_path, label in merged_paths:
            if not os.path.exists(merged_path):
                continue
            cprint()
            merged_df = pd.read_csv(merged_path, sep=";")
            cprint(
                f"[bold]{label}:[/bold] [yellow]{Path(merged_path).name}[/yellow] "
                f"({len(merged_df)} papers)"
            )
            console.print(_build_stats_grid(merged_df))
        return

    search_id, query, year, only_open_access = read_search_params(file, experiment_path)
    print_search_header(pid, search_id, query, year, only_open_access)

    csv_path = os.path.join(searches_path, f"{search_id}__results.csv")

    if not os.path.exists(csv_path):
        print_info_not_search_file(file, csv_path)
        return

    cprint()
    df = pd.read_csv(csv_path, sep=";")
    tbl_original = _build_stats_grid(df)
    cprint(
        f"[bold]Results:[/bold] [yellow]{Path(csv_path).name}[/yellow] ({len(df)} papers)"
    )
    console.print(tbl_original)

    filtered_path = os.path.join(searches_path, f"{search_id}__filtered.csv")
    tbl_filtered, filtered_df = None, []
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path, sep=";")
        tbl_filtered = _build_stats_grid(filtered_df)

    if tbl_filtered is not None:
        cprint()
        cprint(
            f"[bold]Filtered:[/bold] [green]{Path(filtered_path).name}[/green] ({len(filtered_df)} papers)"
        )
        console.print(tbl_filtered)


def _log_filter_applied(logs_path, search_file, desc, input_n, output_n):
    """Append one line documenting an applied filter to ``filters-applied.log``."""
    ensure_directory_exists(logs_path)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = (
        f"{timestamp} search file: {search_file}; filter applied: {desc} ; "
        f"input rows: {input_n}; output rows: {output_n}\n"
    )
    log_path = os.path.join(logs_path, FILTERS_LOG_FILENAME)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line)


def _filter_search(search_id, experiment_path, filters):
    """Apply the given exclusion ``filters`` to one search, rewriting its ``__filtered.csv``.

    Reads the existing ``__filtered.csv`` if present (so filters chain), otherwise the
    ``__results.csv``. Each applied filter is recorded in ``searches/logs/filters-applied.log``.
    """
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    results_path = os.path.join(searches_path, f"{search_id}__results.csv")
    filtered_path = os.path.join(searches_path, f"{search_id}__filtered.csv")

    # Re-apply to the already-filtered file if it exists, else fall back to the results file.
    source_path = filtered_path if os.path.exists(filtered_path) else results_path
    if not os.path.exists(source_path):
        print_info_not_search_file(search_id, results_path)
        return

    df = pd.read_csv(source_path, sep=";")

    cprint(f"Source: [bold]{Path(source_path).name}[/bold]")
    logs_path = os.path.join(searches_path, SEARCHES_LOGS_DIR)

    current = df
    for desc, keep_mask in filters:
        input_n = len(current)
        current = current[keep_mask(current)]
        output_n = len(current)
        cprint(
            f"Excluded [bold cyan]{desc}[/bold cyan]: "
            f"[bold white]{input_n}[/bold white] → [bold green]{output_n}[/bold green]"
        )
        if not settings.dry_run:
            _log_filter_applied(
                logs_path, Path(filtered_path).name, desc, input_n, output_n
            )

    if not settings.dry_run:
        current.to_csv(filtered_path, index=False, encoding="utf-8", sep=";")
        logger.debug("Wrote filtered results to '%s'", filtered_path)
        cprint(f"Saved filtered results to: [bold]{Path(filtered_path).name}[/bold]\n")
    else:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{filtered_path}'")


@app.command(name="filter")
def filter_results(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID. If omitted, the filter is applied to every search in the project.",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to look up results for. If not provided, uses PROJECT_ID from .env.",
    ),
    language: str = typer.Option(
        None,
        "--language",
        help="Exclude rows whose language equals this code (e.g. 'de').",
    ),
    source: str = typer.Option(
        None,
        "--source",
        help="Exclude rows whose entry_source equals this value (e.g. 'manually added').",
    ),
    doi: str = typer.Option(
        None,
        "--doi",
        help="Exclude rows whose doi equals this value.",
    ),
    downloaded: bool = typer.Option(
        False,
        "--downloaded",
        help="Exclude rows that are not downloaded (keep only downloaded papers).",
    ),
) -> None:
    """Exclude papers from searches by one or more criteria, rewriting ``__filtered.csv``.

    Reads the existing ``__filtered.csv`` if present (so filters chain), otherwise the
    ``__results.csv``. With ``--file`` a single search is filtered; without it, every search in the
    project is filtered. Each applied filter is recorded in ``searches/logs/filters-applied.log``.
    """

    # Build the ordered list of active exclusion filters: (description, keep-mask builder).
    filters = []
    if language is not None:
        filters.append((f"language={language}", lambda d: d["language"] != language))
    if source is not None:
        filters.append((f"source={source}", lambda d: d["entry_source"] != source))
    if doi is not None:
        filters.append((f"doi={doi}", lambda d: d["doi"] != doi))
    if downloaded:
        filters.append(
            (
                "downloaded",
                lambda d: d["pdf_downloaded"]
                .astype(str)
                .str.strip()
                .str.lower()
                .isin({"true", "1"}),
            )
        )

    if not filters:
        raise UnexpectedCLIParamsException(
            "Provide at least one filter criterion: --language, --source, --doi, --downloaded."
        )

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    if file is not None:
        search_ids = [read_search_params(file, experiment_path)[0]]
    else:
        search_ids = [p.stem for p in sorted(Path(searches_path).glob("*.yaml"))]
        if not search_ids:
            cprint("No searches found.")
            return

    for search_id in search_ids:
        _, query, year, only_open_access = read_search_params(
            f"{search_id}.yaml", experiment_path
        )
        print_search_header(pid, search_id, query, year, only_open_access)
        _filter_search(search_id, experiment_path, filters)


def _search_id_from_file(name):
    """Return the bare search id for a search filename (strips extension + role suffix)."""
    stem = os.path.splitext(os.path.basename(name))[0]
    for suffix in ("__results", "__filtered"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _sync_search(search_id, experiment_path, add_new_rows=True):
    """Reconcile a search's ``__results.csv`` (and ``__filtered.csv``) against the papers/ folder.

    When ``add_new_rows`` is ``False`` only files listed in existing rows are checked; PDFs present
    in ``papers/`` but not already listed are not appended as new rows.
    """
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    results_csv = os.path.join(searches_path, f"{search_id}__results.csv")
    filtered_csv = os.path.join(searches_path, f"{search_id}__filtered.csv")

    if not os.path.exists(results_csv):
        return

    df = pd.read_csv(results_csv, sep=";")
    updated_df, updated_count, added_count = synf_df_runs_of_search_and_papers(
        df, papers_path, add_new_rows=add_new_rows
    )

    if not settings.dry_run:
        updated_df.to_csv(results_csv, index=False, encoding="utf-8", sep=";")
        logger.debug("Synced results CSV: '%s'", results_csv)
    else:
        cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{results_csv}'")

    if os.path.exists(filtered_csv):
        filtered_df = pd.read_csv(filtered_csv, sep=";")
        updated_filtered_df, f_updated, f_added = synf_df_runs_of_search_and_papers(
            filtered_df, papers_path, add_new_rows=add_new_rows
        )
        if not settings.dry_run:
            updated_filtered_df.to_csv(
                filtered_csv, index=False, encoding="utf-8", sep=";"
            )
            logger.debug("Synced filtered CSV: '%s'", filtered_csv)
        else:
            cprint(f"[bold yellow]Dry run:[/bold yellow] would write '{filtered_csv}'")

    cprint(f"Rows updated: [bold cyan]{updated_count}[/bold cyan]")
    cprint(f"New rows added: [bold green]{added_count}[/bold green]")
    cprint("[bold]Sync complete.[/bold]")


@app.command()
def sync(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID. If omitted, every search in the project is synced.",
    ),
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    add_local_extra_pdfs: bool = typer.Option(
        False,
        "--add-local-extra-pdfs",
        help="Also append new rows for PDFs found in papers/ that are not already listed. "
        "By default only files listed in existing rows are checked. Only applies with --file; "
        "when syncing all searches it is not applied.",
    ),
) -> None:
    """Sync CSV result files against the papers/ folder of the project.

    With ``--file`` a single search is synced; without it, every search in the project is
    synced (in that case ``--add-local-extra-pdfs`` is not applied, so no new rows are added).
    """

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)
    searches_path = os.path.join(experiment_path, SEARCHES_DIR)

    if file is not None:
        search_ids = [read_search_params(file, experiment_path)[0]]
        add_new_rows = add_local_extra_pdfs
    else:
        search_ids = [p.stem for p in sorted(Path(searches_path).glob("*.yaml"))]
        if not search_ids:
            cprint("No searches found.")
            return
        add_new_rows = False  # bulk sync never appends local extra PDFs

    for search_id in search_ids:
        _, query, year, only_open_access = read_search_params(
            f"{search_id}.yaml", experiment_path
        )
        print_search_header(pid, search_id, query, year, only_open_access)
        _sync_search(search_id, experiment_path, add_new_rows=add_new_rows)
