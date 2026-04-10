"""Search group commands."""

import json
import os
import uuid
from pathlib import Path

import pandas as pd
import requests
import typer
import yaml
from requests.adapters import HTTPAdapter, Retry
from rich.table import Table

from llmexer.commands.papers import _make_structured_filename
from llmexer.common import (
    ensure_directory_exists,
    get_experiment_directory_path,
    get_proper_eid,
    make_http_session,
)
from llmexer.configs import console, settings
from llmexer.constants import PAPERS_DIR, SEARCHES_DIR
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
DEFAULT_SEARCH_YEAR_PARAM = "2020-2025"
DEFAULT_OPEN_ACCESS_PARAM = False

DEFAULT_VALUE_ENTRY_SOURCE = "Semantic Scholar"

# Semantic Scholar API constants
_SEM_SCHOLAR_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_SEM_SCHOLAR_FIELDS = (
    "paperId,title,authors,abstract,isOpenAccess,externalIds,year,"
    "referenceCount,citationCount,fieldsOfStudy,citationStyles,publicationTypes"
)
_PAPER_CSV_COLUMNS = [
    "sem_scholar_paper_id",
    "year",
    "title",
    "authors",
    "abstract",
    "isOpenAccess",
    "doi",
    "language",
    "citationCount",
    "referenceCount",
    "entry_source",
    "pdf_filename",
    "txt_filename",
    "markdown_filename",
    "pdf_downloaded",
]


def _detect_language(title: str | None, abstract: str | None) -> str:
    """Detect language from title and abstract separately.
    Returns 'unclear' if they disagree or on failure or missing text."""
    from langdetect import LangDetectException, detect

    try:
        if title and str(title).strip() and abstract and str(abstract).strip():
            lang_title = detect(str(title).strip())
            lang_abstract = detect(str(abstract).strip())
            return lang_title if lang_title == lang_abstract else "unclear"
        text = " ".join(filter(None, [title, abstract])).strip()
        if not text:
            return "unclear"
        return detect(text)
    except LangDetectException:
        return "unclear"


def generate_search_id() -> str:
    """
    Generate a unique experiment ID formatted as 'YYYYMMDD-GUID'

    Returns:
      str: A string in the format 'YYYYMMDD-UUID'.
    """
    from datetime import date, datetime, timedelta, timezone

    now_utc = datetime.now(timezone.utc)
    formatted_datetime = now_utc.strftime("%Y%m%d")
    unique_id = str(uuid.uuid4())[:8]
    return f"{formatted_datetime}-{unique_id}"


def save_search_query(
    experiment_path: str,
    query: str,
    year: str = DEFAULT_SEARCH_YEAR_PARAM,
    only_open_access: bool = DEFAULT_OPEN_ACCESS_PARAM,
) -> str:
    """_summary_

    Args:
        query (str): _description_
        year (str, optional): _description_. Defaults to DEFAULT_SEARCH_YEAR_PARAM.
        only_open_access (bool, optional): _description_. Defaults to DEFAULT_OPEN_ACCESS_PARAM.

    Returns:
        str: _description_
    """

    # Create searches directory inside the experiment folder
    searches_path = os.path.join(experiment_path, "searches")
    ensure_directory_exists(searches_path)
    search_id = generate_search_id()

    yaml_filename = f"{search_id}.yaml"
    yaml_path = os.path.join(searches_path, yaml_filename)

    # Create YAML content
    search_config = {"query": query, "year": year, "onlyOpenAccess": only_open_access}

    # Write YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(search_config, f, default_flow_style=False, sort_keys=False)

    return search_id, yaml_filename


def run_semantic_scholar_search(
    query: str,
    year: str,
    only_open_access: bool,
    batch_size: int,
    limit_size: int,
    json_path: str,
    csv_path: str,
) -> tuple[list[dict], list[dict]]:
    """Call the Semantic Scholar bulk search API with pagination.

    Returns:
        (raw_json_results, records): raw_json_results is a list of raw API response dicts (one per page);
        records is a list of flattened paper dicts with PAPER_CSV_COLUMNS fields.
    """

    session = make_http_session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    params: dict = {
        "query": query,
        "fields": _SEM_SCHOLAR_FIELDS,
        "limit": min(batch_size, 1000),
    }

    if year:
        params["year"] = year
    if only_open_access:
        params["openAccessPdf"] = ""

    raw_json_results: list[dict] = []
    records: list[dict] = []

    while True:
        response = session.get(_SEM_SCHOLAR_BULK_URL, params=params)
        response.raise_for_status()
        data = response.json()
        raw_json_results.append(data)

        papers = data.get("data", [])
        for paper in papers:
            if limit_size is not None and len(records) >= limit_size:
                break
            ext_ids = paper.get("externalIds") or {}
            pdf_filename = _make_structured_filename(
                paper.get("year"),
                next(
                    (
                        (a.get("name") or "").strip().split()[-1]
                        for a in (paper.get("authors") or [])
                        if (a.get("name") or "").strip()
                    ),
                    None,
                ),
                paper.get("title"),
                ext_ids.get("DOI"),
            )
            records.append(
                {
                    "sem_scholar_paper_id": paper.get("paperId"),
                    "year": paper.get("year"),
                    "title": paper.get("title"),
                    "authors": "; ".join(
                        a.get("name", "") for a in paper.get("authors", [])
                    ),
                    "abstract": paper.get("abstract"),
                    "isOpenAccess": paper.get("isOpenAccess"),
                    "doi": ext_ids.get("DOI"),
                    "language": _detect_language(
                        paper.get("title"), paper.get("abstract")
                    ),
                    "referenceCount": paper.get("referenceCount"),
                    "citationCount": paper.get("citationCount"),
                    "entry_source": DEFAULT_VALUE_ENTRY_SOURCE,
                    "pdf_filename": pdf_filename,
                    "txt_filename": "",
                    "markdown_filename": "",
                    "pdf_downloaded": False,
                }
            )

        console.print(f"Retrieved {len(records)} paper(s) so far...")
        logger.debug(
            "Page retrieved: %d papers in page, %d total", len(papers), len(records)
        )

        token = data.get("token")
        if not token:
            break
        params["token"] = token

        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)

        df = pd.DataFrame(records, columns=_PAPER_CSV_COLUMNS)
        df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(raw_json_results, fh, ensure_ascii=False, indent=4)
    logger.debug("Wrote raw response to '%s'", json_path)

    df = pd.DataFrame(records, columns=_PAPER_CSV_COLUMNS)
    df.sort_values(by=["year", "title"], ascending=[False, True]).to_csv(
        csv_path, index=False, encoding="utf-8", sep=";"
    )
    logger.debug("Wrote CSV to '%s'", csv_path)

    console.print(f"[bold green]Total retrieved:[/bold green] {len(records)} paper(s)")
    console.print(f"File with raw responses ([magenta]JSON[/magenta]):\n  {json_path}")
    console.print(f"File with results ([magenta]CSV[/magenta]):\n  {csv_path}")

    return records


def read_search_params(file, experiment_path, query_default: str = None):

    file_stem = os.path.splitext(os.path.basename(file))[0]
    search_id = file_stem

    # Load parameters from config file if provided
    if os.path.isabs(file):
        search_file_path = file
    else:
        search_file_path = os.path.join(experiment_path, SEARCHES_DIR, file)

    if not os.path.exists(search_file_path):
        raise LLMExerException(f"Search file does not exist: '{file}' ")

    with open(search_file_path, "r", encoding="utf-8") as f:
        search_params = yaml.safe_load(f)

    query = search_params.get("query", query_default)
    year = search_params.get("year", DEFAULT_SEARCH_YEAR_PARAM)
    only_open_access = search_params.get("onlyOpenAccess", DEFAULT_OPEN_ACCESS_PARAM)

    return search_id, query, year, only_open_access


def print_search_header(eid, search_id, query, year, only_open_access):
    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="bold white", no_wrap=True)
    header.add_column()
    header.add_row("Experiment:", f"[bold yellow]{eid}[/bold yellow]")
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
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to store the search config. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Create a new search configuration YAML file"""

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    search_id, yaml_filename = save_search_query(
        experiment_path, query, DEFAULT_SEARCH_YEAR_PARAM, DEFAULT_OPEN_ACCESS_PARAM
    )

    console.print(f"Created search config: [bold yellow]{yaml_filename}[/bold yellow]")
    console.print(f"Query: [bold green]{query}[/bold green]")


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
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to be used to store search results. If not provided, uses EXPERIMENT_ID from .env.",
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

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    if query:
        year = DEFAULT_SEARCH_YEAR_PARAM
        only_open_access = DEFAULT_OPEN_ACCESS_PARAM
        search_id, yaml_filename = save_search_query(
            experiment_path, query, DEFAULT_SEARCH_YEAR_PARAM, DEFAULT_OPEN_ACCESS_PARAM
        )
        console.print(f"Query saved to: [bold blue]{yaml_filename}[/bold blue]")
    if file:
        search_id, query, year, only_open_access = read_search_params(
            file, experiment_path, query
        )
        console.print(f"Loaded config from: [bold blue]{file}[/bold blue]")
    elif not query:
        raise LLMExerException(
            "No query provided. Use --query or --file with a YAML file."
        )

    console.print(f"Limit: [bold blue]{limit}[/bold blue]")

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    ensure_directory_exists(searches_path)

    json_path = os.path.join(searches_path, f"{search_id}_results_raw.json")
    csv_path = os.path.join(searches_path, f"{search_id}_results.csv")

    if not rewrite:
        existing = [p for p in [json_path, csv_path] if os.path.exists(p)]
        if existing:
            raise SearchResultsAlreadyExistException(
                f"Result files already exist: {existing}. Use --rewrite to overwrite."
            )

    if settings.dry_run:
        console.print(f"[bold yellow]Dry run:[/bold yellow] would write '{json_path}'")
        console.print(f"[bold yellow]Dry run:[/bold yellow] would write '{csv_path}'")
        return

    print_search_header(eid, search_id, query, year, only_open_access)

    run_semantic_scholar_search(
        query, year, only_open_access, batch, limit, json_path, csv_path
    )


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
    table2.add_column("Count", style="green", justify="right")
    table2.add_column("%", style="yellow", justify="right")

    oa_true = int(df["isOpenAccess"].sum())
    table2.add_row(
        f"[white]Open Access:[/white] [magenta]True[/magenta]",
        str(oa_true),
        _pct(oa_true),
    )

    if "entry_source" in df.columns:
        for src, cnt in df["entry_source"].fillna("").value_counts().items():
            table2.add_row(
                f"[white]Entry Source:[/white] [magenta]{src}[/magenta]",
                str(int(cnt)),
                _pct(int(cnt)),
            )

    for lang, cnt in df["language"].value_counts().items():
        table2.add_row(
            f"[white]Language:[/white] [magenta]{lang}[/magenta]",
            str(int(cnt)),
            _pct(int(cnt)),
        )

    if "pdf_downloaded" in df.columns:
        dl_existing = int(df["pdf_downloaded"].fillna(False).astype(bool).sum())
        dl_missing = total - dl_existing
        table2.add_row(
            "[white]PDF:[/white] [bold green]existing[/bold green]",
            str(dl_existing),
            _pct(dl_existing),
        )
        table2.add_row(
            "[white]PDF:[/white] [bold red]missing[/bold red]",
            str(dl_missing),
            _pct(dl_missing),
        )

    if "txt_filename" in df.columns:
        txt_existing = int(
            df["txt_filename"].fillna("").astype(str).str.strip().ne("").sum()
        )
        txt_missing = total - txt_existing
        table2.add_row(
            "[white]TXT:[/white] [bold green]existing[/bold green]",
            str(txt_existing),
            _pct(txt_existing),
        )
        table2.add_row(
            "[white]TXT:[/white] [bold red]missing[/bold red]",
            str(txt_missing),
            _pct(txt_missing),
        )

    if "markdown_filename" in df.columns:
        md_existing = int(
            df["markdown_filename"].fillna("").astype(str).str.strip().ne("").sum()
        )
        md_missing = total - md_existing
        table2.add_row(
            "[white]Markdown:[/white] [bold green]existing[/bold green]",
            str(md_existing),
            _pct(md_existing),
        )
        table2.add_row(
            "[white]Markdown:[/white] [bold red]missing[/bold red]",
            str(md_missing),
            _pct(md_missing),
        )

    layout = Table.grid(padding=(0, 2))
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(table1, table2)

    return layout


@app.command()
def stats(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID.",
    ),
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to look up results for. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Display statistics for a completed search result."""

    if file is None:
        raise UnexpectedCLIParamsException("--file is required.")

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    search_id, query, year, only_open_access = read_search_params(file, experiment_path)
    print_search_header(eid, search_id, query, year, only_open_access)

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    csv_path = os.path.join(searches_path, f"{search_id}_results.csv")

    if not os.path.exists(csv_path):
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Results file not found: '{csv_path}'"
        )
        console.print(
            f"Run [bold cyan]search run --file {file} --eid {eid}[/bold cyan] first."
        )
        return

    df = pd.read_csv(csv_path, sep=";")
    console.print()
    tbl_original = _build_stats_grid(df)

    filtered_path = os.path.join(searches_path, f"{search_id}_filtered.csv")
    tbl_filtered = None
    if os.path.exists(filtered_path):
        filtered_df = pd.read_csv(filtered_path, sep=";")
        tbl_filtered = _build_stats_grid(filtered_df)
    else:
        filtered_df = []

    console.print(
        f"[bold]Results:[/bold] [yellow]{Path(csv_path).name}[/yellow] ({len(df)} papers)"
    )
    console.print(tbl_original)
    if tbl_filtered is not None:
        console.print()
        console.print(
            f"[bold]Filtered:[/bold] [green]{Path(filtered_path).name}[/green] ({len(filtered_df)} papers)"
        )
        console.print(tbl_filtered)


@app.command(name="filter")
def filter_results(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID.",
    ),
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to look up results for. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    language: str = typer.Option(
        "en",
        "--language",
        help="Filter by language code (e.g. 'en', 'de'). Default: 'en'.",
    ),
) -> None:
    """Filter search results CSV by language, saving a new _filtered.csv."""

    if file is None:
        raise UnexpectedCLIParamsException("--file is required.")

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    search_id, query, year, only_open_access = read_search_params(file, experiment_path)
    print_search_header(eid, search_id, query, year, only_open_access)

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    csv_path = os.path.join(searches_path, f"{search_id}_results.csv")

    if not os.path.exists(csv_path):
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Results file not found: '{csv_path}'"
        )
        console.print(
            f"Run [bold cyan]search run --file {file} --eid {eid}[/bold cyan] first."
        )
        return

    df = pd.read_csv(csv_path, sep=";")
    filtered_df = df[df["language"] == language]

    total = len(df)
    remaining = len(filtered_df)
    filtered_out = total - remaining
    console.print(f"Language filter: [bold cyan]{language}[/bold cyan]")
    console.print(f"Total: [bold white]{total}[/bold white]")
    console.print(f"Filtered out: [bold red]{filtered_out}[/bold red]")
    console.print(f"Remaining: [bold green]{remaining}[/bold green]")

    filtered_path = os.path.join(searches_path, f"{search_id}_filtered.csv")

    if not settings.dry_run:
        filtered_df.to_csv(filtered_path, index=False, encoding="utf-8", sep=";")
        logger.debug("Wrote filtered results to '%s'", filtered_path)
        console.print(
            f"Saved filtered results to: [bold]{Path(filtered_path).name}[/bold]"
        )
    else:
        console.print(
            f"[bold yellow]Dry run:[/bold yellow] would write '{filtered_path}'"
        )


def _sync_df(df: pd.DataFrame, papers_path: str) -> tuple[pd.DataFrame, int, int]:
    """Sync pdf_downloaded, txt_filename, markdown_filename for existing rows and
    append new rows for PDFs found in papers_path that are not yet in the DataFrame.

    Returns:
        (updated_df, updated_count, added_count)
    """
    papers_dir = Path(papers_path)
    updated_count = 0

    # Ensure string columns stay as object dtype even when all values are NaN
    for col in ("txt_filename", "markdown_filename", "pdf_filename", "entry_source"):
        if col in df.columns and df[col].dtype != object:
            df[col] = df[col].astype(object)

    known_pdf_filenames = set(
        df["pdf_filename"].dropna().astype(str).str.strip().loc[lambda s: s != ""]
    )

    for idx, row in df.iterrows():
        pdf_filename = str(row.get("pdf_filename", "") or "").strip()
        if not pdf_filename:
            continue
        stem = (
            pdf_filename[:-4] if pdf_filename.lower().endswith(".pdf") else pdf_filename
        )

        changed = False
        if (papers_dir / pdf_filename).exists():
            if not df.at[idx, "pdf_downloaded"]:
                df.at[idx, "pdf_downloaded"] = True
                changed = True

        txt_val = row.get("txt_filename", "")
        txt_empty = pd.isna(txt_val) or str(txt_val).strip() == ""
        txt_candidate = f"{stem}.txt"
        if txt_empty and (papers_dir / txt_candidate).exists():
            df.at[idx, "txt_filename"] = txt_candidate
            changed = True

        md_val = row.get("markdown_filename", "")
        md_empty = pd.isna(md_val) or str(md_val).strip() == ""
        md_candidate = f"{stem}.md"
        if md_empty and (papers_dir / md_candidate).exists():
            df.at[idx, "markdown_filename"] = md_candidate
            changed = True

        if changed:
            updated_count += 1

    new_rows = []
    if papers_dir.exists():
        for pdf_path in sorted(papers_dir.glob("*.pdf")):
            pdf_name = pdf_path.name
            if pdf_name in known_pdf_filenames:
                continue
            stem = pdf_path.stem
            txt_exists = (papers_dir / f"{stem}.txt").exists()
            md_exists = (papers_dir / f"{stem}.md").exists()
            new_row = {col: "" for col in _PAPER_CSV_COLUMNS}
            new_row["pdf_filename"] = pdf_name
            new_row["pdf_downloaded"] = True
            new_row["entry_source"] = "manually added"
            new_row["txt_filename"] = f"{stem}.txt" if txt_exists else ""
            new_row["markdown_filename"] = f"{stem}.md" if md_exists else ""
            new_rows.append(new_row)

    added_count = len(new_rows)
    if new_rows:
        df = pd.concat(
            [df, pd.DataFrame(new_rows, columns=_PAPER_CSV_COLUMNS)], ignore_index=True
        )

    return df, updated_count, added_count


@app.command()
def sync(
    file: str = typer.Option(
        None,
        "--file",
        help="YAML search config filename or bare search ID.",
    ),
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Sync CSV result files against the papers/ folder of the experiment."""

    if file is None:
        raise UnexpectedCLIParamsException("--file is required.")

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    search_id, query, year, only_open_access = read_search_params(file, experiment_path)
    print_search_header(eid, search_id, query, year, only_open_access)

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    results_csv = os.path.join(searches_path, f"{search_id}_results.csv")
    filtered_csv = os.path.join(searches_path, f"{search_id}_filtered.csv")

    if not os.path.exists(results_csv):
        console.print(
            f"[bold yellow]Warning:[/bold yellow] Results file not found: '{results_csv}'"
        )
        console.print(
            f"Run [bold cyan]search run --file {file} --eid {eid}[/bold cyan] first."
        )
        return

    df = pd.read_csv(results_csv, sep=";")
    updated_df, updated_count, added_count = _sync_df(df, papers_path)

    if not settings.dry_run:
        updated_df.to_csv(results_csv, index=False, encoding="utf-8", sep=";")
        logger.debug("Synced results CSV: '%s'", results_csv)
    else:
        console.print(
            f"[bold yellow]Dry run:[/bold yellow] would write '{results_csv}'"
        )

    if os.path.exists(filtered_csv):
        filtered_df = pd.read_csv(filtered_csv, sep=";")
        updated_filtered_df, f_updated, f_added = _sync_df(filtered_df, papers_path)
        if not settings.dry_run:
            updated_filtered_df.to_csv(
                filtered_csv, index=False, encoding="utf-8", sep=";"
            )
            logger.debug("Synced filtered CSV: '%s'", filtered_csv)
        else:
            console.print(
                f"[bold yellow]Dry run:[/bold yellow] would write '{filtered_csv}'"
            )

    console.print(f"Rows updated: [bold cyan]{updated_count}[/bold cyan]")
    console.print(f"New rows added: [bold green]{added_count}[/bold green]")
    console.print("[bold]Sync complete.[/bold]")
