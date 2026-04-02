"""Search group commands."""

import json
import os
import uuid

import pandas as pd
import requests
import typer
import yaml
from requests.adapters import HTTPAdapter, Retry

from llmexer.common import (
    ensure_directory_exists,
    get_experiment_directory_path,
    get_proper_eid,
)
from llmexer.configs import console, settings
from llmexer.constants import SEARCHES_DIR
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

# Semantic Scholar API constants
_S2_BULK_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
_S2_FIELDS = "paperId,title,authors,abstract,isOpenAccess,externalIds"
_PAPER_CSV_COLUMNS = [
    "s2_paper_id",
    "title",
    "authors",
    "abstract",
    "isOpenAccess",
    "doi",
]


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

    yaml_filename = f"search_{search_id}.yaml"
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

    session = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))

    params: dict = {
        "query": query,
        "fields": _S2_FIELDS,
        "limit": min(batch_size, 1000),
    }

    if year:
        params["year"] = year
    if only_open_access:
        params["openAccessPdf"] = ""

    raw_json_results: list[dict] = []
    records: list[dict] = []

    while True:
        response = session.get(_S2_BULK_URL, params=params)
        response.raise_for_status()
        data = response.json()
        raw_json_results.append(data)

        papers = data.get("data", [])
        for paper in papers:
            if limit_size is not None and len(records) >= limit_size:
                break
            ext_ids = paper.get("externalIds") or {}
            records.append(
                {
                    "s2_paper_id": paper.get("paperId"),
                    "title": paper.get("title"),
                    "authors": "; ".join(
                        a.get("name", "") for a in paper.get("authors", [])
                    ),
                    "abstract": paper.get("abstract"),
                    "isOpenAccess": paper.get("isOpenAccess"),
                    "doi": ext_ids.get("DOI"),
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
    df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")
    logger.debug("Wrote CSV to '%s'", csv_path)

    console.print(f"[bold green]Done:[/bold green] {len(records)} paper(s) saved to:")
    console.print(f"  {json_path}")
    console.print(f"  {csv_path}")

    return records


@app.command()
def new(
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
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
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
    if file:
        # Derive search_id from the YAML filename stem (strip "search_" prefix and ".yaml" suffix)
        file_stem = os.path.splitext(os.path.basename(file))[0]
        search_id = (
            file_stem[len("search_") :]
            if file_stem.startswith("search_")
            else file_stem
        )
        # Load parameters from config file if provided
        if os.path.isabs(file):
            search_file_path = file
        else:
            search_file_path = os.path.join(experiment_path, SEARCHES_DIR, file)

        if not os.path.exists(search_file_path):
            raise LLMExerException(f"Config file '{file}' does not exist.")

        with open(search_file_path, "r", encoding="utf-8") as f:
            search_params = yaml.safe_load(f)

        query = search_params.get("query", query)
        year = search_params.get("year", DEFAULT_SEARCH_YEAR_PARAM)
        only_open_access = search_params.get(
            "onlyOpenAccess", DEFAULT_OPEN_ACCESS_PARAM
        )

        console.print(f"Loaded config from: [bold blue]{file}[/bold blue]")
    elif not query:
        raise LLMExerException(
            "No query provided. Use --query or --file with a YAML file."
        )

    console.print(f"Experiment: [bold yellow]{eid}[/bold yellow]")
    console.print(f"Search ID: [bold yellow]{search_id}[/bold yellow]")
    console.print(f"Query: [bold green]{query}[/bold green]")
    console.print(f"Year: [bold cyan]{year}[/bold cyan]")
    console.print(f"Only Open Access: [bold magenta]{only_open_access}[/bold magenta]")
    console.print(f"Limit: [bold blue]{limit}[/bold blue]")

    searches_path = os.path.join(experiment_path, SEARCHES_DIR)
    ensure_directory_exists(searches_path)

    json_path = os.path.join(searches_path, f"{search_id}_results_raw.json")
    csv_path = os.path.join(searches_path, f"{search_id}_results.csv")

    if not force_overwrite:
        existing = [p for p in [json_path, csv_path] if os.path.exists(p)]
        if existing:
            raise SearchResultsAlreadyExistException(
                f"Result files already exist: {existing}. Use --force-rewrite to overwrite."
            )

    if settings.dry_run:
        console.print(f"[bold yellow]Dry run:[/bold yellow] would write '{json_path}'")
        console.print(f"[bold yellow]Dry run:[/bold yellow] would write '{csv_path}'")
        return

    run_semantic_scholar_search(
        query, year, only_open_access, batch, limit, json_path, csv_path
    )
