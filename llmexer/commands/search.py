"""Search group commands."""

import os
import uuid
from datetime import datetime, timezone

import pandas as pd
import typer
import yaml
from semanticscholar import SemanticScholar

from llmexer.common import ensure_directory_exists
from llmexer.configs import console, settings
from llmexer.constants import EXPERIMENTS_PATH, SEARCHES_DIR
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    LLMExerException,
    UnexpectedCLIParamsException,
)
from llmexer.logger import get_logger

logger = get_logger()

app = typer.Typer(help="Search online digital libraries for papers and metadata.")

# Default values
DEFAULT_QUERY_PARAM = "influence of machine learning on computer science"
DEFAULT_SEARCH_YEAR_PARAM = "2020-2025"
DEFAULT_OPEN_ACCESS_PARAM = False


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


def get_proper_eid(eid: str) -> str:
    """Use current experiment if eid not provided.

    Args:
        eid (str): _description_

    Raises:
        ExperimentIDRequiredException: _description_

    Returns:
        str: _description_
    """

    if eid is None:
        if settings.experiment_id:
            eid = settings.experiment_id
        else:
            raise ExperimentIDRequiredException(
                "No experiment ID provided. Use --eid or set EXPERIMENT_ID in .env file."
            )

    return eid


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

    # Generate search ID and filename
    search_id = generate_search_id()

    yaml_filename = f"search_{search_id}.yaml"
    yaml_path = os.path.join(searches_path, yaml_filename)

    # Create YAML content
    search_config = {"query": query, "year": year, "onlyOpenAccess": only_open_access}

    # Write YAML file
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(search_config, f, default_flow_style=False, sort_keys=False)

    return search_id, yaml_filename


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
    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' does not exist.")

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
        100,
        "--limit",
        help="Maximum number of papers to retrieve (fetched in batches of 100)",
    ),
) -> None:
    """Runs a new search and saves results"""

    if query is not None and file is not None:
        raise UnexpectedCLIParamsException("Only one can be set: query or file")

    eid = get_proper_eid(eid)
    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)
    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' does not exist.")

    if query:
        year = DEFAULT_SEARCH_YEAR_PARAM
        only_open_access = DEFAULT_OPEN_ACCESS_PARAM
        search_id, yaml_filename = save_search_query(
            experiment_path, query, DEFAULT_SEARCH_YEAR_PARAM, DEFAULT_OPEN_ACCESS_PARAM
        )
    if file:
        search_id = file
        # Load parameters from config file if provided
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

    # # Perform search with pagination
    # console.print("\n[bold cyan]Starting search...[/bold cyan]")
    # logger.info(f"Searching Semantic Scholar: query='{query}', year='{year}', open_access={only_open_access}")

    # # Initialize Semantic Scholar client
    # sch = SemanticScholar()

    # papers_data = []
    # save_interval, total_papers, offset = 100, 0, 0
    # batch_size = 100  # API limit per request

    # import logging
    # logging.getLogger("semanticscholar").setLevel(logging.DEBUG)

    # try:
    #     while total_papers < limit:
    #         # Calculate how many papers to fetch in this batch
    #         current_batch_size = min(batch_size, limit - total_papers)

    #         console.print(f"[bold cyan]Fetching papers {offset + 1} to {offset + current_batch_size}...[/bold cyan]")
    #         logger.info(f"Fetching batch: offset={offset}, limit={current_batch_size}")

    #         # Fetch batch
    #         results = sch.search_paper(
    #             query,
    #             year=year,
    #             open_access_pdf=only_open_access,
    #             limit=1,
    #             fields=["paperId", "externalIds", "title", "authors", "abstract", "isOpenAccess", "year"],
    #         )
    #         print("1")
    #         batch_count = 0
    #         for paper in results:
    #             print("2")
    #             # Extract DOI from externalIds
    #             doi = paper.externalIds.get("DOI", "") if paper.externalIds else ""

    #             # Extract author names
    #             authors = ", ".join([author.name for author in paper.authors]) if paper.authors else ""

    #             paper_data = {
    #                 "DOI": doi,
    #                 "TITLE": paper.title or "",
    #                 "AUTHORS": authors,
    #                 "ABSTRACT": paper.abstract or "",
    #                 "IsOpenAccess": paper.isOpenAccess or False,
    #                 "Year": paper.year or "",
    #                 "PaperId": paper.paperId or "",
    #             }
    #             papers_data.append(paper_data)
    #             total_papers += 1
    #             batch_count += 1

    #             # Save every 100 papers
    #             if total_papers % save_interval == 0:
    #                 df = pd.DataFrame(papers_data)
    #                 output_filename = f"search_{search_id}_partial_{total_papers}.csv"
    #                 output_path = os.path.join(searches_path, output_filename)
    #                 df.to_csv(output_path, index=False)
    #                 console.print(f"[bold blue]✓ Saved {total_papers} papers to {output_filename}[/bold blue]")
    #                 logger.info(f"Saved partial results: {total_papers} papers to {output_filename}")

    #         # If we got fewer results than requested, we've reached the end
    #         if batch_count < current_batch_size:
    #             console.print(f"[bold yellow]Reached end of results (found {total_papers} papers)[/bold yellow]")
    #             break

    #         offset += batch_count

    #         # Stop if we've reached the limit
    #         if total_papers >= limit:
    #             break

    # except Exception as e:
    #     logger.error(f"Error searching Semantic Scholar: {e}")
    #     raise LLMExerException(f"Search failed: {e}")

    # # Save final results
    # if papers_data:
    #     df = pd.DataFrame(papers_data)
    #     output_filename = f"search_{search_id}_final.csv"
    #     output_path = os.path.join(searches_path, output_filename)
    #     df.to_csv(output_path, index=False)

    #     console.print(f"\n[bold green]✓ Search completed![/bold green]")
    #     console.print(f"Total papers retrieved: [bold yellow]{total_papers}[/bold yellow]")
    #     console.print(f"Final results saved to: [bold blue]{output_filename}[/bold blue]")
    #     logger.info(f"Search completed: {total_papers} papers saved to {output_filename}")
    # else:
    #     console.print("[bold red]No papers found for this query.[/bold red]")
    #     logger.warning(f"No papers found for query: {query}")
