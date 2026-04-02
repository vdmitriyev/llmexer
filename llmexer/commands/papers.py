"""Search group commands."""

import os
import shutil
from pathlib import Path
from typing import Optional

import requests
import typer

from llmexer.common import ensure_directory_exists
from llmexer.configs import console, logger, settings
from llmexer.constants import EXPERIMENTS_PATH, PAPERS_DIR
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    PaperAddException,
    PaperAlreadyExistsException,
    UnexpectedCLIParamsException,
)

app = typer.Typer(help="Work with papers.")


@app.command()
def rename(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to be used to store search results. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Renames papers of the given experiment"""

    # Use current experiment if eid not provided
    if eid is None:
        if settings.experiment_id:
            eid = settings.experiment_id
        else:
            raise ExperimentIDRequiredException(
                "No experiment ID provided. Use --eid or set EXPERIMENT_ID in .env file."
            )

    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")


@app.command()
def add(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to add papers to. If not provided, uses EXPERIMENT_ID from .env.",
    ),
    file: Optional[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a PDF file to copy into the papers subdirectory.",
    ),
    directory: Optional[Path] = typer.Option(
        None,
        "--directory",
        "-d",
        help="Path to a directory; all PDF files found recursively will be copied.",
    ),
    url: Optional[str] = typer.Option(
        None,
        "--url",
        help="URL of a PDF to download into the papers subdirectory.",
    ),
) -> None:
    """Adds PDF paper(s) to the papers subdirectory of the current experiment."""

    provided = sum(p is not None for p in [file, directory, url])
    if provided != 1:
        raise UnexpectedCLIParamsException(
            "Exactly one of --file, --directory, or --url must be provided."
        )

    if eid is None:
        if settings.experiment_id:
            eid = settings.experiment_id
        else:
            raise ExperimentIDRequiredException(
                "No experiment ID provided. Use --eid or set EXPERIMENT_ID in .env file."
            )

    experiment_path = os.path.join(EXPERIMENTS_PATH, eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")

    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    ensure_directory_exists(papers_path)

    if file is not None:
        src = Path(file).resolve()
        if not src.exists() or src.suffix.lower() != ".pdf":
            raise PaperAddException(f"'{file}' does not exist or is not a PDF file.")

        dst = os.path.join(papers_path, src.name)
        if os.path.exists(dst):
            console.print(
                f"A paper already exists in the papers directory: [bold yellow]{src.name}[/bold yellow]"
            )
            return

        if not settings.dry_run:
            shutil.copy2(str(src), dst)

        logger.debug("Copied '%s' -> '%s'", src, dst)
        console.print(
            f"[bold green]Added[/bold green] '{src.name}' to experiment '{eid}'."
        )

    elif directory is not None:
        dir_path = Path(directory).resolve()
        if not dir_path.is_dir():
            raise PaperAddException(f"'{directory}' is not a valid directory.")
        pdfs = [
            Path(root) / fname
            for root, _, files in os.walk(dir_path)
            for fname in files
            if fname.lower().endswith(".pdf")
        ]
        already_exists = []
        for index, src in enumerate(pdfs):
            dst = os.path.join(papers_path, src.name)
            if os.path.exists(dst):
                console.print(
                    f"A paper already exists in the papers directory [{index+1}/{len(pdfs)}]: [bold yellow]{src.name}[/bold yellow]"
                )
                already_exists.append(src.name)
        copied_papers_cnt = 0
        if not settings.dry_run:
            for src in pdfs:
                if src.name not in already_exists:
                    console.print(f"Copying paper: [bold green]{src.name}[/bold green]")
                    dst = os.path.join(papers_path, src.name)
                    shutil.copy2(str(src), dst)
                    logger.debug("Copied '%s' -> '%s'", src, dst)
                    copied_papers_cnt += 1

        console.print(
            f"[bold green]Added[/bold green] {copied_papers_cnt} PDF(s) to experiment '{eid}'."
        )

    else:  # url
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise PaperAddException(f"Failed to download '{url}': {exc}") from exc

        # Determine filename: Content-Disposition > final URL path > original URL path
        filename = None
        content_disposition = response.headers.get("Content-Disposition", "")
        if "filename=" in content_disposition:
            for part in content_disposition.split(";"):
                part = part.strip()
                if part.lower().startswith("filename="):
                    filename = part.split("=", 1)[1].strip().strip('"')
                    break
        if not filename:
            filename = Path(response.url.split("?")[0]).name
        if not filename:
            filename = Path(url.split("?")[0]).name

        if not filename.lower().endswith(".pdf"):
            raise PaperAddException(
                f"Could not resolve a PDF filename from URL '{url}' "
                f"(resolved name: '{filename}')."
            )

        dst = os.path.join(papers_path, filename)
        if os.path.exists(dst):
            raise PaperAlreadyExistsException(
                f"A paper named '{filename}' already exists in the papers directory."
            )
        if not settings.dry_run:
            try:
                with open(dst, "wb") as fh:
                    for chunk in response.iter_content(chunk_size=8192):
                        fh.write(chunk)
            except requests.RequestException as exc:
                raise PaperAddException(f"Failed to download '{url}': {exc}") from exc
            logger.debug("Downloaded '%s' -> '%s'", url, dst)
        console.print(
            f"[bold green]Downloaded[/bold green] '{filename}' to experiment '{eid}'."
        )
