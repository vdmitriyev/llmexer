"""Search group commands."""

import os
import shutil
from pathlib import Path
from typing import Optional

import pypdf
import requests
import typer

from llmexer.common import (
    ensure_directory_exists,
    get_experiment_directory_path,
    get_proper_eid,
)
from llmexer.configs import console, logger, settings
from llmexer.constants import EXPERIMENTS_PATH, PAPERS_DIR
from llmexer.exceptions import (
    ExperimentIDRequiredException,
    ExperimentNotExistsException,
    PaperAddException,
    PaperAlreadyExistsException,
    PaperExtractException,
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

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

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

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

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


@app.command()
def extract(
    eid: str = typer.Option(
        None,
        "--eid",
        help="Experiment ID to extract papers from. If not provided, uses EXPERIMENT_ID from .env.",
    ),
) -> None:
    """Extracts text from all PDFs in the papers subdirectory and saves as .txt files."""

    eid = get_proper_eid(eid)
    experiment_path = get_experiment_directory_path(eid)

    if not os.path.exists(experiment_path):
        raise ExperimentNotExistsException(f"Experiment '{eid}' not exist.")

    papers_path = os.path.join(experiment_path, PAPERS_DIR)

    if not os.path.exists(papers_path):
        console.print(
            f"[bold yellow]Warning:[/bold yellow] No papers directory found for experiment '{eid}'."
        )
        return

    pdfs = [
        Path(papers_path) / fname
        for fname in os.listdir(papers_path)
        if fname.lower().endswith(".pdf")
    ]

    if not pdfs:
        console.print(
            f"[bold yellow]Warning:[/bold yellow] No PDF files found in papers directory for experiment '{eid}'."
        )
        return

    processed = 0
    skipped = 0
    cnt_pdfs = len(pdfs)
    for index, pdf_path in enumerate(pdfs):
        stem = pdf_path.stem
        txt_path = pdf_path.parent / f"{stem}.txt"

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as exc:
            console.print(
                f"[{index+1}/{cnt_pdfs}] [bold yellow]skipped[/bold yellow] '{pdf_path.name}'. Extraction failed ({exc})."
            )
            logger.debug("Extraction failed for '%s': %s", pdf_path, exc)
            skipped += 1
            continue

        if not text.strip():
            console.print(
                f"[{index+1}/{cnt_pdfs}] [bold yellow]warning:[/bold yellow] '{pdf_path.name}' could not be parsed - no text content found. No .txt file written."
            )
            logger.debug(
                "Empty content after extraction for '%s', skipping write.", pdf_path
            )
            skipped += 1
            continue

        if not settings.dry_run:
            txt_path.write_text(text, encoding="utf-8")
            logger.debug("Wrote '%s'", txt_path)

        console.print(
            f"[{index+1}/{cnt_pdfs}] [bold green]extracted:[/bold green] '{pdf_path.name}'"
        )
        processed += 1

    console.print(
        f"Extracted: [bold green]{processed}[/bold green]. Skipped: [bold red]{skipped}[/bold red]"
    )
