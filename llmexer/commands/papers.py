"""Papers group commands of the CLI interface."""

import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pypdf
import typer

from llmexer.base.papers import (
    UNPAYWALL_EMAIL,
    PDFProcessor,
    download_pdf_from_url,
    extract_via_docling,
    resolve_unpaywall_pdf_url,
)
from llmexer.common import (
    ensure_directory_exists,
    get_project_directory_path,
    get_proper_pid,
)
from llmexer.configs import console, cprint, logger, settings
from llmexer.constants import (
    DEFAULT_DOCLING_URL,
    PAPERS_DIR,
    SEARCHES_DIR,
    SEARCHES_LOGS_DIR,
)
from llmexer.exceptions import (
    PaperAddException,
    PaperAlreadyExistsException,
    PaperDownloadException,
    PaperExtractException,
    ProjectNotExistsException,
    UnexpectedCLIParamsException,
)

app = typer.Typer(help="Work with papers.")


@app.command()
def add(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to add papers to. If not provided, uses PROJECT_ID from .env.",
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
    """Adds PDF paper(s) to the papers subdirectory of the current project"""

    provided = sum(p is not None for p in [file, directory, url])
    if provided != 1:
        raise UnexpectedCLIParamsException(
            "Exactly one of --file, --directory, or --url must be provided."
        )

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    ensure_directory_exists(papers_path)

    if file is not None:
        src = Path(file).resolve()
        if not src.exists() or src.suffix.lower() != ".pdf":
            raise PaperAddException(f"'{file}' does not exist or is not a PDF file.")

        dst = os.path.join(papers_path, src.name)
        if os.path.exists(dst):
            cprint(
                f"A paper already exists in the papers directory: [bold yellow]{src.name}[/bold yellow]"
            )
            return

        if not settings.dry_run:
            shutil.copy2(str(src), dst)

        logger.debug("Copied '%s' -> '%s'", src, dst)
        cprint(f"[bold green]Added[/bold green] '{src.name}' to project '{pid}'.")

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
                cprint(
                    f"A paper already exists in the papers directory [{index+1}/{len(pdfs)}]: [bold yellow]{src.name}[/bold yellow]"
                )
                already_exists.append(src.name)
        copied_papers_cnt = 0
        if not settings.dry_run:
            for src in pdfs:
                if src.name not in already_exists:
                    cprint(f"Copying paper: [bold green]{src.name}[/bold green]")
                    dst = os.path.join(papers_path, src.name)
                    shutil.copy2(str(src), dst)
                    logger.debug("Copied '%s' -> '%s'", src, dst)
                    copied_papers_cnt += 1

        cprint(
            f"[bold green]Added[/bold green] {copied_papers_cnt} PDF(s) to project '{pid}'."
        )

    else:  # url
        filename = download_pdf_from_url(url, papers_path)
        cprint(f"[bold green]Downloaded[/bold green] '{filename}' to project '{pid}'.")


@app.command()
def download(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to download papers into. If not provided, uses PROJECT_ID from .env.",
    ),
    doi: List[str] = typer.Option(
        None,
        "--doi",
        help="DOI of the paper to download. Can be specified multiple times.",
    ),
    search_file: Optional[str] = typer.Option(
        None,
        "--search-file",
        help="Search results CSV filename inside searches/ (e.g. '20260401-bfdd863d__results.csv' or '20260401-bfdd863d__filtered.csv'). "
        "Iterates all rows and downloads each paper by DOI.",
    ),
    email: Optional[str] = typer.Option(
        UNPAYWALL_EMAIL,
        "--email",
        help="Email address for Unpaywall API. Falls back to UNPAYWALL_EMAIL env var.",
    ),
) -> None:
    """Download open-access PDF(s) by DOI using the Unpaywall API"""

    provided = sum(p is not None and p != [] for p in [doi or None, search_file])
    if provided != 1:
        raise UnexpectedCLIParamsException(
            "Exactly one of --doi or --search-file must be provided."
        )

    resolved_email = email or os.getenv("UNPAYWALL_EMAIL")
    if not resolved_email:
        raise PaperDownloadException(
            "Unpaywall email is required. Use --email or set UNPAYWALL_EMAIL in .env."
        )

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    papers_path = os.path.join(experiment_path, PAPERS_DIR)
    ensure_directory_exists(papers_path)

    # Build the list of (doi, pdf_filename, title) triples to process
    download_items: List[tuple] = []  # (doi_str, pdf_filename, title_str)

    use_forced_name = False
    failed_csv_path: Optional[str] = None
    if doi:
        for single_doi in doi:
            sanitized = "".join(
                c if (c.isalnum() or c in "-_.") else "_" for c in single_doi
            )
            download_items.append((single_doi, f"{sanitized}.pdf", None))
    else:
        use_forced_name = True
        searches_path = os.path.join(experiment_path, SEARCHES_DIR)
        csv_path = os.path.join(searches_path, search_file)
        if not os.path.exists(csv_path):
            raise PaperDownloadException(
                f"Search file '{search_file}' not found in searches/ directory for project '{pid}'."
            )
        stem = Path(search_file).stem
        logs_path = os.path.join(searches_path, SEARCHES_LOGS_DIR)
        failed_csv_path = os.path.join(logs_path, f"{stem}_download_failed.csv")
        df = pd.read_csv(csv_path, sep=";")
        for _, row in df.iterrows():
            single_doi = row.get("doi")
            if not single_doi or (isinstance(single_doi, float)):
                continue
            title = row.get("title")
            pdf_filename = str(row.get("pdf_filename"))
            download_items.append(
                (str(single_doi), pdf_filename, str(title) if title else None)
            )

    succeeded, failed, exists, skipped = 0, 0, 0, 0
    cnt = len(download_items)
    failed_records: List[dict] = []
    succeeded_dois: set[str] = set()

    for index, (single_doi, pdf_filename, item_title) in enumerate(download_items):
        label = f"[{index+1}/{cnt}]"

        pdf_url = None
        try:
            pdf_url = resolve_unpaywall_pdf_url(single_doi, resolved_email)
        except PaperDownloadException as exc:
            cprint(f"{label} [bold yellow]skipped[/bold yellow] '{single_doi}': {exc}")
            skipped += 1
            failed_records.append(
                {
                    "doi": single_doi,
                    "url": None,
                    "title": item_title,
                    "pdf_filename": pdf_filename,
                    "pdf_downloaded": False,
                }
            )
            continue

        download_kwargs = (
            {"forced_name": pdf_filename}
            if use_forced_name
            else {"fallback_name": pdf_filename}
        )
        try:
            filename = download_pdf_from_url(pdf_url, papers_path, **download_kwargs)
        except PaperAlreadyExistsException as exc:
            cprint(f"{label} [bold green]exists[/bold green] '{single_doi}': {exc}")
            exists += 1
            failed_records.append(
                {
                    "doi": single_doi,
                    "url": pdf_url,
                    "title": item_title,
                    "pdf_filename": pdf_filename,
                    "pdf_downloaded": True,
                }
            )
            continue
        except PaperAddException as exc:
            cprint(f"{label} [bold red]failed[/bold red] '{single_doi}': {exc}")
            failed += 1
            failed_records.append(
                {
                    "doi": single_doi,
                    "url": pdf_url,
                    "title": item_title,
                    "pdf_filename": pdf_filename,
                    "pdf_downloaded": False,
                }
            )
            continue

        logger.debug("Downloaded DOI '%s' -> '%s'", single_doi, filename)
        cprint(
            f"{label} [bold green]downloaded[/bold green] '{single_doi}' as '{filename}'."
        )
        succeeded += 1
        succeeded_dois.add(single_doi)

    if failed_csv_path and not settings.dry_run:
        if failed_records:
            ensure_directory_exists(os.path.dirname(failed_csv_path))
            pd.DataFrame(
                failed_records,
                columns=["doi", "title", "url", "pdf_filename", "pdf_downloaded"],
            ).to_csv(failed_csv_path, index=False, encoding="utf-8", sep=";")
            logger.debug("Saved failed records to '%s'", failed_csv_path)
            cprint(f"Failed list saved to: [bold]{Path(failed_csv_path).name}[/bold]")
        if succeeded_dois:
            df.loc[df["doi"].isin(succeeded_dois), "pdf_downloaded"] = True
            df.to_csv(csv_path, index=False, encoding="utf-8", sep=";")
            logger.debug("Updated 'downloaded' status in '%s'", csv_path)

    cprint(
        f"Downloaded: [bold green]{succeeded}[/bold green]. Exists: [bold green]{exists}[/bold green]. Skipped: [bold yellow]{skipped}[/bold yellow]. Failed: [bold red]{failed}[/bold red]"
    )


@app.command()
def extract(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID to extract papers from. If not provided, uses PROJECT_ID from .env.",
    ),
    processor: PDFProcessor = typer.Option(
        PDFProcessor.pypdf,
        "--processor",
        help="Extraction backend: 'pypdf' (local, saves .txt) or 'docling' (remote server, saves .md).",
    ),
    docling_url: Optional[str] = typer.Option(
        None,
        "--docling-url",
        help=f"Docling server URL. Overrides DOCLING_URL from .env[default: {DEFAULT_DOCLING_URL}].",
    ),
    docling_user: Optional[str] = typer.Option(
        None,
        "--docling-user",
        help="Docling server username. Overrides DOCLING_USER from .env.",
    ),
    docling_password: Optional[str] = typer.Option(
        None,
        "--docling-password",
        help="Docling server password. Overrides DOCLING_PASSWORD from .env.",
    ),
    rewrite: bool = typer.Option(
        False,
        "--rewrite",
        help="Force rewrite of existing extracted files. By default, already-extracted files are skipped.",
    ),
) -> None:
    """Extracts text from all PDFs in the papers subdirectory and saves as .txt (pypdf) or .md (docling) files"""

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    if not os.path.exists(experiment_path):
        raise ProjectNotExistsException(f"Project '{pid}' not exist.")

    papers_path = os.path.join(experiment_path, PAPERS_DIR)

    if not os.path.exists(papers_path):
        cprint(
            f"[bold yellow]Warning:[/bold yellow] No papers directory found for project '{pid}'."
        )
        return

    pdfs = [
        Path(papers_path) / fname
        for fname in os.listdir(papers_path)
        if fname.lower().endswith(".pdf")
    ]

    if not pdfs:
        cprint(
            f"[bold yellow]Warning:[/bold yellow] No PDF files found in papers directory for project '{pid}'."
        )
        return

    if processor == PDFProcessor.docling:
        resolved_url = docling_url or os.getenv("DOCLING_URL") or DEFAULT_DOCLING_URL
        resolved_user = docling_user or os.getenv("DOCLING_USER") or ""
        resolved_password = docling_password or os.getenv("DOCLING_PASSWORD") or ""
        docling_auth = (resolved_user, resolved_password)
        cprint(f"Processor: [bold blue]docling[/bold blue] ({resolved_url})")
    else:
        cprint("Processor: [bold blue]pypdf[/bold blue]")

    processed = 0
    skipped = 0
    cnt_pdfs = len(pdfs)
    for index, pdf_path in enumerate(pdfs):
        stem = pdf_path.stem
        label = f"[{index+1}/{cnt_pdfs}]"

        with console.status(f"{label} Processing '{pdf_path.name}'..."):
            if processor == PDFProcessor.pypdf:
                txt_path = pdf_path.parent / f"{stem}.txt"

                if not rewrite and txt_path.exists():
                    cprint(
                        f"{label} [bold blue]skipped[/bold blue] '{pdf_path.name}' (already extracted)."
                    )
                    skipped += 1
                    continue

                try:
                    reader = pypdf.PdfReader(str(pdf_path))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as exc:
                    cprint(
                        f"{label} [bold yellow]skipped[/bold yellow] '{pdf_path.name}'. Extraction failed ({exc})."
                    )
                    logger.debug("Extraction failed for '%s': %s", pdf_path, exc)
                    skipped += 1
                    continue

                if not text.strip():
                    cprint(
                        f"{label} [bold yellow]warning:[/bold yellow] '{pdf_path.name}' could not be parsed - no text content found. No .txt file written."
                    )
                    logger.debug(
                        "Empty content after extraction for '%s', skipping write.",
                        pdf_path,
                    )
                    skipped += 1
                    continue

                if not settings.dry_run:
                    txt_path.write_text(text, encoding="utf-8")
                    logger.debug("Wrote '%s'", txt_path)

            else:  # docling
                md_path = pdf_path.parent / f"{stem}.md"

                if not rewrite and md_path.exists():
                    cprint(
                        f"{label} [bold blue]skipped[/bold blue] '{pdf_path.name}' (already extracted)."
                    )
                    skipped += 1
                    continue

                try:
                    md_content = extract_via_docling(
                        pdf_path, resolved_url, docling_auth
                    )
                except PaperExtractException as exc:
                    cprint(
                        f"{label} [bold yellow]skipped[/bold yellow] '{pdf_path.name}'. Extraction failed ({exc})."
                    )
                    logger.debug(
                        "Docling extraction failed for '%s': %s", pdf_path, exc
                    )
                    skipped += 1
                    continue

                if not md_content.strip():
                    cprint(
                        f"{label} [bold yellow]warning:[/bold yellow] '{pdf_path.name}' could not be parsed - no text content found. No .md file written."
                    )
                    logger.debug(
                        "Empty content after docling extraction for '%s', skipping write.",
                        pdf_path,
                    )
                    skipped += 1
                    continue

                if not settings.dry_run:
                    md_path.write_text(md_content, encoding="utf-8")
                    logger.debug("Wrote '%s'", md_path)

        cprint(f"{label} [bold green]extracted:[/bold green] '{pdf_path.name}'")
        processed += 1

    cprint(
        f"Extracted: [bold green]{processed}[/bold green]. Skipped: [bold red]{skipped}[/bold red]"
    )
