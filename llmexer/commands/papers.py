"""Papers group commands of the CLI interface."""

import os
import shutil
from pathlib import Path
from typing import List, Optional

import pandas as pd
import pypdf
import typer
from rich.table import Table

from llmexer.base.papers import (
    PAPER_TYPES,
    UNPAYWALL_EMAIL,
    PaperEntry,
    PaperSortBy,
    PDFProcessor,
    download_pdf_from_url,
    extract_via_docling,
    format_created_timestamp,
    is_paper_file,
    list_paper_files,
    resolve_unpaywall_pdf_url,
    scan_papers,
)
from llmexer.common import (
    ensure_directory_exists,
    format_bytes_console_ui,
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

app = typer.Typer(help="Manages the papers.")


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
        raise UnexpectedCLIParamsException("Exactly one of --file, --directory, or --url must be provided.")

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
            cprint(f"A paper already exists in the papers directory: [bold yellow]{src.name}[/bold yellow]")
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

        cprint(f"[bold green]Added[/bold green] {copied_papers_cnt} PDF(s) to project '{pid}'.")

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
        raise UnexpectedCLIParamsException("Exactly one of --doi or --search-file must be provided.")

    resolved_email = email or os.getenv("UNPAYWALL_EMAIL")
    if not resolved_email:
        raise PaperDownloadException("Unpaywall email is required. Use --email or set UNPAYWALL_EMAIL in .env.")

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
            sanitized = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in single_doi)
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
            download_items.append((str(single_doi), pdf_filename, str(title) if title else None))

    succeeded, failed, exists, skipped = 0, 0, 0, 0
    cnt = len(download_items)
    failed_records: List[dict] = []

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

        download_kwargs = {"forced_name": pdf_filename} if use_forced_name else {"fallback_name": pdf_filename}
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
        cprint(f"{label} [bold green]downloaded[/bold green] '{single_doi}' as '{filename}'.")
        succeeded += 1

    if failed_csv_path and failed_records and not settings.dry_run:
        ensure_directory_exists(os.path.dirname(failed_csv_path))
        pd.DataFrame(
            failed_records,
            columns=["doi", "title", "url", "pdf_filename", "pdf_downloaded"],
        ).to_csv(failed_csv_path, index=False, encoding="utf-8", sep=";")
        logger.debug("Saved failed records to '%s'", failed_csv_path)
        cprint(f"Failed list saved to: [bold]" f"{os.path.join(SEARCHES_LOGS_DIR, Path(failed_csv_path).name)}[/bold]")

    cprint(
        f"Downloaded: [bold green]{succeeded}[/bold green]. Exists: [bold green]{exists}[/bold green]. Skipped: [bold yellow]{skipped}[/bold yellow]. Failed: [bold red]{failed}[/bold red]"
    )

    # When downloading from a search file, reconcile the whole search against papers/.
    if search_file and not settings.dry_run:
        from llmexer.commands.search import _search_id_from_file, _sync_search

        _sync_search(_search_id_from_file(search_file), experiment_path, add_new_rows=False)


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
    skip_if_md: bool = typer.Option(
        False,
        "--skip-if-md",
        help="pypdf only: skip PDFs that already have a .md (markdown) extraction, so TXT is "
        "produced only for PDFs without markdown. Ignored for the docling processor.",
    ),
) -> None:
    """Extracts text from all PDFs in the papers subdirectory and saves as .txt (pypdf) or .md (docling) files"""

    pid = get_proper_pid(pid)
    experiment_path = get_project_directory_path(pid)

    if not os.path.exists(experiment_path):
        raise ProjectNotExistsException(f"Project '{pid}' not exist.")

    papers_path = os.path.join(experiment_path, PAPERS_DIR)

    if not os.path.exists(papers_path):
        cprint(f"[bold yellow]Warning:[/bold yellow] No papers directory found for project '{pid}'.")
        return

    pdfs = [Path(papers_path) / fname for fname in os.listdir(papers_path) if fname.lower().endswith(".pdf")]

    if not pdfs:
        cprint(f"[bold yellow]Warning:[/bold yellow] No PDF files found in papers directory for project '{pid}'.")
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
    exists = 0
    errors = 0
    cnt_pdfs = len(pdfs)
    for index, pdf_path in enumerate(pdfs):
        stem = pdf_path.stem
        label = f"[{index+1}/{cnt_pdfs}]"

        with console.status(f"{label} Processing '{pdf_path.name}'..."):

            if processor == PDFProcessor.pypdf:
                txt_path = pdf_path.parent / f"{stem}.txt"
                md_path = pdf_path.parent / f"{stem}.md"

                if skip_if_md and md_path.exists():
                    cprint(f"{label} [bold blue]existing[/bold blue] '{pdf_path.name}' (markdown already extracted).")
                    exists += 1
                    continue

                if not rewrite and txt_path.exists():
                    cprint(f"{label} [bold blue]existing[/bold blue] '{pdf_path.name}' (already extracted).")
                    exists += 1
                    continue

                try:
                    reader = pypdf.PdfReader(str(pdf_path))
                    text = "\n".join(page.extract_text() or "" for page in reader.pages)
                except Exception as exc:
                    cprint(f"{label} [bold red]error[/bold red] '{pdf_path.name}'. Extraction failed ({exc}).")
                    logger.debug("Extraction failed for '%s': %s", pdf_path, exc)
                    errors += 1
                    continue

                if not text.strip():
                    cprint(
                        f"{label} [bold yellow]skipped[/bold yellow] '{pdf_path.name}' could not be parsed - no text content found. No .txt file written."
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
                    cprint(f"{label} [bold blue]existing[/bold blue] '{pdf_path.name}' (already extracted).")
                    exists += 1
                    continue

                try:
                    md_content = extract_via_docling(pdf_path, resolved_url, docling_auth)
                except PaperExtractException as exc:
                    cprint(f"{label} [bold red]error[/bold red] '{pdf_path.name}'. Extraction failed ({exc}).")
                    logger.debug("Docling extraction failed for '%s': %s", pdf_path, exc)
                    errors += 1
                    continue

                if not md_content.strip():
                    cprint(
                        f"{label} [bold yellow]skipped[/bold yellow] '{pdf_path.name}' could not be parsed - no text content found. No .md file written."
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
        f"Extracted: [bold green]{processed}[/bold green]. "
        f"Skipped: [bold yellow]{skipped}[/bold yellow]. "
        f"Existing: [bold blue]{exists}[/bold blue]. "
        f"Error: [bold red]{errors}[/bold red]"
    )


# Badge colour per file type in `papers view`; any other type is magenta.
_TYPE_STYLES = {"pdf": "bold blue", "txt": "bold cyan", "md": "bold green"}

_SORT_BY_HELP = "Sort papers by 'alpha' (alphabetical), 'date' (creation date) or 'size' (file size)."
_DESC_HELP = "Sort in descending order."


def _papers_path(pid: str) -> str:
    return os.path.join(get_project_directory_path(pid), PAPERS_DIR)


def _describe_sort(sort_by: PaperSortBy, desc: bool) -> str:
    if sort_by == PaperSortBy.date:
        return "sorted by creation date, " + ("newest first" if desc else "oldest first")
    if sort_by == PaperSortBy.size:
        return "sorted by PDF size, " + ("largest first" if desc else "smallest first")
    return "sorted alphabetically, " + ("Z to A" if desc else "A to Z")


def _type_badge(ext: str) -> str:
    style = _TYPE_STYLES.get(ext, "bold magenta")
    return f"[{style}]{ext.upper() or '?'}[/{style}]"


def _size_cell(size: Optional[int], italic: bool = False) -> str:
    if size is None:
        return "[dim]—[/dim]"
    formatted = format_bytes_console_ui(size)
    return f"[italic]{formatted}[/italic]" if italic else formatted


def _created_cell(entry: PaperEntry) -> str:
    """The PDF's creation time. A paper without a PDF names the file the time comes from."""
    created_at = format_created_timestamp(entry.created)
    if entry.has_pdf:
        return created_at
    return f"{created_at} {_type_badge(entry.created_type)}"


@app.command(name="list")
def list_papers(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    sort_by: PaperSortBy = typer.Option(PaperSortBy.alpha, "--sort-by", help=_SORT_BY_HELP),
    desc: bool = typer.Option(False, "--desc", help=_DESC_HELP),
    created: bool = typer.Option(False, "--created/--no-created", help="Show the creation time of each file."),
) -> None:
    """List the files in the papers subdirectory of the project"""

    pid = get_proper_pid(pid)
    papers_path = _papers_path(pid)

    files = list_paper_files(papers_path, sort_by, desc)
    if not files:
        cprint("No papers found.")
        return

    for path in files:
        line = f"[green]{path.name}[/green]"
        if created:
            created_at = format_created_timestamp(path.stat().st_ctime)
            line += f"  [cyan]{created_at}[/cyan]"
        cprint(line, highlight=False)


@app.command(name="view", hidden=True)
@app.command(name="table", hidden=True)
@app.command(name="show")
def view(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
    sort_by: PaperSortBy = typer.Option(PaperSortBy.alpha, "--sort-by", help=_SORT_BY_HELP),
    desc: bool = typer.Option(False, "--desc", help=_DESC_HELP),
) -> None:
    """Show the papers of the project as a table with types and sizes (aliases: view, table)"""

    pid = get_proper_pid(pid)
    papers_path = _papers_path(pid)

    entries = scan_papers(papers_path, sort_by, desc, only_paper_types=True)
    other_files = [path for path in list_paper_files(papers_path, sort_by, desc) if not is_paper_file(path)]
    if not entries and not other_files:
        cprint("No papers found.")
        return

    header = Table(show_header=False, box=None, padding=(0, 1))
    header.add_column(style="bold white", no_wrap=True)
    header.add_column()
    header.add_row("Project:", f"[bold yellow]{pid}[/bold yellow]")
    header.add_row("Folder:", f"[cyan]{papers_path}[/cyan]")
    console.print(header)

    if entries:
        cprint()
        console.print(_build_papers_table(entries, sort_by, desc))

    if other_files:
        cprint()
        console.print(_build_other_files_table(other_files, sort_by, desc))

    cprint("\nExample to view papers stats:")
    cprint(f"[bold yellow]llmexer papers stats --pid {pid}[/bold yellow]")


def _build_papers_table(entries: List[PaperEntry], sort_by: PaperSortBy, desc: bool) -> Table:
    """One row per paper: its PDF, TXT and MD files, their sizes and the PDF creation time."""

    pdf_total = sum(entry.size("pdf") or 0 for entry in entries)
    txt_total = sum(entry.size("txt") or 0 for entry in entries)
    md_total = sum(entry.size("md") or 0 for entry in entries)
    file_count = sum(len(entry.files) for entry in entries)

    table = Table(
        title="Papers",
        caption=_describe_sort(sort_by, desc),
        show_footer=True,
        header_style="bold",
        footer_style="bold",
    )
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Paper", footer=f"{len(entries)} paper(s), {file_count} file(s)", overflow="fold")
    table.add_column("Type", no_wrap=True)
    table.add_column("PDF size", justify="right", no_wrap=True, footer=format_bytes_console_ui(pdf_total))
    table.add_column("TXT size", justify="right", no_wrap=True, footer=format_bytes_console_ui(txt_total))
    table.add_column("MD size", justify="right", no_wrap=True, footer=format_bytes_console_ui(md_total))
    table.add_column("Created (PDF)", style="cyan", no_wrap=True)

    for index, entry in enumerate(entries, start=1):
        types = [ext for ext in PAPER_TYPES if ext in entry.files]
        badges = " ".join(_type_badge(ext) for ext in types)
        table.add_row(
            str(index),
            entry.stem,
            badges,
            _size_cell(entry.size("pdf"), italic=True),
            _size_cell(entry.size("txt")),
            _size_cell(entry.size("md"), italic=True),
            _created_cell(entry),
        )

    return table


def _build_other_files_table(files: List[Path], sort_by: PaperSortBy, desc: bool) -> Table:
    """One row per file in ``papers/`` that is not a PDF, TXT or MD."""

    total_size = sum(path.stat().st_size for path in files)

    table = Table(
        title="Other files",
        caption=_describe_sort(sort_by, desc).replace("PDF size", "file size"),
        show_footer=True,
        header_style="bold",
        footer_style="bold",
    )
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("File", footer=f"{len(files)} file(s)", overflow="fold")
    table.add_column("Type", no_wrap=True)
    table.add_column("Size", justify="right", no_wrap=True, footer=format_bytes_console_ui(total_size))
    table.add_column("Created", style="cyan", no_wrap=True)

    for index, path in enumerate(files, start=1):
        stat = path.stat()
        table.add_row(
            str(index),
            path.name,
            _type_badge(path.suffix.lower().lstrip(".")),
            format_bytes_console_ui(stat.st_size),
            format_created_timestamp(stat.st_ctime),
        )

    return table


def _build_papers_stats_grid(entries: List[PaperEntry]) -> Table:
    """Build the 2-column stats grid (Papers by Year + Stats Breakdown) for the papers of a project."""

    total = len(entries)

    def _pct(n):
        return f"{round(n / total * 100, 1)}%" if total else "0%"

    year_counts = pd.Series([entry.year or "unknown" for entry in entries]).value_counts()
    known_years = sorted((y for y in year_counts.index if y != "unknown"), reverse=True)
    table1 = Table(title="Papers by Year", expand=True)
    table1.add_column("Year", style="cyan", no_wrap=True, min_width=7)
    table1.add_column("Count", style="green", justify="right", min_width=5)
    table1.add_column("%", style="yellow", justify="right")
    for year in known_years + (["unknown"] if "unknown" in year_counts.index else []):
        count = int(year_counts[year])
        table1.add_row(year, str(count), _pct(count))

    table2 = Table(title="Stats Breakdown", expand=True)
    table2.add_column("Stat", no_wrap=True)
    table2.add_column("Count", style="cyan", justify="right")
    table2.add_column("%", style="cyan", justify="right")
    table2.add_row("[white]Total:[/white] [bold white]papers[/bold white]", str(total), "100%")

    checks = [
        ("PDF", sum(entry.has_pdf for entry in entries)),
        ("TXT", sum(entry.has_txt for entry in entries)),
        ("Markdown", sum(entry.has_md for entry in entries)),
        ("Text (TXT or MD)", sum(entry.has_txt or entry.has_md for entry in entries)),
    ]
    for label, existing in checks:
        missing = total - existing
        table2.add_row(
            f"[white]{label}:[/white] [bold green]existing[/bold green]",
            f"[bold green]{existing}[/bold green]",
            f"[bold green]{_pct(existing)}[/bold green]",
        )
        table2.add_row(
            f"[white]{label}:[/white] [bold red]missing[/bold red]",
            f"[bold red]{missing}[/bold red]",
            f"[bold red]{_pct(missing)}[/bold red]",
        )

    with_other = sum(bool(entry.other_types) for entry in entries)
    if with_other:
        table2.add_row(
            "[white]Other files:[/white] [magenta]present[/magenta]",
            f"[magenta]{with_other}[/magenta]",
            f"[magenta]{_pct(with_other)}[/magenta]",
        )

    layout = Table.grid(padding=(0, 2))
    layout.add_column(ratio=1)
    layout.add_column(ratio=1)
    layout.add_row(table1, table2)

    return layout


def _build_file_types_table(entries: List[PaperEntry]) -> Table:
    """Build the per-type table: number of files and their total, average and largest size."""

    sizes_by_type: dict = {}
    for entry in entries:
        for ext, path in entry.files.items():
            sizes_by_type.setdefault(ext, []).append(path.stat().st_size)

    ordered_types = [ext for ext in PAPER_TYPES if ext in sizes_by_type]
    ordered_types += sorted(ext for ext in sizes_by_type if ext not in PAPER_TYPES)

    all_sizes = [size for sizes in sizes_by_type.values() for size in sizes]

    table = Table(title="Files by Type", show_footer=True, footer_style="bold")
    table.add_column("Type", no_wrap=True, footer="Total")
    table.add_column("Files", style="cyan", justify="right", footer=str(len(all_sizes)))
    table.add_column("Total size", style="green", justify="right", footer=format_bytes_console_ui(sum(all_sizes)))
    table.add_column(
        "Avg size",
        style="yellow",
        justify="right",
        footer=format_bytes_console_ui(sum(all_sizes) / len(all_sizes)),
    )
    table.add_column("Largest", style="magenta", justify="right", footer=format_bytes_console_ui(max(all_sizes)))

    for ext in ordered_types:
        sizes = sizes_by_type[ext]
        table.add_row(
            _type_badge(ext),
            str(len(sizes)),
            format_bytes_console_ui(sum(sizes)),
            format_bytes_console_ui(sum(sizes) / len(sizes)),
            format_bytes_console_ui(max(sizes)),
        )

    return table


@app.command()
def stats(
    pid: str = typer.Option(
        None,
        "--pid",
        help="Project ID. If not provided, uses PROJECT_ID from .env.",
    ),
) -> None:
    """Display statistics for the papers subdirectory of the project"""

    pid = get_proper_pid(pid)
    papers_path = _papers_path(pid)

    entries = scan_papers(papers_path)
    if not entries:
        cprint("No papers found.")
        return

    file_count = sum(len(entry.files) for entry in entries)
    total_size = sum(entry.total_size for entry in entries)

    cprint(f"[bold]Project:[/bold] [cyan]{pid}[/cyan]")
    cprint(
        f"[bold]Papers:[/bold] [yellow]{PAPERS_DIR}/[/yellow] "
        f"({len(entries)} papers, {file_count} files, {format_bytes_console_ui(total_size)})"
    )
    cprint()
    console.print(_build_papers_stats_grid(entries))
    cprint()
    console.print(_build_file_types_table(entries))
