"""Base methods and feature to be used in papers CLI command."""

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import requests
import urllib3

from llmexer.common import make_http_session
from llmexer.configs import logger, settings
from llmexer.exceptions import (
    PaperAddException,
    PaperAlreadyExistsException,
    PaperDownloadException,
    PaperExtractException,
)

DOCLING_TIMEOUT = 300
UNPAYWALL_EMAIL = "llmexer.unpaywall@local.local"

# File types a paper is made of: the PDF and its extracted text (pypdf / docling).
PAPER_TYPES = ("pdf", "txt", "md")

_YEAR_PREFIX = re.compile(r"^(\d{4})_")


class PaperSortBy(str, Enum):
    alpha = "alpha"
    date = "date"
    size = "size"


class PDFProcessor(str, Enum):
    pypdf = "pypdf"
    docling = "docling"


def remove_empty_image_placeholders(markdown: str) -> str:
    """
    Replaces useless image content from Docling markdown output with a comment:
    1. Standalone <!-- image --> / <!-- image:xxx --> placeholder blocks
    2. Inline base64-embedded images: ![Image](data:image/png;base64,...)
    """

    REPLACEMENT = "<!--  Image has been removed. Keeping placeholder only -->\n"

    text = markdown.replace("\r\n", "\n")

    text = re.sub(
        r"(?m)^[ \t]*<!--\s*image(?::[^\-]*)?\s*-->[ \t]*$\n?",
        REPLACEMENT,
        text,
    )

    text = re.sub(
        r"(?m)^[ \t]*!\[[^\]]*\]\(data:image/[^;]+;base64,[^)]*\)[ \t]*$\n?",
        REPLACEMENT,
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def extract_via_docling(pdf_path: Path, url: str, auth: tuple) -> str:
    """Upload a PDF to a docling-serve instance and return the extracted Markdown.


    Raises PaperExtractException on network errors or unexpected response shapes.
    """
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    endpoint = f"{url.rstrip('/')}/v1/convert/file"
    payload = {
        "options": json.dumps(
            {
                "from_formats": ["pdf"],
                "to_formats": ["md"],
                "do_ocr": True,
                "image_export_mode": "placeholder",
                "force_ocr": False,
                "ocr_preset": "easyocr",
                "ocr_lang": ["de", "en"],
                "do_picture_classification": True,
                "do_picture_description": True,
                "picture_description_preset": "default",
                "images_scale": 2.0,
                "pdf_backend": "dlparse_v2",
                "table_mode": "accurate",
                "do_table_structure": True,
                "abort_on_error": False,
            }
        )
    }

    try:
        with pdf_path.open("rb") as fh:
            files = {"files": (pdf_path.name, fh, "application/pdf")}
            session = make_http_session()
            resp = session.post(
                endpoint,
                auth=auth,
                files=files,
                data=payload,
                verify=False,
                timeout=DOCLING_TIMEOUT,
            )
    except requests.RequestException as exc:
        raise PaperExtractException(f"Docling request failed for '{pdf_path.name}': {exc}") from exc

    if resp.status_code != 200:
        raise PaperExtractException(
            f"Docling server returned HTTP {resp.status_code} for '{pdf_path.name}': {resp.text}"
        )

    try:
        data = resp.json()
    except ValueError as exc:
        raise PaperExtractException(f"Docling returned non-JSON response for '{pdf_path.name}'.") from exc

    docs = data.get("documents") or [data.get("document", {})]
    for doc in docs:
        md = doc.get("md_content") or doc.get("markdown") or doc.get("content")
        if md:
            no_images_md = remove_empty_image_placeholders(md)
            return no_images_md

    raise PaperExtractException(f"Could not find Markdown content in docling response for '{pdf_path.name}'.")


def download_pdf_from_url(
    url: str,
    papers_path: str,
    fallback_name: Optional[str] = None,
    forced_name: Optional[str] = None,
) -> str:
    """Download a PDF from `url` into `papers_path`. Returns the resolved filename.

    If `forced_name` is given it is always used as the filename (e.g. structured YEAR_TITLE_DOI.pdf).
    If `fallback_name` is given it is used only when the URL cannot provide a .pdf filename.
    Raises PaperAddException on network error or non-PDF filename (when no fallback/forced name).
    Raises PaperAlreadyExistsException if destination already exists.
    Respects settings.dry_run (skips write but still returns filename).
    """
    if forced_name is not None:
        dst = os.path.join(papers_path, forced_name)
        if os.path.exists(dst):
            raise PaperAlreadyExistsException(f"A paper named '{forced_name}' already exists in the papers directory.")

    try:
        session = make_http_session()
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PaperAddException(f"Failed to download '{url}': {exc}") from exc

    if forced_name:
        filename = forced_name
    else:
        # Determine filename: Content-Disposition > final URL path > original URL path > fallback
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
        if not filename or not filename.lower().endswith(".pdf"):
            if fallback_name:
                filename = fallback_name
            else:
                raise PaperAddException(
                    f"Could not resolve a PDF filename from URL '{url}' " f"(resolved name: '{filename}')."
                )

    dst = os.path.join(papers_path, filename)
    if os.path.exists(dst):
        raise PaperAlreadyExistsException(f"A paper named '{forced_name}' already exists in the papers directory.")

    if not settings.dry_run:
        try:
            with open(dst, "wb") as fh:
                for chunk in response.iter_content(chunk_size=8192):
                    fh.write(chunk)
        except requests.RequestException as exc:
            raise PaperAddException(f"Failed to download '{url}': {exc}") from exc
        logger.debug("Downloaded '%s' -> '%s'", url, dst)

    return filename


def resolve_unpaywall_pdf_url(doi: str, email: str) -> str:
    """Query Unpaywall for the best open-access PDF URL for the given DOI.

    Raises PaperDownloadException if the API request fails, the response is not valid
    JSON, there is no open-access location, or the location has no pdf URL.
    """
    unpaywall_url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    try:
        session = make_http_session()
        response = session.get(unpaywall_url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise PaperDownloadException(f"Unpaywall API request failed for DOI '{doi}': {exc}") from exc

    try:
        data = response.json()
    except ValueError as exc:
        raise PaperDownloadException(f"Unpaywall returned non-JSON response for DOI '{doi}'.") from exc

    best_oa = data.get("best_oa_location")
    if not best_oa:
        raise PaperDownloadException(f"No open-access location found for DOI '{doi}'.")

    pdf_url = best_oa.get("url_for_pdf")
    if not pdf_url:
        raise PaperDownloadException(f"Open-access location found for DOI '{doi}' but no PDF URL is available.")

    return pdf_url


def get_first_author_last_name(paper: dict) -> str | None:
    for author in paper.get("authors") or []:
        name = (author.get("name") or "").strip()
        if name:
            return name.split()[-1]
    return None


def make_structured_filename(
    year: Optional[str],
    author: Optional[str],
    title: Optional[str],
    doi: Optional[str],
) -> str:
    """Build a filename in the form YEAR_AUTHOR_TITLE_DOI.pdf, sanitizing each part."""

    def _clean(value: str) -> str:
        return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(value)).strip("_")

    year_part = _clean(year) if year and str(year).strip() else "NO_year"
    author_part = _clean(author) if author and str(author).strip() else "NO_author"
    title_part = _clean(title).lower()[:60].title() if title and str(title).strip() else "NO_title"
    doi_part = _clean(doi).lower() if doi and str(doi).strip() else "NO_doi"
    return f"{year_part}_{author_part}_{title_part}_{doi_part}.pdf"


@dataclass
class PaperEntry:
    """One paper in ``papers/``: every file that shares a stem (``X.pdf``, ``X.txt``, ``X.md``)."""

    stem: str
    files: dict[str, Path] = field(default_factory=dict)

    @property
    def has_pdf(self) -> bool:
        return "pdf" in self.files

    @property
    def has_txt(self) -> bool:
        return "txt" in self.files

    @property
    def has_md(self) -> bool:
        return "md" in self.files

    @property
    def other_types(self) -> list[str]:
        return sorted(ext for ext in self.files if ext not in PAPER_TYPES)

    @property
    def created_type(self) -> str:
        """Type of the file ``created`` is taken from: the PDF, or the oldest file when there is no PDF."""
        if self.has_pdf:
            return "pdf"
        return min(self.files, key=lambda ext: self.files[ext].stat().st_ctime)

    @property
    def created(self) -> float:
        """Creation time of the PDF, or of the oldest file when there is no PDF."""
        return self.files[self.created_type].stat().st_ctime

    @property
    def total_size(self) -> int:
        return sum(path.stat().st_size for path in self.files.values())

    @property
    def year(self) -> Optional[str]:
        """Year from a structured file name (``YEAR_AUTHOR_TITLE_DOI.pdf``), else ``None``."""
        match = _YEAR_PREFIX.match(self.stem)
        return match.group(1) if match else None

    @property
    def sort_size(self) -> int:
        """Size used by ``--sort-by size``: the PDF, or all files when there is no PDF."""
        pdf_size = self.size("pdf")
        return pdf_size if pdf_size is not None else self.total_size

    def size(self, ext: str) -> Optional[int]:
        path = self.files.get(ext)
        return path.stat().st_size if path is not None else None


def _file_type(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def is_paper_file(path: Path) -> bool:
    """True for the file types a paper is made of (``PAPER_TYPES``)."""
    return _file_type(path) in PAPER_TYPES


def list_paper_files(papers_path: str, sort_by: PaperSortBy = PaperSortBy.alpha, desc: bool = False) -> list[Path]:
    """Return the files directly in ``papers_path``, sorted. Empty when the folder is missing."""

    if not os.path.isdir(papers_path):
        return []

    files = [path for path in Path(papers_path).iterdir() if path.is_file()]

    if sort_by == PaperSortBy.date:
        files.sort(key=lambda path: path.stat().st_ctime, reverse=desc)
    elif sort_by == PaperSortBy.size:
        files.sort(key=lambda path: path.stat().st_size, reverse=desc)
    else:
        files.sort(key=lambda path: path.name.lower(), reverse=desc)

    return files


def scan_papers(
    papers_path: str,
    sort_by: PaperSortBy = PaperSortBy.alpha,
    desc: bool = False,
    only_paper_types: bool = False,
) -> list[PaperEntry]:
    """Group the files in ``papers_path`` by stem into papers, sorted. Empty when the folder is missing.

    With ``only_paper_types``, files that are not PDF, TXT or MD are left out.
    """

    entries: dict[str, PaperEntry] = {}
    for path in list_paper_files(papers_path):
        if only_paper_types and not is_paper_file(path):
            continue
        entry = entries.setdefault(path.stem, PaperEntry(stem=path.stem))
        entry.files[_file_type(path)] = path

    papers = list(entries.values())

    if sort_by == PaperSortBy.date:
        papers.sort(key=lambda entry: entry.created, reverse=desc)
    elif sort_by == PaperSortBy.size:
        papers.sort(key=lambda entry: entry.sort_size, reverse=desc)
    else:
        papers.sort(key=lambda entry: entry.stem.lower(), reverse=desc)

    return papers


def format_created_timestamp(timestamp: float) -> str:
    """Return a creation time as ``YYYY-MM-DD HH:MM:SS`` in UTC, like ``project list``."""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
