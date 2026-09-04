"""Helpers to render search result CSVs as standalone HTML pages."""

import os
import re
from html import unescape
from urllib.parse import quote

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from llmexer.constants import PACKAGE_DATA_PATH
from llmexer.logger import get_logger

logger = get_logger()

# Bundled Jinja2 template (see `llmexer/data/`), shipped via `[tool.setuptools.package-data]`.
TEMPLATE_NAME = "search_export.html.j2"

# Cells longer than this get a collapsed preview plus a `more`/`less` toggle.
LONG_TEXT_THRESHOLD = 160
# Number of characters kept in the collapsed preview.
LONG_TEXT_PREVIEW = 140
# Number of characters kept in a text column's sort key.
SORT_KEY_MAX_LENGTH = 120

_TRUE_VALUES = {"true", "1", "yes"}
_FALSE_VALUES = {"false", "0", "no"}

# Columns rendered as external links: lowercased column name -> URL template.
LINK_COLUMNS = {"doi": "https://doi.org/{value}"}

# Columns that get a copy-to-clipboard button.
COPY_COLUMNS = {"title", "abstract", "authors", "doi", "pdf_filename"}

# Columns dropped from the rendered table. `sem_scholar_paper_id` is the pre-OpenAlex name
# of the same internal id and is still present in older search CSVs.
HIDDEN_COLUMNS = {"search_engine_internal_id", "sem_scholar_paper_id"}

# Shorter header labels: lowercased column name -> label shown in the table.
COLUMN_LABELS = {
    "language": "lang",
    "citationcount": "citations",
    "referencecount": "references",
    "isopenaccess": "OpenAccess",
}

# Columns pulled out of their CSV position: lowercased column name -> the lowercased
# column it should directly follow. Applied in order; anything unlisted keeps CSV order.
COLUMN_AFTER = {
    "doi": "abstract",
    "pdf_downloaded": "isopenaccess",
}

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")

# Sort key prefix forcing empty cells to the end of an ascending sort.
_EMPTY_SORT_KEY = "￿"


def _sanitize(text: str) -> str:
    """Strip markup, control characters and whitespace runs out of a cell value.

    Publishers embed HTML in abstracts and the CSVs carry raw newlines; both would show up
    verbatim in the table. Rendering stays autoescaped on top of this, so escaping is not
    delegated here - this only decides what text reaches the page.
    """

    text = unescape(text)
    text = _TAG_RE.sub(" ", text)
    text = "".join(c if c >= " " and c != "\x7f" else " " for c in text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def _normalize(value) -> str:
    """Return a cell value as a sanitized string ('' for NaN/None)."""

    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return _sanitize(str(value))


def _as_bool(text: str):
    """Return True/False for a boolean-ish string, otherwise None."""

    lowered = text.lower()
    if lowered in _TRUE_VALUES:
        return True
    if lowered in _FALSE_VALUES:
        return False
    return None


def _is_number(text: str) -> bool:
    """Return whether the string parses as a number."""

    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _classify_column(values) -> str:
    """Classify a column as 'bool', 'number', 'long' or 'text' from its values.

    Empty cells are ignored; a column with no values at all is plain 'text'.
    """

    non_empty = [v for v in values if v != ""]
    if not non_empty:
        return "text"

    if all(_as_bool(v) is not None for v in non_empty):
        return "bool"

    if all(_is_number(v) for v in non_empty):
        return "number"

    if any(len(v) > LONG_TEXT_THRESHOLD for v in non_empty):
        return "long"

    return "text"


def _build_link(text: str, link_template: str) -> str:
    """Return an ``http(s)`` URL for a cell value, or '' when it cannot be linked.

    Values that already carry an ``http(s)`` scheme are used verbatim; anything else is
    percent-encoded into the template. No other scheme can ever reach the rendered href.
    """

    if text.lower().startswith(("http://", "https://")):
        return text

    return link_template.format(value=quote(text, safe="/:"))


def _build_cell(text: str, kind: str, link_template: str = None) -> dict:
    """Build the template context for a single table cell.

    The `sort` and `filter` keys are precomputed here so the page's JavaScript never has
    to re-derive them while sorting or filtering.
    """

    cell = {
        "text": text,
        "short": text,
        "truncated": False,
        "bool": None,
        "href": "",
        "filter": text.lower(),
    }

    if text == "":
        cell["sort"] = _EMPTY_SORT_KEY
        return cell

    if link_template:
        cell["href"] = _build_link(text, link_template)

    if kind == "bool":
        cell["bool"] = _as_bool(text)
        cell["sort"] = "1" if cell["bool"] else "0"
        # Normalize the displayed label so mixed 'True'/'yes'/'1' inputs read uniformly.
        cell["filter"] = "true" if cell["bool"] else "false"
        return cell

    if kind == "number":
        cell["sort"] = text
        return cell

    if kind == "long" and len(text) > LONG_TEXT_THRESHOLD:
        cell["short"] = text[:LONG_TEXT_PREVIEW].rstrip()
        cell["truncated"] = True

    # Long cells only need a sort *prefix* — carrying whole abstracts in `data-sort`
    # would roughly double the page size for no benefit.
    cell["sort"] = text[:SORT_KEY_MAX_LENGTH].lower()
    return cell


def _order_columns(names: list) -> list:
    """Return the CSV column names reordered per `COLUMN_AFTER`.

    A column whose anchor is missing from the CSV keeps its original position.
    """

    ordered = list(names)
    lowered = {str(n).lower(): n for n in ordered}

    for name, anchor in COLUMN_AFTER.items():
        if name not in lowered or anchor not in lowered:
            continue
        ordered.remove(lowered[name])
        ordered.insert(ordered.index(lowered[anchor]) + 1, lowered[name])

    return ordered


def build_export_context(df: pd.DataFrame, title: str, project_id: str, source_csv: str, generated_at: str) -> dict:
    """Build the full rendering context for the HTML export template."""

    columns = []
    cell_columns = []
    for name in _order_columns(list(df.columns)):
        lowered = str(name).lower()
        if lowered in HIDDEN_COLUMNS:
            continue

        values = [_normalize(v) for v in df[name].tolist()]
        kind = _classify_column(values)
        link_template = LINK_COLUMNS.get(lowered)
        columns.append(
            {
                "key": str(name),
                # Only the displayed label is shortened - `key` stays the CSV column name.
                "label": COLUMN_LABELS.get(lowered, str(name)),
                "kind": kind,
                # Not "copy": Jinja resolves `column.copy` to the dict's built-in
                # copy() method, which is always truthy.
                "copyable": lowered in COPY_COLUMNS,
            }
        )
        cell_columns.append([_build_cell(v, kind, link_template) for v in values])

    rows = [list(row) for row in zip(*cell_columns)] if cell_columns else []

    return {
        "title": title,
        "project_id": project_id,
        "source_csv": source_csv,
        "generated_at": generated_at,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
    }


def render_search_export_html(context: dict) -> str:
    """Render the bundled Jinja2 template with the given context."""

    env = Environment(
        loader=FileSystemLoader(str(PACKAGE_DATA_PATH)),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    return env.get_template(TEMPLATE_NAME).render(**context)


def export_csv_to_html(csv_path: str, html_path: str, project_id: str, generated_at: str) -> int:
    """Render a search result CSV to an HTML file. Returns the number of exported rows.

    Dry-run handling belongs to the caller; this always writes.
    """

    df = pd.read_csv(csv_path, sep=";", dtype=str, keep_default_na=False)

    context = build_export_context(
        df,
        title=os.path.splitext(os.path.basename(csv_path))[0],
        project_id=project_id,
        source_csv=os.path.basename(csv_path),
        generated_at=generated_at,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_search_export_html(context))

    logger.debug("Exported '%s' to '%s'", csv_path, html_path)

    return context["row_count"]
