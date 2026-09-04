"""Helpers to render search result CSVs as standalone HTML pages.

The cell-level machinery is shared with ``experiment export`` and lives in
:mod:`llmexer.base.html_export`; what stays here is the presentation config that
only makes sense for a search result CSV.
"""

import os

import pandas as pd

# Re-exported: these were defined here before the exporters started sharing them,
# and `_sanitize` / `_classify_column` / `LONG_TEXT_THRESHOLD` remain part of this
# module's surface for the tests that import them from it.
from llmexer.base.html_export import (  # noqa: F401  # pylint: disable=unused-import
    LONG_TEXT_PREVIEW,
    LONG_TEXT_THRESHOLD,
    SORT_KEY_MAX_LENGTH,
    _as_bool,
    _build_cell,
    _build_link,
    _classify_column,
    _is_number,
    _normalize,
    _sanitize,
    render_template,
)
from llmexer.logger import get_logger

logger = get_logger()

# Bundled Jinja2 template (see `llmexer/data/`), shipped via `[tool.setuptools.package-data]`.
TEMPLATE_NAME = "search_export.html.j2"

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

    return render_template(TEMPLATE_NAME, context)


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
