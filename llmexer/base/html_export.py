"""Building blocks shared by the HTML exporters.

``search export`` renders a CSV and ``experiment export`` renders a SQLite
database, but both produce the same kind of page: one sortable, filterable table
whose cells carry precomputed sort/filter keys and an optional collapsed preview.
Everything that is not specific to either data source lives here - cell building,
value sanitization, column classification and the Jinja2 environment.
"""

import re
from html import unescape
from urllib.parse import quote

import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape

from llmexer.constants import PACKAGE_DATA_PATH

# Cells longer than this get a collapsed preview plus a `more`/`less` toggle.
LONG_TEXT_THRESHOLD = 160
# Number of characters kept in the collapsed preview.
LONG_TEXT_PREVIEW = 140
# Number of characters kept in a text column's sort key.
SORT_KEY_MAX_LENGTH = 120

_TRUE_VALUES = {"true", "1", "yes"}
_FALSE_VALUES = {"false", "0", "no"}

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Collapses horizontal whitespace only, so line structure survives.
_INLINE_WHITESPACE_RE = re.compile(r"[^\S\n]+")

# Sort key prefix forcing empty cells to the end of an ascending sort.
_EMPTY_SORT_KEY = "￿"


def _strip_markup(text: str) -> str:
    """Unescape entities and drop HTML tags, leaving a space behind each."""

    return _TAG_RE.sub(" ", unescape(text))


def _sanitize(text: str) -> str:
    """Strip markup, control characters and whitespace runs out of a cell value.

    Publishers embed HTML in abstracts and the CSVs carry raw newlines; both would show up
    verbatim in the table. Rendering stays autoescaped on top of this, so escaping is not
    delegated here - this only decides what text reaches the page.
    """

    text = _strip_markup(text)
    text = "".join(c if c >= " " and c != "\x7f" else " " for c in text)

    return _WHITESPACE_RE.sub(" ", text).strip()


def sanitize_multiline(text: str) -> str:
    """Sanitize a cell value while keeping its newlines and indentation.

    :func:`_sanitize` collapses every whitespace run into a single space, which is
    right for a one-line table cell but destroys the layout of anything meant to be
    read as preformatted text - a pretty-printed JSON response above all. This keeps
    newlines (and the leading spaces that follow them) and only collapses horizontal
    runs, so indentation survives without a stray tab widening the column.
    """

    text = _strip_markup(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(c if c >= " " or c == "\n" else " " for c in text)
    text = text.replace("\x7f", " ")

    return _INLINE_WHITESPACE_RE.sub(" ", text).strip()


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


def render_template(template_name: str, context: dict) -> str:
    """Render one of the bundled Jinja2 templates with the given context.

    ``autoescape`` is switched on explicitly rather than left to the extension:
    ``.j2`` is not among the suffixes Jinja enables by default, so without
    ``default=True`` every interpolation on these pages would be unescaped.
    """

    env = Environment(
        loader=FileSystemLoader(str(PACKAGE_DATA_PATH)),
        autoescape=select_autoescape(default=True, default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )

    return env.get_template(template_name).render(**context)
