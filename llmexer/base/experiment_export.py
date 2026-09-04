"""Helpers to render a generated experiment database as a standalone HTML page.

The counterpart of :mod:`llmexer.base.search_export`, reading an
``experiment_*.db`` instead of a CSV. Both share their cell machinery and Jinja2
environment with :mod:`llmexer.base.html_export`, so the two pages look and behave
the same; what differs is the fixed set of columns exported here and the JSON
treatment of ``response_text``.
"""

import json
import os
import shlex
from datetime import datetime

from llmexer.base.dao import ExperimentDAO
from llmexer.base.html_export import (
    LONG_TEXT_PREVIEW,
    LONG_TEXT_THRESHOLD,
    SORT_KEY_MAX_LENGTH,
    _build_cell,
    _sanitize,
    render_template,
    sanitize_multiline,
)
from llmexer.logger import get_logger

logger = get_logger()

# Bundled Jinja2 template (see `llmexer/data/`), shipped via `[tool.setuptools.package-data]`.
TEMPLATE_NAME = "experiment_export.html.j2"

# Characters of `code` kept in the collapsed view. The full value stays in the page
# behind the expand toggle, so it is still readable and still what the copy button
# puts on the clipboard.
CODE_PREVIEW_LENGTH = 5

# Characters of the generated `try` command kept in the collapsed view. Every
# command opens with the same words, so the previews look alike -- the row is
# already identified by the other columns and this one exists to be copied.
TRY_PREVIEW_LENGTH = 32

# The exported table, in order: (column key, source key in the DAO row, cell kind).
# `_provider` rather than `provider_name` is the normalised, lower-cased table
# suffix, which is what `stats` reports too. A `None` source is a value this
# module derives rather than reads.
EXPORT_COLUMNS = [
    ("provider", "_provider", "text"),
    ("model", "model_name", "text"),
    ("profile", "profile_name", "text"),
    ("code", "code", "code"),
    ("response_text", "response_text", "json"),
    ("tokens", None, "number"),
    ("status", "status", "text"),
    ("seconds", "elapsed_seconds", "seconds"),
    ("timestamp", "timestamp", "timestamp"),
    ("try", None, "try"),
]

# Cell kinds that render right-aligned, like plain numbers do.
NUMERIC_KINDS = {"number", "seconds"}


def format_seconds(value) -> str:
    """Elapsed time as a one-decimal string; '' when the row has not run.

    Raw floats reach here as ``2.4700000000000002`` and are unreadable in a
    table. A value that is not a number is passed through as it stands - the
    database is user-editable.
    """

    if value is None or value == "":
        return ""
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return str(value)


def format_timestamp(value) -> str:
    """An ISO timestamp trimmed to whole seconds, keeping its UTC offset.

    ``run`` stores ``datetime.now(timezone.utc).isoformat()``, whose microseconds
    are noise in a report. The offset is kept: these are UTC, and dropping it
    would silently make them read as local time. Anything that does not parse is
    returned untouched.
    """

    if value is None or value == "":
        return ""
    try:
        return datetime.fromisoformat(str(value)).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return str(value)


def _split_code(code: str, model_name: str, profile_name: str) -> tuple:
    """Recover ``(data_id, prompt_id)`` from a generated ``code``.

    ``_combination_row`` builds it as
    ``f"{data_id}_{prompt_id}_{model_name}_{profile_name}"``, so the known suffix
    comes off first and what remains splits on its **first** underscore: a prompt
    file may well be called ``screening_v2``, while the data IDs the tool
    generates (``D01`` / ``P01`` / ``S01``) carry none.

    Returns ``("", "")`` when the code does not have that shape, so the caller
    can leave the cell empty instead of emitting a command for the wrong
    combination.
    """

    suffix = f"_{model_name}_{profile_name}"
    if not code or not model_name or not profile_name or not code.endswith(suffix):
        return "", ""

    head = code[: -len(suffix)]
    data_id, separator, prompt_id = head.partition("_")
    if not separator or not data_id or not prompt_id:
        return "", ""

    return data_id, prompt_id


def build_try_command(row: dict, project_id: str, db_name: str) -> str:
    """The ``experiment try`` invocation that re-runs this row, or '' if unknown.

    Every argument is spelled out so the command runs whatever ``.env`` holds and
    records the try in the same database. ``--model`` / ``--provider`` are always
    passed because one profile name can cover several models.
    """

    model_name = str(row.get("model_name") or "")
    profile_name = str(row.get("profile_name") or "")
    provider = str(row.get("_provider") or row.get("provider_name") or "")

    data_id, prompt_id = _split_code(str(row.get("code") or ""), model_name, profile_name)
    if not data_id:
        return ""

    parts = [
        "llmexer",
        "experiment",
        "try",
        "--pid",
        project_id,
        "--file",
        db_name,
        "--data-id",
        data_id,
        "--prompt",
        prompt_id,
        "--profile",
        profile_name,
        "--model",
        model_name,
        "--provider",
        provider,
    ]

    # shlex.quote only adds quotes where a value actually needs them, so common
    # names such as 'gpt-oss:120b' stay readable.
    return " ".join(shlex.quote(part) for part in parts)


def row_tokens(row: dict) -> int:
    """Token count for a row: ``total_tokens``, else ``usage_tokens``, else 0.

    The same fallback ``ExperimentDAO.stats()`` applies, so the export and the
    stats table can never report different numbers for the same run.
    """

    for key in ("total_tokens", "usage_tokens"):
        value = row.get(key)
        if value:
            return int(value)
    return 0


def _strip_code_fence(text: str) -> str:
    """Drop a Markdown code fence wrapping a response, if there is one.

    Models routinely answer with their JSON inside ```` ```json ... ``` ````. Left
    in place that prefix makes every such answer unparseable, so the fence is
    peeled off before parsing - the text itself is never modified otherwise.
    """

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # First line is the fence, optionally carrying a language tag ("```json").
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def format_response_text(text: str) -> tuple:
    """Pretty-print a response as JSON, falling back to the text as it stands.

    Returns ``(rendered, is_json)``. A response that does not parse is not an
    error - plenty of prompts ask for prose - so it is returned unchanged with
    ``is_json`` False and rendered as plain text.
    """

    if not text or not text.strip():
        return "", False

    candidate = _strip_code_fence(text)
    if not candidate:
        return text, False

    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, TypeError, ValueError):
        return text, False

    # A bare string or number is valid JSON but reads better as what it already is.
    if not isinstance(parsed, (dict, list)):
        return text, False

    return json.dumps(parsed, indent=2, ensure_ascii=False), True


def _build_response_cell(text: str) -> dict:
    """Build the ``response_text`` cell, pretty-printing JSON where possible."""

    rendered, is_json = format_response_text(sanitize_multiline(text))

    cell = {
        "text": rendered,
        "short": rendered,
        "truncated": False,
        "bool": None,
        "href": "",
        "filter": rendered.lower(),
        "sort": rendered[:SORT_KEY_MAX_LENGTH].lower(),
        "is_json": is_json,
    }

    if rendered == "":
        cell["sort"] = "￿"
        return cell

    if len(rendered) > LONG_TEXT_THRESHOLD:
        cell["short"] = rendered[:LONG_TEXT_PREVIEW].rstrip()
        cell["truncated"] = True

    return cell


def _build_collapsed_cell(text: str, preview_length: int) -> dict:
    """Build a cell whose display is cut to ``preview_length`` characters.

    Used for the two columns that carry a value too wide to sit in a table -
    the generated ``code`` and the ``try`` command. Only the preview is shown;
    the expand toggle and the copy button both reach the full value, and the
    sort/filter keys stay the whole string so filtering still matches on it.
    """

    cell = _build_cell(text, "text")
    if len(text) > preview_length:
        cell["short"] = text[:preview_length]
        cell["truncated"] = True
    return cell


def _cell_value(row: dict, key, kind: str) -> str:
    """The raw string a cell renders, before it is turned into a cell dict."""

    if kind == "number" and key is None:
        return str(row_tokens(row))

    value = row.get(key)
    if value is None:
        return ""

    if kind == "seconds":
        return format_seconds(value)
    if kind == "timestamp":
        return format_timestamp(value)

    return _sanitize(str(value))


def build_export_context(rows: list, title: str, project_id: str, source_db: str, generated_at: str) -> dict:
    """Build the full rendering context for the experiment export template."""

    columns = [
        {
            "key": key,
            "label": key,
            "kind": kind,
            # Not "copy": Jinja resolves `column.copy` to the dict's built-in
            # copy() method, which is always truthy.
            "copyable": True,
        }
        for key, _source, kind in EXPORT_COLUMNS
    ]

    table_rows = []
    for row in rows:
        cells = []
        for _key, source, kind in EXPORT_COLUMNS:
            if kind == "json":
                cells.append(_build_response_cell(str(row.get(source) or "")))
            elif kind == "code":
                cells.append(_build_collapsed_cell(_cell_value(row, source, kind), CODE_PREVIEW_LENGTH))
            elif kind == "try":
                command = build_try_command(row, project_id, source_db)
                cells.append(_build_collapsed_cell(command, TRY_PREVIEW_LENGTH))
            else:
                cells.append(_build_cell(_cell_value(row, source, kind), kind))
        table_rows.append(cells)

    return {
        "title": title,
        "project_id": project_id,
        "source_db": source_db,
        "generated_at": generated_at,
        "columns": columns,
        "rows": table_rows,
        "row_count": len(table_rows),
    }


def render_experiment_export_html(context: dict) -> str:
    """Render the bundled Jinja2 template with the given context."""

    return render_template(TEMPLATE_NAME, context)


def export_db_to_html(db_path: str, html_path: str, project_id: str, generated_at: str) -> int:
    """Render an experiment database to an HTML file. Returns the row count.

    Every generated row is exported, run or not: an unrun one simply carries an
    empty response and no status. Dry-run handling belongs to the caller; this
    always writes.
    """

    with ExperimentDAO(db_path) as dao:
        rows = dao.fetch_rows()

    context = build_export_context(
        rows,
        title=os.path.splitext(os.path.basename(db_path))[0],
        project_id=project_id,
        source_db=os.path.basename(db_path),
        generated_at=generated_at,
    )

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(render_experiment_export_html(context))

    logger.debug("Exported '%s' to '%s'", db_path, html_path)

    return context["row_count"]
