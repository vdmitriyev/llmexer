"""Count a project's records for a PRISMA 2020 flow diagram, and draw it as SVG.

Copied into a project's ``analysis/`` folder by ``llmexer analysis add-prisma`` and
imported there as a top-level module, so this file imports nothing from ``llmexer``
and nothing from its sibling analysis modules (pandas, PyYAML and the standard
library only). ``llmexer analysis draw-prisma`` runs the very same code.

Two consequences of that rule, both intentional:

* :func:`strip_code_fence` is a verbatim copy of ``llmexer.common.strip_code_fence``,
  and :func:`dedup_key` follows ``llmexer.base.search._dedup_key``.
  ``tests/test_analysis_prisma.py`` asserts they stay in step.
* Every count comes with a short note saying where it came from, so the diagram
  can be checked by hand. A count that cannot be derived is ``None`` and is drawn
  as ``n = ?``; it can be filled in through the ``overrides`` of the counts file.
"""

import json
import os
import re
import shutil
import sqlite3
import textwrap
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Project layout (duplicated from llmexer on purpose, see the module docstring)
# ---------------------------------------------------------------------------

SEARCHES_DIR = "searches"
SEARCH_LOGS_DIR = "logs"
RESULTS_SUFFIX = "__results.csv"
FILTERED_SUFFIX = "__filtered.csv"
COUNTS_SUFFIX = "__counts.json"
MERGED_MARKER = "__merged"
# Replaced outputs are copied here first (same folder as the notebook backups).
BACKUP_DIR = ".backup"
CSV_SEPARATOR = ";"

SOURCE_SEMANTIC_SCHOLAR = "Semantic Scholar"
SOURCE_OPENALEX = "OpenAlex"

EXPERIMENT_TABLE_PREFIX = "experiment_"
_PROVIDER_RE = re.compile(r"^[a-z0-9_]+$")

# How the answers of several models (and prompts) for one record become one decision.
RULES = ("majority", "any", "all")
KEPT, EXCLUDED, UNDECIDED = "kept", "excluded", "undecided"

YES_VALUES = {"yes", "y", "true", "1", "include", "included"}
NO_VALUES = {"no", "n", "false", "0", "exclude", "excluded"}

# ---------------------------------------------------------------------------
# The boxes of the diagram, in reading order
# ---------------------------------------------------------------------------

BOXES = {
    "identified_databases": "Records identified per database",
    "duplicates": "Duplicate records removed",
    "automation": "Records marked as ineligible by automation tools",
    "other_removed": "Records removed for other reasons",
    "screened": "Records screened",
    "screen_excluded": "Records excluded",
    "sought": "Reports sought for retrieval",
    "not_retrieved": "Reports not retrieved",
    "assessed": "Reports assessed for eligibility",
    "excluded_reasons": "Reports excluded, by reason",
    "included": "Studies included in review",
}

COUNTS_FILE_HEADER = """\
# PRISMA 2020 counts, written by `llmexer analysis draw-prisma`.
#
# `settings` are the options of the last run; `draw-prisma` re-uses each one you
# do not pass again (`--ignore-settings` starts from scratch). `inputs` names the
# files read. `derived` is recomputed from the project on every run, with the
# source of each number. Do not edit it - edit `overrides` instead: a value there
# replaces the derived one in the diagram and is kept as it is on every later run.
# `identified_databases` and `excluded_reasons` take a mapping of name -> count.
"""


def strip_code_fence(text: str) -> str:
    """Drop a Markdown code fence wrapping a value, if there is one."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    # First line is the fence, optionally carrying a language tag ("```json").
    lines = lines[1:]
    if lines and lines[-1].strip().startswith("```"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def _is_missing(value) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == "" or str(value).strip().lower() == "nan"


def _normalize_text(value) -> str:
    if _is_missing(value):
        return ""
    return " ".join(str(value).split()).strip().lower()


def dedup_key(row) -> str | None:
    """DOI-based key when present, else a normalized-title key; the rule `search merge` uses."""

    doi = _normalize_text(row.get("doi"))
    if doi:
        return f"doi:{doi}"
    title = _normalize_text(row.get("title"))
    if title:
        return f"title:{title}"
    return None


def _read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, sep=CSV_SEPARATOR, encoding="utf-8")


def _keys(df: pd.DataFrame) -> set:
    keys = set()
    for _, row in df.iterrows():
        key = dedup_key(row)
        if key is not None:
            keys.add(key)
    return keys


# ---------------------------------------------------------------------------
# Identification
# ---------------------------------------------------------------------------


def list_searches(project_dir) -> list[str]:
    """Ids of every search that has a ``__results.csv``, sorted."""

    searches_dir = Path(project_dir) / SEARCHES_DIR
    if not searches_dir.is_dir():
        return []
    ids = []
    for path in sorted(searches_dir.glob(f"*{RESULTS_SUFFIX}")):
        if MERGED_MARKER in path.name:
            continue
        ids.append(path.name[: -len(RESULTS_SUFFIX)])
    return ids


def identified_by_database(project_dir, searches) -> tuple[dict, str]:
    """Records found per database, summed over ``searches``, before any duplicate was removed.

    Exact when ``search run`` wrote ``logs/<id>__counts.json`` (llmexer 0.6.1 and later).
    Older searches only keep the OpenAlex records Semantic Scholar did not find,
    so their OpenAlex count (and the duplicates between the engines) is too low.
    """

    searches_dir = Path(project_dir) / SEARCHES_DIR
    totals: Counter = Counter()
    approximate = []
    for search_id in searches:
        counts_path = searches_dir / SEARCH_LOGS_DIR / f"{search_id}{COUNTS_SUFFIX}"
        if counts_path.is_file():
            counts = json.loads(counts_path.read_text(encoding="utf-8"))
            totals[SOURCE_SEMANTIC_SCHOLAR] += int(counts.get("semantic_scholar", 0))
            totals[SOURCE_OPENALEX] += int(counts.get("openalex", 0))
            continue

        approximate.append(search_id)
        results = _read_csv(searches_dir / f"{search_id}{RESULTS_SUFFIX}")
        sources = results["entry_source"] if "entry_source" in results.columns else pd.Series(dtype=str)
        for source, count in sources.fillna("unknown").value_counts().items():
            totals[str(source)] += int(count)

    by_database = {name: count for name, count in totals.items() if count}
    note = f"search counts of {len(searches)} search(es)"
    if approximate:
        note += (
            f"; approximate for {', '.join(approximate)} (run before llmexer 0.6.1: "
            "OpenAlex records also found by Semantic Scholar are not counted)"
        )
    return by_database, note


def _within_search_duplicates(project_dir, searches) -> int:
    searches_dir = Path(project_dir) / SEARCHES_DIR
    total = 0
    for search_id in searches:
        counts_path = searches_dir / SEARCH_LOGS_DIR / f"{search_id}{COUNTS_SUFFIX}"
        if counts_path.is_file():
            counts = json.loads(counts_path.read_text(encoding="utf-8"))
            total += int(counts.get("openalex_duplicates", 0))
    return total


def _record_keys(project_dir, searches, suffix: str, fallback_suffix: str | None = None) -> set:
    searches_dir = Path(project_dir) / SEARCHES_DIR
    keys = set()
    for search_id in searches:
        path = searches_dir / f"{search_id}{suffix}"
        if not path.is_file() and fallback_suffix:
            path = searches_dir / f"{search_id}{fallback_suffix}"
        if path.is_file():
            keys |= _keys(_read_csv(path))
    return keys


def duplicates_removed(project_dir, searches) -> tuple[int, str]:
    """Duplicates between the engines of one search, plus those between searches."""

    searches_dir = Path(project_dir) / SEARCHES_DIR
    within = _within_search_duplicates(project_dir, searches)
    rows = 0
    for search_id in searches:
        rows += len(_read_csv(searches_dir / f"{search_id}{RESULTS_SUFFIX}"))
    unique = len(_record_keys(project_dir, searches, RESULTS_SUFFIX))
    across = rows - unique
    note = (
        f"{within} between Semantic Scholar and OpenAlex (counts files) + {across} between searches "
        "or without DOI/title (same DOI, else same title)"
    )
    return within + across, note


def removed_by_automation(project_dir, searches) -> tuple[int, str]:
    """Unique records dropped by ``search filter`` (language, source, DOI, downloaded)."""

    before = _record_keys(project_dir, searches, RESULTS_SUFFIX)
    after = _record_keys(project_dir, searches, FILTERED_SUFFIX, fallback_suffix=RESULTS_SUFFIX)
    removed = len(before - after)
    filters = _filters_applied(project_dir, searches)
    note = "unique records in __results.csv but not in __filtered.csv"
    note += f" (filters: {', '.join(filters)})" if filters else " (no filters applied)"
    return removed, note


def _filters_applied(project_dir, searches) -> list[str]:
    log_path = Path(project_dir) / SEARCHES_DIR / SEARCH_LOGS_DIR / "filters-applied.log"
    if not log_path.is_file():
        return []
    wanted = {f"{search_id}{FILTERED_SUFFIX}" for search_id in searches}
    found = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        match = re.search(r"search file: (\S+); filter applied: (.*?) ; input rows", line)
        if match and match.group(1) in wanted and match.group(2) not in found:
            found.append(match.group(2))
    return found


def records_after_identification(project_dir, searches) -> int:
    """Unique records left after duplicates and automation tools: the default 'screened'."""

    return len(_record_keys(project_dir, searches, FILTERED_SUFFIX, fallback_suffix=RESULTS_SUFFIX))


# ---------------------------------------------------------------------------
# Screening and eligibility: decisions from LLM answers
# ---------------------------------------------------------------------------


def parse_answer(value):
    """The model's JSON answer as a dict, or ``None`` when it is not a JSON object."""

    if _is_missing(value):
        return None
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(strip_code_fence(str(value)))
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def answer_field(answer: dict | None, field: str):
    """Value of ``field`` in an answer; a dotted name reaches into nested objects."""

    current = answer
    for part in field.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def to_vote(value) -> bool | None:
    """``True`` for yes-like values, ``False`` for no-like values, ``None`` otherwise."""

    if isinstance(value, bool):
        return value
    text = _normalize_text(value)
    if text in YES_VALUES:
        return True
    if text in NO_VALUES:
        return False
    return None


def load_answers(db_path) -> pd.DataFrame:
    """Every generated row of an experiment database: data_id, model and the answer text."""

    columns = ["data_id", "prompt_id", "provider_name", "model_name", "profile_name", "response_text"]
    if db_path is None or not Path(db_path).is_file():
        return pd.DataFrame(columns=columns)

    frames = []
    with sqlite3.connect(db_path) as connection:
        names = [row[0] for row in connection.execute("select name from sqlite_master where type = 'table'")]
        for name in sorted(names):
            if not name.startswith(EXPERIMENT_TABLE_PREFIX):
                continue
            if not _PROVIDER_RE.match(name[len(EXPERIMENT_TABLE_PREFIX) :]):
                continue
            # Table names are whitelisted against _PROVIDER_RE just above.
            frame = pd.read_sql(f"select * from {name}", connection)  # nosec B608
            frames.append(frame[[column for column in columns if column in frame.columns]])
    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def decide(votes: list, rule: str) -> str:
    """One decision from the yes/no votes for a record. ``None`` votes are ignored.

    ``majority``: more yes than no keeps it, more no excludes it, a tie keeps it.
    ``any``: one yes keeps it. ``all``: one no excludes it. No usable vote: undecided.
    """

    if rule not in RULES:
        raise ValueError(f"rule must be one of {', '.join(RULES)}, got {rule!r}")

    yes = sum(1 for vote in votes if vote is True)
    no = sum(1 for vote in votes if vote is False)
    if yes + no == 0:
        return UNDECIDED
    if rule == "any":
        return KEPT if yes else EXCLUDED
    if rule == "all":
        return EXCLUDED if no else KEPT
    return KEPT if yes >= no else EXCLUDED


def screening_decisions(answers: pd.DataFrame, data_ids, field: str, rule: str) -> pd.DataFrame:
    """One row per data_id: yes / no / unusable vote counts and the decision under ``rule``.

    Raises:
        ValueError: when no answer holds ``field`` at all (most likely a typo).
    """

    parsed = [parse_answer(value) for value in answers.get("response_text", pd.Series(dtype=object))]
    values = [answer_field(answer, field) for answer in parsed]
    if answers.shape[0] and all(value is None for value in values):
        available = sorted({key for answer in parsed if answer for key in answer})
        raise ValueError(
            f"No answer has the field '{field}'. Fields found: {', '.join(available) or 'none (no JSON answers)'}."
        )

    votes_by_id: dict = {}
    for data_id, value in zip(answers.get("data_id", []), values):
        votes_by_id.setdefault(str(data_id), []).append(to_vote(value))

    rows = []
    for data_id in data_ids:
        votes = votes_by_id.get(str(data_id), [])
        rows.append(
            {
                "data_id": str(data_id),
                "yes": sum(1 for vote in votes if vote is True),
                "no": sum(1 for vote in votes if vote is False),
                "unusable": sum(1 for vote in votes if vote is None),
                "decision": decide(votes, rule),
            }
        )
    return pd.DataFrame(rows, columns=["data_id", "yes", "no", "unusable", "decision"])


# ---------------------------------------------------------------------------
# Which answers count, and the decision file (one row per record and stage)
# ---------------------------------------------------------------------------

# Answer filters: option name -> column of the experiment tables.
FILTER_COLUMNS = {"prompt": "prompt_id", "provider": "provider_name", "model": "model_name", "profile": "profile_name"}

STAGE_SCREENING, STAGE_FULLTEXT = "screening", "fulltext"
REASON_NOT_STATED = "Reason not stated"
MANUAL_COLUMNS = ["manual", "manual_reason"]
DECISION_COLUMNS = [
    "stage",
    "data_id",
    "title",
    "yes",
    "no",
    "unusable",
    "llm_decision",
    "llm_reason",
    "manual",
    "manual_reason",
    "decision",
]


def _filter_value(key: str, value) -> str:
    text = str(value).strip()
    if key == "prompt" and text.endswith(".txt"):
        text = text[: -len(".txt")]
    if key == "provider":
        text = text.lower()
    return text


def filter_answers(answers: pd.DataFrame, filters: dict | None) -> pd.DataFrame:
    """Only the answers that match every filter.

    ``filters`` maps ``prompt`` / ``provider`` / ``model`` / ``profile`` to the
    values to keep; an empty list keeps everything. A value no answer has stops
    with the values found, since a typo would otherwise silently drop all votes.
    """

    for key, wanted_values in (filters or {}).items():
        if not wanted_values:
            continue
        if key not in FILTER_COLUMNS:
            raise ValueError(f"Unknown answer filter '{key}'. Known: {', '.join(FILTER_COLUMNS)}.")
        wanted = {_filter_value(key, value) for value in wanted_values}
        found = answers[FILTER_COLUMNS[key]].map(lambda value, key=key: _filter_value(key, value))
        unknown = sorted(wanted - set(found))
        if answers.shape[0] and unknown:
            raise ValueError(
                f"No answer has {key} {', '.join(repr(value) for value in unknown)}. "
                f"Found: {', '.join(sorted(set(found))) or 'none'}."
            )
        answers = answers[found.isin(wanted)]
    return answers


def describe_filters(filters: dict | None) -> str:
    """``"prompt 'p1', model 'a' or 'b'"``, or ``""`` when nothing is filtered."""

    parts = []
    for key, values in (filters or {}).items():
        if values:
            parts.append(f"{key} " + " or ".join(f"'{value}'" for value in values))
    return ", ".join(parts)


def _data_table(data_csv) -> tuple[list[str], dict]:
    """``(ids, title_by_id)`` of an experiment's ``data.csv``."""

    data = _read_csv(data_csv)
    if "ID" not in data.columns:
        raise ValueError(f"'{data_csv}' has no 'ID' column.")
    title_column = next((name for name in ("Title", "title") if name in data.columns), None)
    ids, titles = [], {}
    for _, row in data.iterrows():
        if _is_missing(row["ID"]):
            continue
        data_id = str(row["ID"]).strip()
        if data_id not in titles:
            ids.append(data_id)
            titles[data_id] = "" if title_column is None or _is_missing(row[title_column]) else str(row[title_column])
    return ids, titles


def _data_ids(data_csv) -> list[str]:
    return _data_table(data_csv)[0]


def _llm_reasons(answers: pd.DataFrame, data_ids, reason_field: str | None) -> dict:
    """The first stated reason per record (a yes/no value is not a reason)."""

    reasons = {}
    if not reason_field:
        return reasons
    wanted = {str(data_id) for data_id in data_ids}
    for data_id, value in zip(answers["data_id"].astype(str), answers["response_text"]):
        if data_id not in wanted or data_id in reasons:
            continue
        candidate = answer_field(parse_answer(value), reason_field)
        if not _is_missing(candidate) and to_vote(candidate) is None:
            reasons[data_id] = str(candidate).strip()
    return reasons


def apply_manual(decisions: pd.DataFrame, manual: pd.DataFrame | None, stage: str) -> pd.DataFrame:
    """Add the ``manual`` columns for ``stage`` and the final ``decision``: a manual yes/no wins."""

    decisions = decisions.copy()
    by_id = {}
    if manual is not None and manual.shape[0]:
        for _, row in manual[manual["stage"] == stage].iterrows():
            by_id[str(row["data_id"])] = (row["manual"], row["manual_reason"])
    decisions["manual"] = [by_id.get(data_id, ("", ""))[0] for data_id in decisions["data_id"]]
    decisions["manual_reason"] = [by_id.get(data_id, ("", ""))[1] for data_id in decisions["data_id"]]

    final = []
    for llm_decision, manual_value in zip(decisions["llm_decision"], decisions["manual"]):
        vote = to_vote(manual_value) if manual_value else None
        if vote is None:
            final.append(llm_decision)
        else:
            final.append(KEPT if vote else EXCLUDED)
    decisions["decision"] = final
    return decisions


def screen(
    data_csv,
    db_path,
    field: str,
    rule: str,
    *,
    filters: dict | None = None,
    manual: pd.DataFrame | None = None,
    stage: str = STAGE_SCREENING,
    reason_field: str | None = None,
) -> tuple[pd.DataFrame, str]:
    """Decisions for every record of an experiment's ``data.csv``, in ``DECISION_COLUMNS``.

    ``llm_decision`` comes from the answers left after ``filters``, combined by
    ``rule``; ``decision`` is the manual value from the decision file where there
    is one. Undecided records (no usable answer) count as kept.
    """

    answers = filter_answers(load_answers(db_path), filters)
    data_ids, titles = _data_table(data_csv)
    decisions = screening_decisions(answers, data_ids, field, rule).rename(columns={"decision": "llm_decision"})
    decisions.insert(0, "stage", stage)
    decisions.insert(2, "title", [titles[data_id] for data_id in decisions["data_id"]])
    excluded = decisions.loc[decisions["llm_decision"] == EXCLUDED, "data_id"]
    reasons = _llm_reasons(answers, excluded, reason_field)
    decisions["llm_reason"] = [reasons.get(data_id, "") for data_id in decisions["data_id"]]
    decisions = apply_manual(decisions, manual, stage)[DECISION_COLUMNS]

    models = answers[["provider_name", "model_name", "profile_name"]].drop_duplicates().shape[0]
    prompts = answers["prompt_id"].nunique()
    note = f"field '{field}', rule '{rule}' over {models} model/profile(s) and {prompts} prompt(s)"
    if describe_filters(filters):
        note += f" (only {describe_filters(filters)})"
    note += f", {Path(db_path).name}"
    undecided = int((decisions["decision"] == UNDECIDED).sum())
    if undecided:
        note += f"; {undecided} undecided (no usable answer) counted as kept"
    manual_count = int((decisions["manual"] != "").sum())
    if manual_count:
        note += f"; {manual_count} manual decision(s)"
    return decisions, note


def not_retrieved(project_dir, searches, data_csv, kept_ids) -> tuple[int | None, str]:
    """Kept records whose PDF is not in the project, matched by DOI to the search CSVs."""

    data = _read_csv(data_csv)
    if "filename" in data.columns:
        return 0, "screened from extracted papers (copy-papers): every report is in papers/"
    if "doi" not in data.columns:
        return None, "data.csv has no 'doi' column to match against the searches"

    downloaded = set()
    searches_dir = Path(project_dir) / SEARCHES_DIR
    papers_dir = Path(project_dir) / "papers"
    for search_id in searches:
        for suffix in (RESULTS_SUFFIX, FILTERED_SUFFIX):
            path = searches_dir / f"{search_id}{suffix}"
            if not path.is_file():
                continue
            for _, row in _read_csv(path).iterrows():
                has_pdf = str(row.get("pdf_downloaded")).strip().lower() in {"true", "1"}
                pdf = row.get("pdf_filename")
                if not has_pdf and not _is_missing(pdf):
                    has_pdf = (papers_dir / str(pdf)).is_file()
                doi = _normalize_text(row.get("doi"))
                if has_pdf and doi:
                    downloaded.add(doi)

    doi_by_id = {str(row["ID"]).strip(): _normalize_text(row.get("doi")) for _, row in data.iterrows()}
    missing = [data_id for data_id in kept_ids if doi_by_id.get(str(data_id)) not in downloaded]
    return len(missing), "kept records without a downloaded PDF (pdf_downloaded or papers/), matched by DOI"


def fulltext_assessment(
    data_csv,
    db_path,
    field: str,
    rule: str,
    reason_field: str | None,
    *,
    filters: dict | None = None,
    manual: pd.DataFrame | None = None,
) -> tuple[dict, str, pd.DataFrame]:
    """Assessed, excluded-by-reason and included counts from a full-text experiment, with its decisions."""

    decisions, note = screen(
        data_csv, db_path, field, rule, filters=filters, manual=manual, stage=STAGE_FULLTEXT, reason_field=reason_field
    )
    excluded = decisions[decisions["decision"] == EXCLUDED]
    reasons: Counter = Counter()
    for manual_reason, llm_reason in zip(excluded["manual_reason"], excluded["llm_reason"]):
        reasons[manual_reason or llm_reason or REASON_NOT_STATED] += 1

    result = {
        "assessed": len(decisions),
        "excluded_reasons": dict(sorted(reasons.items())),
        "included": len(decisions) - len(excluded),
    }
    if reason_field:
        note += f", reasons from '{reason_field}'"
    return result, f"full-text experiment: {note}", decisions


# ---------------------------------------------------------------------------
# All boxes together
# ---------------------------------------------------------------------------


def derive(
    project_dir,
    *,
    searches=None,
    screening_data=None,
    screening_db=None,
    field=None,
    rule="majority",
    fulltext_data=None,
    fulltext_db=None,
    fulltext_field=None,
    reason_field=None,
    screening_filters=None,
    fulltext_filters=None,
    manual=None,
) -> tuple[dict, pd.DataFrame]:
    """``(counts, decisions)``: every box as ``{"value", "source"}``, and the decision table.

    ``value`` is ``None`` when unknown. ``decisions`` holds one row per record and
    stage (``DECISION_COLUMNS``), followed by the manual rows of ``manual`` that
    match no record this run, so the decision file never loses what was typed in.
    """

    project_dir = Path(project_dir)
    searches = list(searches) if searches else list_searches(project_dir)
    counts = {key: {"value": None, "source": "not derived - set it in overrides"} for key in BOXES}
    for key in ("screen_excluded", "sought", "not_retrieved", "assessed"):
        counts[key]["source"] = "needs a screening answer field (or an override)"
    for key in ("excluded_reasons", "included"):
        counts[key]["source"] = "needs a full-text experiment and its answer field (or an override)"
    counts["other_removed"] = {"value": 0, "source": "llmexer removes nothing for other reasons"}
    tables = []

    if searches:
        by_database, note = identified_by_database(project_dir, searches)
        counts["identified_databases"] = {"value": by_database, "source": note}
        value, note = duplicates_removed(project_dir, searches)
        counts["duplicates"] = {"value": value, "source": note}
        value, note = removed_by_automation(project_dir, searches)
        counts["automation"] = {"value": value, "source": note}
        counts["screened"] = {
            "value": records_after_identification(project_dir, searches),
            "source": "unique records after duplicates and filters",
        }
    else:
        counts["identified_databases"]["source"] = "no search with a __results.csv"

    kept_ids = None
    if screening_data is not None and Path(screening_data).is_file():
        screened_ids = _data_ids(screening_data)
        counts["screened"] = {"value": len(screened_ids), "source": f"rows of {Path(screening_data).name}"}
        if field and screening_db is None:
            counts["screen_excluded"]["source"] = "no experiment database to read the answers from"
        elif field:
            decisions, note = screen(
                screening_data, screening_db, field, rule, filters=screening_filters, manual=manual
            )
            tables.append(decisions)
            excluded = int((decisions["decision"] == EXCLUDED).sum())
            kept_ids = list(decisions.loc[decisions["decision"] != EXCLUDED, "data_id"])
            counts["screen_excluded"] = {"value": excluded, "source": note}
            counts["sought"] = {"value": len(kept_ids), "source": "records kept at screening"}

    if kept_ids is not None and searches:
        value, note = not_retrieved(project_dir, searches, screening_data, kept_ids)
        counts["not_retrieved"] = {"value": value, "source": note}
        if value is not None:
            counts["assessed"] = {"value": len(kept_ids) - value, "source": "reports sought minus not retrieved"}

    if fulltext_data is not None and fulltext_field and fulltext_db is None:
        for key in ("excluded_reasons", "included"):
            counts[key]["source"] = "the full-text experiment has no database to read the answers from"
    elif fulltext_data is not None and fulltext_field and Path(fulltext_data).is_file():
        result, note, decisions = fulltext_assessment(
            fulltext_data, fulltext_db, fulltext_field, rule, reason_field, filters=fulltext_filters, manual=manual
        )
        tables.append(decisions)
        for key in ("assessed", "excluded_reasons", "included"):
            counts[key] = {"value": result[key], "source": note}
        if kept_ids is not None:
            # With a full-text experiment, a report counts as retrieved when it was assessed.
            missing = set(kept_ids) - set(_data_ids(fulltext_data))
            counts["not_retrieved"] = {
                "value": len(missing),
                "source": "kept at screening but not in the full-text experiment's data.csv",
            }

    decisions = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=DECISION_COLUMNS)
    unmatched = _unmatched_manual(decisions, manual)
    if unmatched.shape[0]:
        decisions = pd.concat([decisions, unmatched], ignore_index=True)
    return counts, decisions.astype(object).fillna("")[DECISION_COLUMNS]


def _unmatched_manual(decisions: pd.DataFrame, manual: pd.DataFrame | None) -> pd.DataFrame:
    if manual is None or not manual.shape[0]:
        return pd.DataFrame(columns=DECISION_COLUMNS)
    matched = set(zip(decisions["stage"], decisions["data_id"].astype(str)))
    rows = [row for _, row in manual.iterrows() if (row["stage"], str(row["data_id"])) not in matched]
    return pd.DataFrame(rows, columns=DECISION_COLUMNS) if rows else pd.DataFrame(columns=DECISION_COLUMNS)


def derive_counts(project_dir, **options) -> dict:
    """Every box as ``{"value": ..., "source": ...}``; see :func:`derive`."""

    return derive(project_dir, **options)[0]


# ---------------------------------------------------------------------------
# Decision file: per-record decisions, with the user's manual column
# ---------------------------------------------------------------------------


def load_decisions_file(path) -> pd.DataFrame:
    """The manual rows of a decision file (``stage``, ``data_id``, ``manual``, ``manual_reason``).

    Raises:
        ValueError: on a missing column or a ``manual`` value that is not yes/no.
    """

    path = Path(path)
    columns = ["stage", "data_id", *MANUAL_COLUMNS]
    if not path.is_file():
        return pd.DataFrame(columns=columns)
    table = pd.read_csv(path, sep=CSV_SEPARATOR, encoding="utf-8", dtype=str, keep_default_na=False)
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"'{path}' has no column(s) {', '.join(missing)}.")
    table = table[columns].apply(lambda column: column.str.strip())
    table = table[(table["manual"] != "") | (table["manual_reason"] != "")]
    bad = [
        f"{row.stage} {row.data_id}: {row.manual!r}"
        for row in table.itertuples()
        if row.manual and to_vote(row.manual) is None
    ]
    if bad:
        raise ValueError(f"'manual' must be yes or no (or empty) in '{path}': {'; '.join(bad)}.")
    return table.reset_index(drop=True)


def decisions_file_text(decisions: pd.DataFrame) -> str:
    # "\n", not os.linesep: the text is written in text mode, which adds the "\r" on Windows.
    return decisions.to_csv(sep=CSV_SEPARATOR, index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# Counts file: settings, derived values + the user's overrides
# ---------------------------------------------------------------------------

# What a run was made with; `draw-prisma` re-uses them when the option is not given.
SETTINGS_KEYS = (
    "exp",
    "searches",
    "field",
    "llm_vote_rule",
    "prompt",
    "filter_provider",
    "filter_model",
    "filter_profile",
    "fulltext_exp",
    "fulltext_field",
    "fulltext_prompt",
    "reason_field",
    "title",
)


def answer_filters(settings: dict, stage: str = STAGE_SCREENING) -> dict:
    """The answer filters of a stage, from the settings: the prompt filter is per stage."""

    prompt_key = "prompt" if stage == STAGE_SCREENING else "fulltext_prompt"
    return {
        "prompt": settings.get(prompt_key) or [],
        "provider": settings.get("filter_provider") or [],
        "model": settings.get("filter_model") or [],
        "profile": settings.get("filter_profile") or [],
    }


def load_counts_file(path) -> tuple[dict, dict]:
    """``(derived, overrides)`` from a counts file; both empty when there is none yet."""

    path = Path(path)
    if not path.is_file():
        return {}, {}
    content = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    overrides = content.get("overrides") or {}
    unknown = sorted(set(overrides) - set(BOXES))
    if unknown:
        raise ValueError(f"Unknown key(s) in the overrides of '{path}': {', '.join(unknown)}.")
    return content.get("derived") or {}, overrides


def load_settings(path) -> dict:
    """The ``settings`` of a counts file; empty when there is none yet."""

    path = Path(path)
    if not path.is_file():
        return {}
    settings = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("settings") or {}
    unknown = sorted(set(settings) - set(SETTINGS_KEYS))
    if unknown:
        raise ValueError(f"Unknown key(s) in the settings of '{path}': {', '.join(unknown)}.")
    return settings


def counts_file_text(derived: dict, overrides: dict, settings: dict | None = None, inputs: dict | None = None) -> str:
    content = {}
    if settings:
        content["settings"] = {key: settings[key] for key in SETTINGS_KEYS if settings.get(key) not in (None, [], "")}
    if inputs:
        content["inputs"] = inputs
    content["derived"] = derived
    content["overrides"] = overrides or {}
    body = yaml.safe_dump(content, sort_keys=False, allow_unicode=True, width=120)
    return COUNTS_FILE_HEADER + body


def save_counts_file(path, derived: dict, overrides: dict, settings: dict | None = None) -> None:
    Path(path).write_text(counts_file_text(derived, overrides, settings), encoding="utf-8")


def effective_counts(derived: dict, overrides: dict) -> dict:
    """The values drawn: an override wins over the derived value."""

    values = {key: (derived.get(key) or {}).get("value") for key in BOXES}
    for key, value in (overrides or {}).items():
        values[key] = value
    return values


def _total(value):
    if isinstance(value, dict):
        return sum(int(v) for v in value.values() if v is not None)
    return value


def check_counts(values: dict) -> list[str]:
    """PRISMA consistency checks; each problem is one warning sentence."""

    warnings = []
    identified = _total(values.get("identified_databases"))
    removed = [values.get(key) for key in ("duplicates", "automation", "other_removed")]
    screened = values.get("screened")
    if identified is not None and screened is not None and None not in removed:
        if screened > identified - sum(removed):
            warnings.append(
                f"Records screened ({screened}) exceed records identified minus removed "
                f"({identified} - {sum(removed)} = {identified - sum(removed)})."
            )
    excluded = values.get("screen_excluded")
    if screened is not None and excluded is not None and excluded > screened:
        warnings.append(f"Records excluded ({excluded}) exceed records screened ({screened}).")
    sought, missing, assessed = values.get("sought"), values.get("not_retrieved"), values.get("assessed")
    if None not in (sought, missing, assessed) and sought != missing + assessed:
        warnings.append(f"Reports sought ({sought}) != not retrieved ({missing}) + assessed ({assessed}).")
    reasons = _total(values.get("excluded_reasons"))
    included = values.get("included")
    if None not in (assessed, reasons, included) and assessed != reasons + included:
        warnings.append(f"Reports assessed ({assessed}) != excluded ({reasons}) + included ({included}).")
    return warnings


# ---------------------------------------------------------------------------
# SVG
# ---------------------------------------------------------------------------

FONT = "Arial, Helvetica, sans-serif"
FONT_SIZE = 13
LINE_HEIGHT = 18
CHARS_PER_LINE = 46
BOX_WIDTH = 360
BOX_PADDING = 12
GAP_X = 70
GAP_Y = 40
BAND_WIDTH = 34
MARGIN = 20
COLOR_BOX = "#ffffff"
COLOR_BORDER = "#1f2937"
COLOR_BAND = "#93c5fd"
COLOR_TITLE = "#fde68a"
COLOR_MISSING = "#b91c1c"


def _n(value) -> str:
    return "?" if value is None else str(value)


def _box_lines(values: dict) -> list[tuple[list[str], list[str]]]:
    """Text lines of each row: (left box, right box). An empty right box is not drawn."""

    databases = values.get("identified_databases")
    identified = ["Records identified from:", f"Databases (n = {_n(_total(databases))})"]
    if isinstance(databases, dict):
        identified += [f"  {name} (n = {count})" for name, count in databases.items()]
    removed = [
        "Records removed before screening:",
        f"  Duplicate records removed (n = {_n(values.get('duplicates'))})",
        f"  Records marked as ineligible by automation tools (n = {_n(values.get('automation'))})",
        f"  Records removed for other reasons (n = {_n(values.get('other_removed'))})",
    ]
    reasons = values.get("excluded_reasons")
    excluded_reports = [f"Reports excluded (n = {_n(_total(reasons))})"]
    if isinstance(reasons, dict):
        excluded_reports += [f"  {reason} (n = {count})" for reason, count in reasons.items()]
    return [
        (identified, removed),
        (
            [f"Records screened (n = {_n(values.get('screened'))})"],
            [f"Records excluded (n = {_n(values.get('screen_excluded'))})"],
        ),
        (
            [f"Reports sought for retrieval (n = {_n(values.get('sought'))})"],
            [f"Reports not retrieved (n = {_n(values.get('not_retrieved'))})"],
        ),
        ([f"Reports assessed for eligibility (n = {_n(values.get('assessed'))})"], excluded_reports),
        (
            [
                f"Studies included in review (n = {_n(values.get('included'))})",
                f"Reports of included studies (n = {_n(values.get('included'))})",
            ],
            [],
        ),
    ]


def _wrap(lines: list[str]) -> list[str]:
    wrapped = []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        wrapped += textwrap.wrap(
            line.strip(),
            CHARS_PER_LINE - indent,
            initial_indent=" " * indent,
            subsequent_indent=" " * (indent + 2),
        ) or [""]
    return wrapped


def _box(x: int, y: int, height: int, lines: list[str]) -> list[str]:
    missing = any("n = ?" in line for line in lines)
    dash = ' stroke-dasharray="6 4"' if missing else ""
    parts = [
        f'<rect x="{x}" y="{y}" width="{BOX_WIDTH}" height="{height}" fill="{COLOR_BOX}" '
        f'stroke="{COLOR_BORDER}" stroke-width="1.5"{dash}/>'
    ]
    for index, line in enumerate(lines):
        indent = (len(line) - len(line.lstrip(" "))) * 4
        color = COLOR_MISSING if "n = ?" in line else COLOR_BORDER
        text_y = y + BOX_PADDING + FONT_SIZE + index * LINE_HEIGHT
        parts.append(
            f'<text x="{x + BOX_PADDING + indent}" y="{text_y}" font-family="{FONT}" '
            f'font-size="{FONT_SIZE}" fill="{color}">{escape(line.strip())}</text>'
        )
    return parts


def _arrow(x1: int, y1: int, x2: int, y2: int) -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLOR_BORDER}" '
        'stroke-width="1.5" marker-end="url(#arrow)"/>'
    )


def _band(y: int, height: int, label: str) -> list[str]:
    center_x = MARGIN + BAND_WIDTH / 2
    center_y = y + height / 2
    return [
        f'<rect x="{MARGIN}" y="{y}" width="{BAND_WIDTH}" height="{height}" rx="6" fill="{COLOR_BAND}"/>',
        f'<text x="{center_x}" y="{center_y}" font-family="{FONT}" font-size="{FONT_SIZE + 1}" '
        f'font-weight="bold" text-anchor="middle" dominant-baseline="middle" '
        f'transform="rotate(-90 {center_x} {center_y})">{escape(label)}</text>',
    ]


def render_svg(values: dict, title: str | None = None) -> str:
    """The PRISMA 2020 flow diagram (new review, databases only) as an SVG document."""

    left_x = MARGIN + BAND_WIDTH + 16
    right_x = left_x + BOX_WIDTH + GAP_X
    width = right_x + BOX_WIDTH + MARGIN

    y = MARGIN
    parts = []
    if title:
        parts.append(
            f'<rect x="{left_x}" y="{y}" width="{right_x + BOX_WIDTH - left_x}" height="32" '
            f'rx="4" fill="{COLOR_TITLE}"/>'
        )
        parts.append(
            f'<text x="{(left_x + right_x + BOX_WIDTH) / 2}" y="{y + 21}" font-family="{FONT}" '
            f'font-size="{FONT_SIZE + 2}" font-weight="bold" text-anchor="middle">{escape(title)}</text>'
        )
        y += 32 + GAP_Y / 2

    # Rows 0 and 4 carry a phase band of their own, which must fit its rotated label.
    band_labels = {0: "Identification", 4: "Included"}
    rows = []
    for index, (left, right) in enumerate(_box_lines(values)):
        left_lines, right_lines = _wrap(left), _wrap(right)
        height = max(len(left_lines), len(right_lines)) * LINE_HEIGHT + 2 * BOX_PADDING
        if index in band_labels:
            height = max(height, len(band_labels[index]) * 9 + 2 * BOX_PADDING)
        rows.append((y, height, left_lines, right_lines))
        y += height + GAP_Y

    for index, (row_y, height, left_lines, right_lines) in enumerate(rows):
        parts += _box(left_x, row_y, height, left_lines)
        if right_lines:
            parts += _box(right_x, row_y, height, right_lines)
            parts.append(_arrow(left_x + BOX_WIDTH, row_y + height // 2, right_x - 4, row_y + height // 2))
        if index + 1 < len(rows):
            next_y = rows[index + 1][0]
            parts.append(_arrow(left_x + BOX_WIDTH // 2, row_y + height, left_x + BOX_WIDTH // 2, next_y - 4))

    # Phase bands: Identification (row 0), Screening (rows 1-3), Included (row 4).
    parts += _band(rows[0][0], rows[0][1], "Identification")
    parts += _band(rows[1][0], rows[3][0] + rows[3][1] - rows[1][0], "Screening")
    parts += _band(rows[4][0], rows[4][1], "Included")

    height = y - GAP_Y + MARGIN
    header = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" '
        f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{COLOR_BORDER}"/></marker></defs>'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>'
    )
    return header + "\n" + "\n".join(parts) + "\n</svg>\n"


def backup_name(folder, stem: str, suffix: str) -> str:
    """The next ``<stem>_backup_<YYYYMMDD>_<NN><suffix>`` name in ``folder`` (copy of llmexer's)."""

    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    prefix = f"{stem}_backup_{date}_"
    counter = 0
    if os.path.isdir(folder):
        for fname in os.listdir(folder):
            if fname.startswith(prefix) and fname.endswith(suffix):
                token = fname[len(prefix) : -len(suffix)]
                try:
                    counter = max(counter, int(token))
                except ValueError:
                    continue
    return f"{prefix}{counter + 1:02d}{suffix}"


def plan_outputs(outputs: list) -> list:
    """``(path, text, backup_path)`` for each ``(path, text)`` about to be written.

    ``backup_path`` is set only when the file exists and its content would change,
    so re-running with the same numbers does not pile up identical copies.
    """

    planned = []
    for path, text in outputs:
        path = Path(path)
        backup = None
        if path.is_file() and path.read_text(encoding="utf-8") != text:
            folder = path.parent / BACKUP_DIR
            backup = folder / backup_name(folder, path.stem, path.suffix)
        planned.append((path, text, backup))
    return planned


def write_outputs(planned: list) -> list:
    """Write the planned files, backing up each changed one first. Returns the backups made."""

    backups = []
    for path, text, backup in planned:
        os.makedirs(path.parent, exist_ok=True)
        if backup is not None:
            os.makedirs(backup.parent, exist_ok=True)
            shutil.copy2(path, backup)
            backups.append(backup)
        path.write_text(text, encoding="utf-8")
    return backups


def output_texts(
    counts_path,
    svg_path,
    derived: dict,
    overrides: dict,
    values: dict,
    title=None,
    *,
    settings: dict | None = None,
    inputs: dict | None = None,
    decisions_path=None,
    decisions: pd.DataFrame | None = None,
) -> list:
    """``(path, text)`` of every output: counts file, SVG and, when there are decisions, the decision file."""

    outputs = [
        (counts_path, counts_file_text(derived, overrides, settings, inputs)),
        (svg_path, render_svg(values, title)),
    ]
    if decisions_path is not None and decisions is not None and decisions.shape[0]:
        outputs.append((decisions_path, decisions_file_text(decisions)))
    return outputs


def save_outputs(counts_path, svg_path, derived: dict, overrides: dict, values: dict, title=None, **files) -> list:
    """Save the outputs (see :func:`output_texts`), backing up the ones that change. Returns the backups made."""

    return write_outputs(plan_outputs(output_texts(counts_path, svg_path, derived, overrides, values, title, **files)))


def write_svg(path, values: dict, title: str | None = None) -> Path:
    path = Path(path)
    os.makedirs(path.parent, exist_ok=True)
    path.write_text(render_svg(values, title), encoding="utf-8")
    return path
