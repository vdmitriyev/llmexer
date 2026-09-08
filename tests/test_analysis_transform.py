"""Tests for the copied `transform` analysis module."""

import json

import pandas as pd
import pytest

from llmexer.base.analysis.transform import (
    FLATTEN_ERROR_COLUMN,
    SEARCH_COLUMNS,
    MissingColumnError,
    export_as_csv,
    flatten_llm_response,
    flattened_only,
    list_papers,
    load_experiment_db,
    load_search_frame,
    parse_json_payload,
    strip_code_fence,
)
from llmexer.base.dao import ExperimentDAO
from llmexer.common import strip_code_fence as shared_strip_code_fence
from tests.db_helpers import LITELLM_ROW, OLLAMA_ROW, seed_db

# One value per edge case the flattener has to survive, in frame order.
_FENCED = '```json\n{"status": "ok", "nested": {"deep": 2}}\n```'
_VALUES = [
    '{"score": 1}',  # plain object
    None,  # never run
    "",  # empty answer
    "not json at all",  # prose where JSON was asked for
    _FENCED,  # fenced JSON, and a key colliding with the frame's own column
    '{"tags": [1, 2]}',  # list nested inside an object
    "[1, 2, 3]",  # top-level array
    {"score": 9},  # already parsed upstream
]

_FENCE_FIXTURES = [
    "plain text",
    '```json\n{"a": 1}\n```',
    "```\nno language tag\n```",
    "```json\nunterminated",
    '   ```json\n{"b": 2}\n```   ',
    "",
]


@pytest.fixture()
def frame():
    """A frame with a non-default index and a `status` column to collide with."""

    return pd.DataFrame(
        {
            "code": [f"D{i:02d}" for i in range(len(_VALUES))],
            "status": ["success"] * len(_VALUES),
            "response_text": _VALUES,
        },
        index=range(100, 100 + len(_VALUES)),
    )


# ---------------------------------------------------------------------------
# strip_code_fence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", _FENCE_FIXTURES)
def test_strip_code_fence_matches_the_shared_helper(text):
    """The copied implementation must not drift from `llmexer.common`.

    `transform.py` is copied into a project and cannot import from `llmexer`, so
    the duplication is deliberate - this is the guard that keeps them in step.
    """

    assert strip_code_fence(text) == shared_strip_code_fence(text)


def test_strip_code_fence_removes_language_tag():
    assert strip_code_fence('```json\n{"a": 1}\n```') == '{"a": 1}'


# ---------------------------------------------------------------------------
# parse_json_payload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, None),
        ("", None),
        ("   ", None),
        ('{"a": 1}', {"a": 1}),
        ({"a": 1}, {"a": 1}),
        ("[1, 2]", [1, 2]),
        (float("nan"), None),
    ],
)
def test_parse_json_payload_returns_no_error(value, expected):
    parsed, error = parse_json_payload(value)

    assert parsed == expected
    assert error == ""


def test_parse_json_payload_reports_invalid_json():
    parsed, error = parse_json_payload("not json")

    assert parsed is None
    assert "invalid JSON" in error


# ---------------------------------------------------------------------------
# flatten
# ---------------------------------------------------------------------------


def test_flatten_llm_response_preserves_rows_and_index(frame):
    """Row count and index must survive - the notebook joins on them later."""

    out = flatten_llm_response(frame)

    assert len(out) == len(frame)
    assert list(out.index) == list(frame.index)


def test_flatten_llm_response_expands_object_keys(frame):
    out = flatten_llm_response(frame)

    assert out.loc[100, "score"] == 1
    assert out.loc[107, "score"] == 9


def test_flatten_llm_response_expands_nested_objects_with_dotted_names(frame):
    out = flatten_llm_response(frame)

    assert out.loc[104, "nested.deep"] == 2


def test_flatten_llm_response_keeps_a_nested_list_as_a_cell_value(frame):
    out = flatten_llm_response(frame)

    assert out.loc[105, "tags"] == [1, 2]


def test_flatten_llm_response_does_not_overwrite_an_existing_column(frame):
    """A model answering {"status": ...} must not clobber the row's own status."""

    out = flatten_llm_response(frame)

    assert list(out["status"]) == ["success"] * len(frame)
    assert out.loc[104, "status_flat"] == "ok"


def test_flatten_llm_response_keeps_a_top_level_list_whole_without_exploding_rows(frame):
    out = flatten_llm_response(frame)

    assert out.loc[106, "response_text_items"] == [1, 2, 3]
    assert len(out) == len(frame)


def test_flatten_llm_response_reports_only_the_malformed_row(frame):
    out = flatten_llm_response(frame)
    errors = out[FLATTEN_ERROR_COLUMN]

    assert errors.loc[103].startswith("invalid JSON")
    assert list(errors.drop(index=103)) == [""] * (len(frame) - 1)


def test_flatten_llm_response_never_raises_on_a_malformed_row(frame):
    """A single bad answer is data, not a bug - it must not abort the notebook."""

    out = flatten_llm_response(frame)

    assert out.loc[103, "score"] != out.loc[103, "score"]  # NaN


def test_flatten_llm_response_missing_column_raises_a_typed_error(frame):
    with pytest.raises(MissingColumnError) as excinfo:
        flatten_llm_response(frame, column="nope")

    assert "nope" in str(excinfo.value)
    assert "response_text" in str(excinfo.value)


def test_flatten_llm_response_missing_column_is_catchable_as_key_error(frame):
    """The copied module cannot import llmexer's exceptions, so it subclasses KeyError."""

    with pytest.raises(KeyError):
        flatten_llm_response(frame, column="nope")


def test_flatten_llm_response_prefix_namespaces_the_new_columns(frame):
    out = flatten_llm_response(frame, prefix="r_")

    assert "r_score" in out.columns
    assert "score" not in out.columns


def test_flatten_llm_response_errors_raise_mode(frame):
    with pytest.raises(ValueError):
        flatten_llm_response(frame, errors="raise")


def test_flatten_llm_response_errors_ignore_mode_adds_no_column(frame):
    out = flatten_llm_response(frame, errors="ignore")

    assert FLATTEN_ERROR_COLUMN not in out.columns


def test_flatten_llm_response_rejects_an_unknown_errors_mode(frame):
    with pytest.raises(ValueError):
        flatten_llm_response(frame, errors="whatever")


def test_flatten_llm_response_does_not_mutate_the_input(frame):
    before = list(frame.columns)
    flatten_llm_response(frame)

    assert list(frame.columns) == before


def test_flatten_llm_response_on_an_empty_frame():
    out = flatten_llm_response(pd.DataFrame({"response_text": pd.Series(dtype="object")}))

    assert out.empty


# ---------------------------------------------------------------------------
# export_as_csv
# ---------------------------------------------------------------------------


def test_export_as_csv_uses_the_project_csv_dialect(tmp_path):
    """Semicolon-separated UTF-8, like every other CSV a project holds."""

    path = export_as_csv(pd.DataFrame({"a": [1], "b": ["x;y"]}), tmp_path / "out.csv")
    text = path.read_text(encoding="utf-8")

    assert text.splitlines()[0] == "a;b"
    assert pd.read_csv(path, sep=";").iloc[0]["b"] == "x;y"


def test_export_as_csv_writes_no_index_column(tmp_path):
    df = pd.DataFrame({"a": [1, 2]}, index=[7, 8])
    path = export_as_csv(df, tmp_path / "out.csv")

    assert list(pd.read_csv(path, sep=";").columns) == ["a"]


def test_export_as_csv_creates_parents_and_returns_the_path(tmp_path):
    path = export_as_csv(pd.DataFrame({"a": [1]}), tmp_path / "nested" / "deep" / "out.csv")

    assert path.is_file()
    assert path == tmp_path / "nested" / "deep" / "out.csv"


def test_export_as_csv_round_trips_the_flattened_frame(tmp_path, frame):
    out = flatten_llm_response(frame)
    path = export_as_csv(out, tmp_path / "flat.csv")
    back = pd.read_csv(path, sep=";")

    assert len(back) == len(out)
    assert "score" in back.columns


# ---------------------------------------------------------------------------
# load_experiment_db
# ---------------------------------------------------------------------------


def _payload(prompt_tokens=11, completion_tokens=7):
    """A stored per-call payload shaped like the one `experiment run` writes."""

    return json.dumps(
        {
            "model": "m",
            "raw_response": {
                "choices": [{"finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        }
    )


@pytest.fixture()
def seeded_db(tmp_path):
    path = tmp_path / "experiment_20240101_01.db"
    seed_db(
        path,
        {
            "ollama": [
                dict(OLLAMA_ROW, ID=1, status="success", total_tokens=100, usage_tokens=100, response_json=_payload()),
                dict(
                    OLLAMA_ROW,
                    ID=2,
                    code="D02_prompt01_llama3.3:latest_ollama-default",
                    status="success",
                    total_tokens=None,
                    usage_tokens=42,
                    response_json=None,
                ),
            ],
            "litellm": [dict(LITELLM_ROW, ID=3)],
        },
    )
    return path


def test_load_experiment_db_matches_the_dao(seeded_db):
    """The plain-sqlite loader must not drift from `ExperimentDAO.fetch_rows()`."""

    with ExperimentDAO(str(seeded_db)) as dao:
        rows = dao.fetch_rows()
    df = load_experiment_db(seeded_db)

    assert len(df) == len(rows)
    assert list(df["ID"]) == [row["ID"] for row in rows]
    assert set(rows[0]).issubset(set(df.columns))


def test_load_experiment_db_joins_the_params_table(seeded_db):
    """Hyperparameters live in params_<provider> and must come along."""

    df = load_experiment_db(seeded_db).set_index("ID")

    assert df.loc[1, "temperature"] == OLLAMA_ROW["temperature"]
    assert df.loc[1, "ollama_context_window"] == OLLAMA_ROW["ollama_context_window"]


def test_load_experiment_db_tags_the_provider(seeded_db):
    df = load_experiment_db(seeded_db).set_index("ID")

    assert df.loc[1, "_provider"] == "ollama"
    assert df.loc[3, "_provider"] == "litellm"


def test_load_experiment_db_keeps_try_tables_out(seeded_db):
    """`try_experiment_*` does not start with `experiment_`, so tries stay out."""

    with ExperimentDAO(str(seeded_db)) as dao:
        dao.ensure_try_tables("ollama")
        dao.append_try_row("ollama", dict(OLLAMA_ROW, ID=99))

    assert 99 not in set(load_experiment_db(seeded_db)["ID"])


def test_load_experiment_db_parses_the_token_split(seeded_db):
    df = load_experiment_db(seeded_db).set_index("ID")

    assert df.loc[1, "prompt_tokens"] == 11
    assert df.loc[1, "completion_tokens"] == 7


def test_missing_usage_stays_null_and_is_never_zero_filled(seeded_db):
    """A zero would understate prompt cost; the gap has to stay visible."""

    df = load_experiment_db(seeded_db).set_index("ID")

    assert pd.isna(df.loc[2, "prompt_tokens"])
    assert str(df["prompt_tokens"].dtype) == "Int64"


def test_load_experiment_db_without_a_database_is_empty():
    """A project with nothing generated yet must not break the notebook."""

    assert load_experiment_db(None).empty


def test_load_experiment_db_raises_on_a_missing_path(tmp_path):
    """A path that was given but does not exist is a typo, not an empty project."""

    with pytest.raises(FileNotFoundError):
        load_experiment_db(tmp_path / "nope.db")


# ---------------------------------------------------------------------------
# load_search_frames / list_papers
# ---------------------------------------------------------------------------

_SEARCH_HEADER = (
    "search_engine_internal_id;year;title;authors;abstract;isOpenAccess;doi;language;"
    "citationCount;referenceCount;entry_source;pdf_filename;txt_filename;markdown_filename;pdf_downloaded"
)


def _write_search(folder, name, header=_SEARCH_HEADER, source="Semantic Scholar"):
    (folder / name).write_text(
        f"{header}\nabc;2024;A title;An author;An abstract;True;10.1/x;en;5;7;{source};;;;False\n",
        encoding="utf-8",
    )


def test_load_search_frame_reads_the_named_results(tmp_path):
    _write_search(tmp_path, "s1__results.csv")
    search = {"search": "s1.yaml", "results": "s1__results.csv", "filter": None}

    df = load_search_frame(tmp_path, search)

    assert len(df) == 1
    assert df.iloc[0]["search_id"] == "s1"
    assert df.iloc[0]["source_file"] == "s1__results.csv"


def test_load_search_frame_reads_one_search_only(tmp_path):
    """Two searches are two populations; combining them would describe neither."""

    _write_search(tmp_path, "s1__results.csv")
    _write_search(tmp_path, "s2__results.csv")

    df = load_search_frame(tmp_path, {"search": "s1.yaml", "results": "s1__results.csv", "filter": None})

    assert list(df["source_file"].unique()) == ["s1__results.csv"]


def test_load_search_frame_for_a_search_never_run(tmp_path):
    """A search created but not yet run has no results CSV."""

    df = load_search_frame(tmp_path, {"search": "s2.yaml", "results": None, "filter": None})

    assert df.empty
    assert set(SEARCH_COLUMNS).issubset(set(df.columns))


def test_load_search_frame_when_the_listed_file_is_gone(tmp_path):
    df = load_search_frame(tmp_path, {"search": "s1.yaml", "results": "s1__results.csv", "filter": None})

    assert df.empty
    assert set(SEARCH_COLUMNS).issubset(set(df.columns))


def test_load_search_frame_without_a_search_at_all(tmp_path):
    """`SEARCHES` is empty on a project that has never run a search."""

    assert load_search_frame(tmp_path, None).empty


def test_load_search_frame_reads_the_filtered_csv(tmp_path):
    _write_search(tmp_path, "s1__filtered.csv")
    search = {"search": "s1.yaml", "results": None, "filter": "s1__filtered.csv"}

    assert len(load_search_frame(tmp_path, search, kind="filter")) == 1
    assert load_search_frame(tmp_path, search, kind="results").empty


def test_load_search_frame_renames_the_legacy_id_column(tmp_path):
    """Pre-OpenAlex CSVs name the column after Semantic Scholar alone."""

    _write_search(
        tmp_path,
        "s1__results.csv",
        header=_SEARCH_HEADER.replace("search_engine_internal_id", "sem_scholar_paper_id"),
    )
    df = load_search_frame(tmp_path, {"search": "s1.yaml", "results": "s1__results.csv", "filter": None})

    assert "search_engine_internal_id" in df.columns
    assert "sem_scholar_paper_id" not in df.columns


def test_load_search_frame_coerces_numeric_columns(tmp_path):
    _write_search(tmp_path, "s1__results.csv")
    df = load_search_frame(tmp_path, {"search": "s1.yaml", "results": "s1__results.csv", "filter": None})

    assert df.iloc[0]["year"] == 2024
    assert str(df["citationCount"].dtype) == "Int64"


def test_load_search_frame_rejects_an_unknown_kind(tmp_path):
    with pytest.raises(ValueError):
        load_search_frame(tmp_path, None, kind="nope")


def test_list_papers_groups_extractions_by_stem(tmp_path):
    for name in ("a.pdf", "a.txt", "b.pdf", "b.md", "c.pdf", "notes.docx"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    df = list_papers(tmp_path).set_index("stem")

    assert list(df.index) == ["a", "b", "c"]
    assert list(df["extracted"]) == [True, True, False]


def test_list_papers_on_a_missing_folder(tmp_path):
    assert list_papers(tmp_path / "nope").empty


# ---------------------------------------------------------------------------
# flattened_only
# ---------------------------------------------------------------------------


def test_flattened_only_keeps_just_the_parsed_answer(frame):
    """The CSV export is a table of what the models said, nothing around it."""

    out = flattened_only(flatten_llm_response(frame))

    assert "score" in out.columns
    for original in ("code", "status", "response_text"):
        assert original not in out.columns


def test_flattened_only_drops_the_parse_metadata(frame):
    """`flatten_error` describes the parse, not the answer."""

    assert FLATTEN_ERROR_COLUMN not in flattened_only(flatten_llm_response(frame)).columns


def test_flattened_only_keeps_the_rows_and_index(frame):
    out = flattened_only(flatten_llm_response(frame))

    assert len(out) == len(frame)
    assert list(out.index) == list(frame.index)


def test_flattened_only_keeps_a_top_level_list(frame):
    """A JSON array is still an answer, so it travels with the values."""

    out = flattened_only(flatten_llm_response(frame))

    assert "response_text_items" in out.columns


def test_flattened_only_on_a_frame_that_was_never_flattened():
    assert flattened_only(pd.DataFrame({"a": [1]})).empty


def test_exported_csv_holds_only_the_answers(tmp_path, frame):
    out = flattened_only(flatten_llm_response(frame))
    path = export_as_csv(out, tmp_path / "flat.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]

    assert "response_text" not in header.replace("response_text_items", "")
    assert "score" in header
