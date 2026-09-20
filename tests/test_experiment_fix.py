"""Tests for the `experiment fix` command and its JSON extraction."""

import json
import os

import pytest
from typer.testing import CliRunner

from llmexer.base.dao import DATAFIX_TABLE, ExperimentDAO
from llmexer.base.datafix import (
    SKIP_NAME_TAKEN,
    SKIP_NO_SUCH_FIELD,
    SKIP_NOT_AN_OBJECT,
    extract_json,
    fixed_value,
    needs_fix,
    rename_json_field,
)
from llmexer.cli import app
from llmexer.exceptions import UnexpectedCLIParamsException
from tests.db_helpers import OLLAMA_ROW, seed_db, try_table_names

runner = CliRunner()

PID = "fix-test-exp"
_DB_NAME = "experiment_20240101_01.db"

_VALID = '{"relevant": true, "reason": "Discusses attention", "score": 9}'
_FENCED = '```json\n{"relevant": false, "reason": "Off topic"}\n```'
_TRAILING = '{"relevant": true, "score": 7}\n\nThis paper is clearly relevant because it evaluates attention.'
_LEADING = 'Here is the analysis you asked for:\n\n```json\n{"relevant": true, "score": 3}\n```'
_PROSE = "This paper is broadly relevant but does not answer the question directly."
_FENCED_FIELD = '```json\n{"paperTitle": "B", "relevant": "no"}\n```'


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


def _ran_row(row_id, code_suffix, response, **overrides):
    """An ollama row carrying the result columns `experiment run` fills in.

    Every row of one provider must carry the same keys: the rows are inserted
    with a single executemany, which rejects a batch with a ragged key set.
    """
    row = dict(OLLAMA_ROW)
    row.update(
        {
            "ID": row_id,
            "code": f"D{code_suffix}_prompt01_llama3.3:latest_ollama-default",
            "response_text": response,
            "status": "success",
            "state": "finished",
            "prompt_tokens": 100,
            "completion_tokens": 42,
            "total_tokens": 142,
            "elapsed_seconds": 2.47,
            "timestamp": "2026-09-04T10:00:00.123456+00:00",
        }
    )
    row.update(overrides)
    return row


@pytest.fixture()
def experiment_with_results(projects_dir):
    """A database with one valid, three broken, one prose and one unrun answer."""
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {
            "ollama": [
                _ran_row(1, "01", _VALID),
                _ran_row(2, "02", _FENCED),
                _ran_row(3, "03", _TRAILING),
                _ran_row(4, "04", _LEADING),
                _ran_row(5, "05", _PROSE),
                _ran_row(6, "06", None, status=None, state=None),
            ]
        },
    )
    return exp_subdir


def _fix(*options, pid=PID):
    return runner.invoke(app, ["experiment", "fix", "--pid", pid, "--file", _DB_NAME, *options])


def _responses(db_path):
    """Map row ID -> stored response_text."""
    with ExperimentDAO(str(db_path)) as dao:
        return {row["ID"]: row["response_text"] for row in dao.fetch_rows()}


def _logs(db_path):
    with ExperimentDAO(str(db_path)) as dao:
        return dao.fetch_datafix_logs()


# --------------------------------------------------------------- extract_json


@pytest.mark.parametrize(
    "text, expected",
    [
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": 1}\n\nBecause of reasons.', '{"a": 1}'),
        ('Here you go:\n{"a": 1}', '{"a": 1}'),
        ('{"a": {"b": [1, 2]}} trailing', '{"a": {"b": [1, 2]}}'),
        ('[{"a": 1}, {"a": 2}] and more', '[{"a": 1}, {"a": 2}]'),
        ('{"a": "a } brace in a string"} tail', '{"a": "a } brace in a string"}'),
        ("no json at all", None),
        ("", None),
        ("{not really json}", None),
    ],
)
def test_extract_json(text, expected):
    assert extract_json(text) == expected


def test_needs_fix_ignores_empty_and_valid():
    assert needs_fix(_VALID) is False
    assert needs_fix(None) is False
    assert needs_fix("") is False
    assert needs_fix("   ") is False
    assert needs_fix(_FENCED) is True


def test_fixed_value_returns_none_for_valid_json():
    assert fixed_value(_VALID) is None
    assert fixed_value(_PROSE) is None


# ----------------------------------------------------------------- the command


def test_fix_without_apply_writes_nothing(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME
    before = _responses(db_path)

    result = _fix("--ensure-json-format")

    assert result.exit_code == 0, result.output
    assert _responses(db_path) == before
    assert DATAFIX_TABLE not in try_table_names(db_path)
    assert "--apply" in result.output


def test_fix_without_apply_writes_nothing_with_many_previews(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME
    before = _responses(db_path)

    result = _fix("--ensure-json-format", "--test", "10")

    assert result.exit_code == 0, result.output
    assert _responses(db_path) == before


def test_test_zero_prints_no_preview_but_still_reports(experiment_with_results):
    result = _fix("--ensure-json-format", "--test", "0")

    assert result.exit_code == 0, result.output
    assert "Original:" not in result.output
    assert "Rows checked:" in result.output


def test_apply_shows_no_examples(experiment_with_results):
    result = _fix("--ensure-json-format", "--apply", "--test", "5")

    assert result.exit_code == 0, result.output
    assert "Original:" not in result.output
    assert "Fixed:" not in result.output
    assert "Fixing 3 row(s)" in result.output


def test_apply_spinner_counts_the_fixed_rows(experiment_with_results, monkeypatch):
    """The spinner text is refreshed once per fixed row, counting up to the total."""
    from llmexer.commands import experiment as experiment_module

    seen = []
    real_status = experiment_module.console.status

    def recording_status(text, **kwargs):
        seen.append(text)
        status = real_status(text, **kwargs)
        real_update = status.update

        def update(new_text, **update_kwargs):
            seen.append(new_text)
            return real_update(new_text, **update_kwargs)

        status.update = update
        return status

    monkeypatch.setattr(experiment_module.console, "status", recording_status)

    result = _fix("--ensure-json-format", "--apply")

    assert result.exit_code == 0, result.output
    assert seen == [
        "[bold blue] Applying fixes ... [/bold blue] "
        f"Repairable: [bold yellow] 3 [/bold yellow] Fixed: [bold green] {done} [/bold green]"
        for done in (0, 1, 2, 3)
    ]


def test_apply_repairs_every_broken_answer(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME

    result = _fix("--ensure-json-format", "--apply")

    assert result.exit_code == 0, result.output
    responses = _responses(db_path)

    assert json.loads(responses[2]) == {"relevant": False, "reason": "Off topic"}
    assert json.loads(responses[3]) == {"relevant": True, "score": 7}
    assert json.loads(responses[4]) == {"relevant": True, "score": 3}

    # Valid, prose and unrun answers are left exactly as they were.
    assert responses[1] == _VALID
    assert responses[5] == _PROSE
    assert responses[6] is None


def test_apply_logs_every_change(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME

    _fix("--ensure-json-format", "--apply")
    logs = _logs(db_path)

    assert [log["row_id"] for log in logs] == [2, 3, 4]
    assert [log["id"] for log in logs] == [1, 2, 3]
    for log in logs:
        assert log["table_name"] == "experiment_ollama"
        assert log["params_code"] == "llama3.3:latest_ollama"
        assert log["profile_name"] == "ollama-default"
        assert log["code"].endswith("_llama3.3:latest_ollama-default")
        assert log["old_value"] != log["new_value"]
        assert log["created_at"]


def test_apply_is_a_no_op_the_second_time(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME

    _fix("--ensure-json-format", "--apply")
    after_first = _responses(db_path)

    result = _fix("--ensure-json-format", "--apply")

    assert result.exit_code == 0, result.output
    assert _responses(db_path) == after_first
    assert len(_logs(db_path)) == 3
    assert "Nothing to fix" in result.output


def test_prose_only_row_is_reported_not_fixed(experiment_with_results):
    result = _fix("--ensure-json-format", "--apply")

    assert result.exit_code == 0, result.output
    assert "without recoverable JSON: " in result.output
    assert _logs(experiment_with_results / _DB_NAME)


def test_dry_run_with_apply_writes_nothing(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME
    before = _responses(db_path)

    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "fix", "--pid", PID, "--file", _DB_NAME, "--ensure-json-format", "--apply"],
    )

    assert result.exit_code == 0, result.output
    assert _responses(db_path) == before
    assert DATAFIX_TABLE not in try_table_names(db_path)
    assert "Dry run" in result.output


def test_preview_prints_the_row_identity(experiment_with_results):
    result = _fix("--ensure-json-format", "--test", "1")

    assert result.exit_code == 0, result.output
    for label in ("id:", "table_name:", "row_id:", "code:", "params_code:", "profile_name:"):
        assert label in result.output
    assert "Original:" in result.output
    assert "Fixed:" in result.output


def test_negative_test_is_rejected(experiment_with_results):
    result = _fix("--test", "-1")

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


# ----------------------------------------------------------- rename_json_field


def test_rename_keeps_the_field_in_place():
    text, skipped = rename_json_field('{"a": 1, "relevant": "yes", "z": 2}', "relevant", "isRelevant")

    assert skipped == ""
    assert list(json.loads(text)) == ["a", "isRelevant", "z"]
    assert json.loads(text)["isRelevant"] == "yes"


def test_rename_leaves_a_nested_field_of_the_same_name_alone():
    """Only the outermost object is touched."""
    source = '{"relevant": "yes", "scores": {"relevant": 3}}'

    text, skipped = rename_json_field(source, "relevant", "isRelevant")

    assert skipped == ""
    parsed = json.loads(text)
    assert "isRelevant" in parsed
    assert parsed["scores"] == {"relevant": 3}


@pytest.mark.parametrize(
    "source, reason",
    [
        ('{"a": 1}', SKIP_NO_SUCH_FIELD),
        ('{"relevant": 1, "isRelevant": 2}', SKIP_NAME_TAKEN),
        ('[{"relevant": 1}]', SKIP_NOT_AN_OBJECT),
        ('"relevant"', SKIP_NOT_AN_OBJECT),
        ("no json at all", SKIP_NOT_AN_OBJECT),
        (None, SKIP_NOT_AN_OBJECT),
    ],
)
def test_rename_skips_and_says_why(source, reason):
    text, skipped = rename_json_field(source, "relevant", "isRelevant")

    assert text is None
    assert skipped == reason


# -------------------------------------------------------- --rename-json-field


@pytest.fixture()
def experiment_with_fields(projects_dir):
    """Rows carrying the field to rename, one of them behind a fence."""
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {
            "ollama": [
                _ran_row(1, "01", '{"paperTitle": "A", "relevant": "yes"}'),
                _ran_row(2, "02", _FENCED_FIELD),
                _ran_row(3, "03", '{"paperTitle": "C"}'),
                _ran_row(4, "04", _PROSE),
            ]
        },
    )
    return exp_subdir


def test_rename_rewrites_only_the_rows_that_have_the_field(experiment_with_fields):
    db_path = experiment_with_fields / _DB_NAME

    result = _fix("--rename-json-field", "relevant", "isRelevant", "--apply")

    assert result.exit_code == 0, result.output
    responses = _responses(db_path)

    assert json.loads(responses[1]) == {"paperTitle": "A", "isRelevant": "yes"}
    # Row 2 is still fenced, so it is not JSON to rename yet; 3 has no such
    # field and 4 is prose. All three are left exactly as they were.
    assert responses[2] == _FENCED_FIELD
    assert responses[3] == '{"paperTitle": "C"}'
    assert responses[4] == _PROSE


def test_a_fenced_row_takes_a_repair_run_then_a_rename_run(experiment_with_fields):
    """One kind of fix per run: repair first, because a fence is not JSON."""
    db_path = experiment_with_fields / _DB_NAME

    _fix("--rename-json-field", "relevant", "isRelevant", "--apply")
    assert _responses(db_path)[2] == _FENCED_FIELD

    _fix("--ensure-json-format", "--apply")
    assert json.loads(_responses(db_path)[2]) == {"paperTitle": "B", "relevant": "no"}

    _fix("--rename-json-field", "relevant", "isRelevant", "--apply")
    assert json.loads(_responses(db_path)[2]) == {"paperTitle": "B", "isRelevant": "no"}

    # One log line per fix, in the order the runs happened.
    logs = _logs(db_path)
    assert [log["row_id"] for log in logs] == [1, 2, 2]
    assert [log["id"] for log in logs] == [1, 2, 3]
    assert logs[1]["old_value"].startswith("```json")


def test_rename_reports_what_it_changed_and_skipped(experiment_with_fields):
    result = _fix("--rename-json-field", "relevant", "isRelevant")

    assert result.exit_code == 0, result.output
    assert "fields renamed: 1" in result.output
    assert SKIP_NO_SUCH_FIELD in result.output
    # The fenced row and the prose row are both "not a JSON object" yet.
    assert SKIP_NOT_AN_OBJECT in result.output
    # A rename run says nothing about repairing.
    assert "repairable" not in result.output


def test_the_preview_names_the_fix_being_proposed(experiment_with_fields):
    renaming = _fix("--rename-json-field", "relevant", "isRelevant", "--test", "5")

    assert "change note:" in renaming.output
    assert "renamed field 'relevant' -> 'isRelevant'" in renaming.output
    assert "repaired JSON" not in renaming.output

    repairing = _fix("--ensure-json-format", "--test", "5")

    assert "repaired JSON" in repairing.output
    assert "renamed" not in repairing.output


def test_rename_without_apply_writes_nothing(experiment_with_fields):
    db_path = experiment_with_fields / _DB_NAME
    before = _responses(db_path)

    result = _fix("--rename-json-field", "relevant", "isRelevant", "--test", "5")

    assert result.exit_code == 0
    assert _responses(db_path) == before
    assert DATAFIX_TABLE not in try_table_names(db_path)


def test_renaming_twice_is_a_no_op_the_second_time(experiment_with_fields):
    db_path = experiment_with_fields / _DB_NAME

    _fix("--rename-json-field", "relevant", "isRelevant", "--apply")
    after_first = _responses(db_path)

    result = _fix("--rename-json-field", "relevant", "isRelevant", "--apply")

    assert result.exit_code == 0
    assert _responses(db_path) == after_first
    # Only the one row that was JSON with the field is ever renamed.
    assert len(_logs(db_path)) == 1


@pytest.mark.parametrize("names", [("a", "a"), ("", "b"), ("a", "  ")])
def test_bad_rename_names_are_rejected(experiment_with_fields, names):
    result = _fix("--rename-json-field", *names)

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)


# --------------------------------------------------------------- the two fixes


def test_naming_no_fix_is_rejected(experiment_with_results):
    """The command has to be told which of the two fixes to run."""
    db_path = experiment_with_results / _DB_NAME
    before = _responses(db_path)

    result = _fix()

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert _responses(db_path) == before


def test_naming_both_fixes_is_rejected(experiment_with_results):
    result = _fix("--ensure-json-format", "--rename-json-field", "a", "b", "--apply")

    assert result.exit_code != 0
    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert DATAFIX_TABLE not in try_table_names(experiment_with_results / _DB_NAME)


# ------------------------------------------------------ datafix_logs.change_note


def test_a_repair_run_notes_what_it_did(experiment_with_results):
    db_path = experiment_with_results / _DB_NAME

    _fix("--ensure-json-format", "--apply")

    assert {log["change_note"] for log in _logs(db_path)} == {"repaired JSON"}


def test_a_rename_run_notes_both_key_names(experiment_with_fields):
    db_path = experiment_with_fields / _DB_NAME

    _fix("--rename-json-field", "relevant", "isRelevant", "--apply")
    logs = _logs(db_path)

    assert len(logs) == 1
    assert logs[0]["change_note"] == "renamed field 'relevant' -> 'isRelevant'"


def test_each_run_keeps_its_own_note(experiment_with_fields):
    """Two runs over one row leave two rows saying which fix each one was."""
    db_path = experiment_with_fields / _DB_NAME

    _fix("--ensure-json-format", "--apply")
    _fix("--rename-json-field", "relevant", "isRelevant", "--apply")

    notes = [(log["row_id"], log["change_note"]) for log in _logs(db_path)]
    assert (2, "repaired JSON") in notes
    assert (2, "renamed field 'relevant' -> 'isRelevant'") in notes
