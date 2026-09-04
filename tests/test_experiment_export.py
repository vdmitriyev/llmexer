"""Tests for the `experiment export` command."""

import os
import re
import shlex
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from llmexer.base.experiment_export import (
    CODE_PREVIEW_LENGTH,
    EXPORT_COLUMNS,
    TRY_PREVIEW_LENGTH,
    build_try_command,
    format_response_text,
    format_seconds,
    format_timestamp,
    row_tokens,
)
from llmexer.cli import app
from llmexer.exceptions import LLMExerException
from tests.db_helpers import LITELLM_ROW, OLLAMA_ROW, find_db, seed_db

runner = CliRunner()

PID = "export-test-exp"
_DB_NAME = "experiment_20240101_01.db"
_HTML_NAME = "experiment_20240101_01.html"

_JSON_RESPONSE = '{"relevant": true, "reason": "Discusses attention", "score": 9}'
_FENCED_RESPONSE = '```json\n{"relevant": false, "reason": "Off topic"}\n```'
_PROSE_RESPONSE = "This paper is broadly relevant but does not answer the question directly."


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


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
            "total_tokens": 142,
            "usage_tokens": 142,
            # Deliberately messy: a raw float and a microsecond timestamp, as
            # `run` actually stores them.
            "elapsed_seconds": 2.4700000000000002,
            "timestamp": "2026-09-04T10:00:00.123456+00:00",
        }
    )
    row.update(overrides)
    return row


@pytest.fixture()
def experiment_with_results(projects_dir):
    """A database with three finished ollama rows and one unrun litellm row."""
    exp_subdir = projects_dir / PID / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {
            "ollama": [
                _ran_row(1, "01", _JSON_RESPONSE),
                # total_tokens missing -> the export falls back to usage_tokens.
                _ran_row(2, "02", _FENCED_RESPONSE, total_tokens=None, usage_tokens=88),
                _ran_row(3, "03", _PROSE_RESPONSE),
            ],
            # Never run: no status, no response.
            "litellm": [dict(LITELLM_ROW, ID=4)],
        },
    )
    return exp_subdir


def _export(pid=PID, *options):
    """Invoke `experiment export` on the seeded database."""
    return runner.invoke(app, ["experiment", "export", "--pid", pid, "--file", _DB_NAME, *options])


def _export_html(exp_subdir):
    """Export the sample database and return the rendered HTML."""
    result = _export()
    assert result.exit_code == 0, result.output
    return (exp_subdir / _HTML_NAME).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Happy path / structure
# ---------------------------------------------------------------------------


def test_export_writes_html_next_to_the_database(experiment_with_results):
    """The page lands beside the .db with the same stem."""
    result = _export()

    assert result.exit_code == 0, result.output
    assert (experiment_with_results / _HTML_NAME).exists()
    assert "4 rows" in result.output


def test_export_renders_the_fields_in_order(experiment_with_results):
    """Exactly the specified columns are exported, in the specified order."""
    html = _export_html(experiment_with_results)

    keys = re.findall(r'<th scope="col" data-col="([^"]+)"', html)
    assert keys == [
        "provider",
        "model",
        "profile",
        "code",
        "response_text",
        "tokens",
        "status",
        "seconds",
        "timestamp",
        "try",
    ]
    assert keys == [key for key, _source, _kind in EXPORT_COLUMNS]


def test_export_carries_the_page_furniture(experiment_with_results):
    """Dark mode, Bootstrap, the icon sprite and the row counters are wired in."""
    html = _export_html(experiment_with_results)

    assert 'data-bs-theme="light"' in html
    assert "cdn.jsdelivr.net/npm/bootstrap@5" in html
    assert html.count('id="icon-copy"') == 1
    assert 'Total: <span class="count-value">4</span>' in html
    assert 'id="shown-count"' in html
    assert 'id="shown-percent"' in html
    # One filter input per column, in the second header row.
    assert html.count('data-filter-col="') == len(EXPORT_COLUMNS)


def test_export_includes_unrun_rows(experiment_with_results):
    """A row that was never run is exported with an empty response and status."""
    html = _export_html(experiment_with_results)

    # The litellm row exists only because unrun rows are kept.
    assert "litellm" in html
    assert "gpt-oss:120b" in html
    body = html[html.index("<tbody>") : html.index("</tbody>")]
    assert len(body.split("<tr>")) - 1 == 4


# ---------------------------------------------------------------------------
# `code` shortening
# ---------------------------------------------------------------------------


def test_export_shortens_the_code_but_keeps_the_full_value(experiment_with_results):
    """`code` shows 5 characters; the whole value stays in the page to copy."""
    html = _export_html(experiment_with_results)

    assert f'<span class="cell-short cell-value">{"D01_p"}&hellip;</span>' in html
    assert len("D01_p") == CODE_PREVIEW_LENGTH
    # The full code is present, collapsed, and is what the copy button reads.
    assert '<span class="cell-full d-none">D01_prompt01_llama3.3:latest_ollama-default</span>' in html
    assert "toggle-more" in html


def test_export_collapsed_cells_start_hidden(experiment_with_results):
    """Every `.cell-full` carries `d-none`, so nothing renders twice."""
    html = _export_html(experiment_with_results)
    body = html[html.index("<tbody>") : html.index("</tbody>")]

    assert re.search(r'class="cell-full(?![^"]*d-none)[^"]*"', body) is None


# ---------------------------------------------------------------------------
# response_text -> JSON
# ---------------------------------------------------------------------------


def test_export_pretty_prints_json_responses(experiment_with_results):
    """A JSON response is indented inside a <pre>, not shown as one flat line."""
    html = _export_html(experiment_with_results)

    assert 'class="cell-value cell-json mb-0"' in html
    # Indented keys, i.e. json.dumps(..., indent=2) actually ran.
    assert "  &#34;relevant&#34;: true," in html
    assert "  &#34;score&#34;: 9" in html


def test_export_parses_a_fenced_json_response(experiment_with_results):
    """```json fences are peeled off before parsing, as models emit them."""
    html = _export_html(experiment_with_results)

    assert "  &#34;reason&#34;: &#34;Off topic&#34;" in html
    # The fence itself never reaches the page.
    assert "```" not in html


def test_export_keeps_non_json_responses_as_plain_text(experiment_with_results):
    """Prose is not an error: it is rendered verbatim, outside a <pre>."""
    html = _export_html(experiment_with_results)

    assert f'<span class="cell-value">{_PROSE_RESPONSE}</span>' in html


# ---------------------------------------------------------------------------
# tokens
# ---------------------------------------------------------------------------


def test_export_falls_back_to_usage_tokens(experiment_with_results):
    """A row with no total_tokens reports usage_tokens instead."""
    html = _export_html(experiment_with_results)
    body = html[html.index("<tbody>") : html.index("</tbody>")]

    second_row = body.split("<tr>")[2]
    tokens_cell = second_row.split("<td ")[6]
    assert '<span class="cell-value">88</span>' in tokens_cell


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"total_tokens": 142, "usage_tokens": 142}, 142),
        ({"total_tokens": None, "usage_tokens": 88}, 88),
        ({"total_tokens": 0, "usage_tokens": 12}, 12),
        ({"total_tokens": None, "usage_tokens": None}, 0),
        ({}, 0),
    ],
)
def test_row_tokens(row, expected):
    """total_tokens wins, usage_tokens is the fallback, absent means zero."""
    assert row_tokens(row) == expected


# ---------------------------------------------------------------------------
# format_response_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected_json",
    [
        ('{"a": 1}', True),
        ('```json\n{"a": 1}\n```', True),
        ("```\n[1, 2, 3]\n```", True),
        ("[1, 2]", True),
        ("not json at all", False),
        ('{"a": 1', False),
        ("", False),
        ("   ", False),
        # Valid JSON, but a bare scalar reads better as the text it already is.
        ('"just a string"', False),
        ("42", False),
    ],
)
def test_format_response_text_detects_json(text, expected_json):
    _rendered, is_json = format_response_text(text)
    assert is_json is expected_json


def test_format_response_text_indents_json():
    rendered, is_json = format_response_text('{"b": 2, "a": 1}')

    assert is_json is True
    assert rendered == '{\n  "b": 2,\n  "a": 1\n}'


def test_format_response_text_returns_invalid_input_unchanged():
    """A response that will not parse comes back exactly as it went in."""
    rendered, is_json = format_response_text(_PROSE_RESPONSE)

    assert is_json is False
    assert rendered == _PROSE_RESPONSE


def test_format_response_text_keeps_non_ascii():
    """ensure_ascii=False, so accented text is not mangled into escapes."""
    rendered, is_json = format_response_text('{"autor": "Müller"}')

    assert is_json is True
    assert "Müller" in rendered


# ---------------------------------------------------------------------------
# seconds / timestamp formatting
# ---------------------------------------------------------------------------


def _cell_of(html, column):
    """The rendered <td> of one column in the first body row."""
    body = html[html.index("<tbody>") : html.index("</tbody>")]
    keys = re.findall(r'<th scope="col" data-col="([^"]+)"', html)
    return body.split("<tr>")[1].split("<td ")[keys.index(column) + 1]


def test_export_rounds_seconds_to_one_decimal(experiment_with_results):
    """A raw float is shown as one decimal, not as 2.4700000000000002."""
    html = _export_html(experiment_with_results)

    assert '<span class="cell-value">2.5</span>' in _cell_of(html, "seconds")
    assert "2.4700000000000002" not in html


def test_export_trims_the_timestamp_to_seconds(experiment_with_results):
    """Microseconds are dropped; the UTC offset is kept."""
    html = _export_html(experiment_with_results)

    assert '<span class="cell-value">2026-09-04T10:00:00+00:00</span>' in _cell_of(html, "timestamp")
    assert ".123456" not in html


def test_export_leaves_unrun_seconds_and_timestamp_empty(experiment_with_results):
    """An unrun row has neither, and neither is rendered as '0.0' or a fake date."""
    html = _export_html(experiment_with_results)
    body = html[html.index("<tbody>") : html.index("</tbody>")]
    keys = re.findall(r'<th scope="col" data-col="([^"]+)"', html)

    unrun = body.split("<tr>")[4]
    for column in ("seconds", "timestamp"):
        cell = unrun.split("<td ")[keys.index(column) + 1]
        assert '<span class="cell-value"></span>' in cell


@pytest.mark.parametrize(
    "value, expected",
    [
        (2.4700000000000002, "2.5"),
        (0, "0.0"),
        (12, "12.0"),
        ("3.14159", "3.1"),
        (None, ""),
        ("", ""),
        ("not a number", "not a number"),
    ],
)
def test_format_seconds(value, expected):
    assert format_seconds(value) == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-09-04T10:00:00.123456+00:00", "2026-09-04T10:00:00+00:00"),
        ("2026-09-04T10:00:00+00:00", "2026-09-04T10:00:00+00:00"),
        ("2026-09-04T10:00:00", "2026-09-04T10:00:00"),
        (None, ""),
        ("", ""),
        # Not parseable: passed through rather than dropped or raised on.
        ("whenever", "whenever"),
    ],
)
def test_format_timestamp(value, expected):
    assert format_timestamp(value) == expected


# ---------------------------------------------------------------------------
# the `try` command column
# ---------------------------------------------------------------------------


def test_export_builds_a_try_command_per_row(experiment_with_results):
    """Each row carries a full `experiment try` invocation for its combination."""
    html = _export_html(experiment_with_results)
    cell = _cell_of(html, "try")

    full = re.search(r'class="cell-full d-none">(.*?)</span>', cell).group(1)
    assert full == (
        f"llmexer experiment try --pid {PID} --file {_DB_NAME} "
        "--data-id D01 --prompt prompt01 --profile ollama-default "
        "--model llama3.3:latest --provider ollama"
    )


def test_export_try_command_is_collapsed_and_copyable(experiment_with_results):
    """Only a short preview is shown; the whole command is there to copy."""
    html = _export_html(experiment_with_results)
    cell = _cell_of(html, "try")

    short = re.search(r'class="cell-short cell-value">(.*?)&hellip;</span>', cell).group(1)
    assert len(short) == TRY_PREVIEW_LENGTH
    assert "cell-full d-none" in cell
    assert "toggle-more" in cell
    assert "copy-btn" in cell


def test_export_try_command_matches_each_row(experiment_with_results):
    """The command names that row's own data ID and provider, not the first row's."""
    html = _export_html(experiment_with_results)
    commands = re.findall(r'class="cell-full d-none">(llmexer experiment try [^<]*)</span>', html)

    assert len(commands) == 4
    assert "--data-id D02" in commands[1]
    # The unrun litellm row still gets a runnable command.
    assert "--provider litellm" in commands[3]
    assert "--model gpt-oss:120b" in commands[3]


def test_export_try_cell_is_empty_for_an_unparseable_code(projects_dir):
    """A code that is not DATAID_PROMPTID_MODEL_PROFILE yields no command."""
    exp_subdir = projects_dir / "odd-code-exp" / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {"ollama": [_ran_row(1, "01", _PROSE_RESPONSE, code="nounderscores")]},
    )

    result = runner.invoke(app, ["experiment", "export", "--pid", "odd-code-exp", "--file", _DB_NAME])

    assert result.exit_code == 0, result.output
    html = (exp_subdir / _HTML_NAME).read_text(encoding="utf-8")
    assert "llmexer experiment try" not in html
    assert '<span class="cell-value"></span>' in _cell_of(html, "try")


def test_build_try_command_quotes_values_that_need_it():
    """shlex.quote protects a value with a space; ordinary names stay bare."""
    row = {
        "code": "D01_prompt01_my model_ollama-default",
        "model_name": "my model",
        "profile_name": "ollama-default",
        "_provider": "ollama",
    }

    command = build_try_command(row, "proj", "experiment_1.db")

    assert "--model 'my model'" in command
    assert "--provider ollama" in command
    assert shlex.split(command)[:3] == ["llmexer", "experiment", "try"]


def test_build_try_command_returns_empty_without_a_usable_code():
    """No command is better than one that would run the wrong combination."""
    row = {"code": "D01_prompt01_other_profile", "model_name": "m", "profile_name": "p", "_provider": "ollama"}

    assert build_try_command(row, "proj", "db.db") == ""


# ---------------------------------------------------------------------------
# `try` command round trip: the exported command really runs
# ---------------------------------------------------------------------------

_LLM_PARAMS_HEADER = (
    "provider;model_name;profile_name;temperature;top_p;max_tokens;"
    "ollama_context_window;ollama_repeat_penalty;vllm_min_p;vllm_best_of;openai_seed;gemini_thinking_level\n"
)
_OLLAMA_PARAMS_ROW = "ollama;llama3.3:latest;ollama-default;0.7;1.0;512;4096;1.1;;;;\n"
_MODELS_HEADER = "provider;model_name;profile_name;notes\n"
_OLLAMA_MODEL_ROW = "ollama;llama3.3:latest;ollama-default;local model\n"


@pytest.fixture()
def mock_ollama(monkeypatch):
    """Replace OllamaProvider with a fake returning a canned response."""
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", usage_tokens=42)

    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)
    return FakeOllamaProvider


@pytest.fixture()
def generated_experiment(projects_dir):
    """A real project put through `generate`, so `try` can resolve its names."""
    pid = "export-roundtrip"
    exp_subdir = projects_dir / pid / "experiment"
    prompts_dir = exp_subdir / "prompts"
    os.makedirs(prompts_dir)

    (exp_subdir / "llms-for-experiment.csv").write_text(_MODELS_HEADER + _OLLAMA_MODEL_ROW, encoding="utf-8")
    (exp_subdir / "data.csv").write_text(
        "ID;Title;Abstract\nD01;Sample Paper Title One;This is the abstract of the first sample paper.\n",
        encoding="utf-8",
    )
    (exp_subdir / "mapping.csv").write_text("data_id;prompt_id\nD01;prompt01\n", encoding="utf-8")
    (prompts_dir / "prompt01.txt").write_text("Title: {{title}}. Abstract: {{abstract}}.", encoding="utf-8")
    (exp_subdir / "llm-params.csv").write_text(_LLM_PARAMS_HEADER + _OLLAMA_PARAMS_ROW, encoding="utf-8")

    result = runner.invoke(app, ["experiment", "generate", "--pid", pid])
    assert result.exit_code == 0, result.output

    return pid, exp_subdir


def test_exported_try_command_actually_runs(generated_experiment, mock_ollama):
    """The command lifted out of the HTML runs as-is and records a try.

    This is what makes the column worth having: the string is not merely
    plausible, it is a working invocation of `experiment try`.
    """
    pid, exp_subdir = generated_experiment
    from tests.db_helpers import read_try_rows

    assert runner.invoke(app, ["experiment", "export", "--pid", pid]).exit_code == 0
    html = next(exp_subdir.glob("experiment_*.html")).read_text(encoding="utf-8")

    command = re.search(r'class="cell-full d-none">(llmexer experiment try [^<]*)</span>', html).group(1)
    argv = shlex.split(command)
    assert argv[0] == "llmexer"

    # Everything after the executable is what the CLI itself receives.
    result = runner.invoke(app, argv[1:])

    assert result.exit_code == 0, result.output
    assert "mocked response" in result.output
    rows = read_try_rows(str(find_db(exp_subdir)), "ollama")
    assert len(rows) == 1
    assert rows[0]["response_text"] == "mocked response"


# ---------------------------------------------------------------------------
# CLI behaviour: dry run, rewrite, errors
# ---------------------------------------------------------------------------


def test_export_dry_run_writes_nothing(experiment_with_results):
    """`--dry-run` announces the target file but writes nothing."""
    result = runner.invoke(
        app,
        ["--dry-run", "experiment", "export", "--pid", PID, "--file", _DB_NAME],
    )

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (experiment_with_results / _HTML_NAME).exists()


def test_export_does_not_overwrite_without_rewrite(experiment_with_results):
    """An existing HTML file is kept unless `--rewrite` is passed."""
    html_path = experiment_with_results / _HTML_NAME
    html_path.write_text("existing", encoding="utf-8")

    result = _export()

    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert html_path.read_text(encoding="utf-8") == "existing"

    result = _export(PID, "--rewrite")

    assert result.exit_code == 0, result.output
    assert html_path.read_text(encoding="utf-8") != "existing"


def test_export_without_database_raises(projects_dir, mock_no_dotenv):
    """A project with no generated database points the user at `generate`."""
    os.makedirs(projects_dir / "empty-exp" / "experiment")

    result = runner.invoke(app, ["experiment", "export", "--pid", "empty-exp"])

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)
    assert "generate" in str(result.exception).lower()


def test_export_missing_file_raises(experiment_with_results):
    """An explicit --file that does not exist is an error, not an empty page."""
    result = _export(PID, "--file", "nope.db")

    assert result.exit_code != 0
    assert isinstance(result.exception, LLMExerException)


def test_export_defaults_to_the_newest_database(experiment_with_results):
    """With no --file the newest experiment_*.db is exported."""
    seed_db(experiment_with_results / "experiment_20240101_02.db", {"ollama": [dict(OLLAMA_ROW)]})

    result = runner.invoke(app, ["experiment", "export", "--pid", PID])

    assert result.exit_code == 0, result.output
    assert (experiment_with_results / "experiment_20240101_02.html").exists()
    assert not (experiment_with_results / _HTML_NAME).exists()


# ---------------------------------------------------------------------------
# Escaping
# ---------------------------------------------------------------------------


def test_export_does_not_emit_raw_markup_from_a_response(projects_dir):
    """A response carrying markup is stripped and escaped, never rendered live."""
    exp_subdir = projects_dir / "escape-exp" / "experiment"
    os.makedirs(exp_subdir)
    seed_db(
        exp_subdir / _DB_NAME,
        {"ollama": [_ran_row(1, "01", "<script>alert(1)</script> done")]},
    )

    result = runner.invoke(app, ["experiment", "export", "--pid", "escape-exp", "--file", _DB_NAME])

    assert result.exit_code == 0, result.output
    html = (exp_subdir / _HTML_NAME).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "alert(1)" in html
