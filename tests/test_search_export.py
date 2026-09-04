"""Tests for the `search export` command."""

import os
import re
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from llmexer.base.search_export import (
    LONG_TEXT_THRESHOLD,
    _classify_column,
    _order_columns,
    _sanitize,
)
from llmexer.cli import app
from llmexer.commands.search import (
    _PAPER_CSV_COLUMNS,
    MERGED_FILTERED_SUFFIX,
    MERGED_RESULTS_SUFFIX,
)
from llmexer.exceptions import LLMExerException, UnexpectedCLIParamsException

runner = CliRunner()

PID = "20260409-test-export"
SEARCH_ID = "20260409-aabbccdd"
LONG_ABSTRACT = "Very long abstract sentence. " * 20


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def mock_no_dotenv(monkeypatch):
    """Mock load_dotenv to prevent loading from .env file."""
    mock_load = Mock(return_value=True)
    monkeypatch.setattr("llmexer.cli.load_dotenv", mock_load)
    return mock_load


def _row(**overrides):
    """Build a search result row with every column present."""
    row = {col: "" for col in _PAPER_CSV_COLUMNS}
    row["pdf_downloaded"] = False
    row.update(overrides)
    return row


def _write_csv(searches_path, filename, rows, columns=None):
    df = pd.DataFrame(rows, columns=columns or _PAPER_CSV_COLUMNS)
    df.to_csv(os.path.join(searches_path, filename), index=False, encoding="utf-8", sep=";")


@pytest.fixture()
def project_with_search(projects_dir, monkeypatch):
    """Create a project with a search YAML and a __results.csv."""
    searches_path = projects_dir / PID / "searches"
    os.makedirs(searches_path)
    monkeypatch.setenv("PROJECT_ID", PID)

    (searches_path / f"{SEARCH_ID}.yaml").write_text(
        yaml.dump({"query": "test query", "year": "2020-2025", "onlyOpenAccess": False}),
        encoding="utf-8",
    )

    _write_csv(
        str(searches_path),
        f"{SEARCH_ID}__results.csv",
        [
            _row(
                search_engine_internal_id="p1",
                year=2023,
                title="Downloaded Paper",
                authors="Alice; Bob",
                abstract=LONG_ABSTRACT,
                isOpenAccess=True,
                doi="10.1000/xyz 123",
                pdf_filename="2023_Alice_Downloaded_Paper.pdf",
                citationCount=42,
                pdf_downloaded=True,
            ),
            _row(
                search_engine_internal_id="p2",
                year=2021,
                title="Missing Paper",
                authors="Carol",
                abstract="Short abstract.",
                isOpenAccess=False,
                citationCount=7,
                pdf_downloaded=False,
            ),
        ],
    )

    return searches_path


def test_export_single_search(projects_dir, mock_no_dotenv, project_with_search):
    """`--file <id>.yaml` renders the results CSV as HTML with the same stem."""
    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", f"{SEARCH_ID}.yaml"])

    assert result.exit_code == 0, result.output
    html_path = project_with_search / f"{SEARCH_ID}__results.html"
    assert html_path.exists()

    html = html_path.read_text(encoding="utf-8")
    # One header cell per CSV column, minus the hidden internal id column.
    assert html.count('<th scope="col"') == len(_PAPER_CSV_COLUMNS) - 1
    assert "search_engine_internal_id" not in html
    # TRUE cells are green, FALSE cells muted.
    assert "text-bg-success" in html
    assert "text-bg-secondary" in html
    # Dark mode and Bootstrap are wired in.
    assert 'data-bs-theme="light"' in html
    assert "cdn.jsdelivr.net/npm/bootstrap@5" in html
    # The long abstract is fully present behind a more/less toggle.
    assert "toggle-more" in html
    assert LONG_ABSTRACT.strip() in html


def test_export_includes_filtered(projects_dir, mock_no_dotenv, project_with_search):
    """Both `__results.csv` and `__filtered.csv` are exported when present."""
    _write_csv(
        str(project_with_search),
        f"{SEARCH_ID}__filtered.csv",
        [_row(search_engine_internal_id="p1", year=2023, title="Downloaded Paper")],
    )

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID])

    assert result.exit_code == 0, result.output
    assert (project_with_search / f"{SEARCH_ID}__results.html").exists()
    assert (project_with_search / f"{SEARCH_ID}__filtered.html").exists()


def test_export_csv_file(projects_dir, mock_no_dotenv, project_with_search):
    """`--csv-file` exports one CSV directly, so merged files can be exported."""
    merged_name = f"{PID}{MERGED_RESULTS_SUFFIX}"
    _write_csv(
        str(project_with_search),
        merged_name,
        [_row(search_engine_internal_id="p1", title="Merged Paper")],
    )

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--csv-file", merged_name])

    assert result.exit_code == 0, result.output
    assert (project_with_search / f"{PID}{MERGED_RESULTS_SUFFIX[:-4]}.html").exists()


def test_export_file_rejects_csv(projects_dir, mock_no_dotenv, project_with_search):
    """`--file` takes a search ID/YAML only — a CSV name is not a search config."""
    result = runner.invoke(
        app,
        ["search", "export", "--pid", PID, "--file", f"{SEARCH_ID}__results.csv"],
    )

    assert isinstance(result.exception, LLMExerException)


def test_export_file_and_csv_file_are_exclusive(projects_dir, mock_no_dotenv, project_with_search):
    """Passing both `--file` and `--csv-file` is rejected."""
    result = runner.invoke(
        app,
        [
            "search",
            "export",
            "--pid",
            PID,
            "--file",
            SEARCH_ID,
            "--csv-file",
            f"{SEARCH_ID}__results.csv",
        ],
    )

    assert isinstance(result.exception, UnexpectedCLIParamsException)


def test_export_all_searches(projects_dir, mock_no_dotenv, project_with_search):
    """Without `--file` every search is exported, plus the merged CSVs."""
    _write_csv(
        str(project_with_search),
        f"{PID}{MERGED_FILTERED_SUFFIX}",
        [_row(search_engine_internal_id="p1", title="Merged Paper")],
    )

    result = runner.invoke(app, ["search", "export", "--pid", PID])

    assert result.exit_code == 0, result.output
    assert (project_with_search / f"{SEARCH_ID}__results.html").exists()
    assert (project_with_search / f"{PID}{MERGED_FILTERED_SUFFIX[:-4]}.html").exists()


def test_export_dry_run_writes_nothing(projects_dir, mock_no_dotenv, project_with_search):
    """`--dry-run` announces the target file but writes nothing."""
    result = runner.invoke(app, ["--dry-run", "search", "export", "--pid", PID, "--file", SEARCH_ID])

    assert result.exit_code == 0, result.output
    assert "Dry run" in result.output
    assert not (project_with_search / f"{SEARCH_ID}__results.html").exists()


def test_export_does_not_overwrite_without_rewrite(projects_dir, mock_no_dotenv, project_with_search):
    """An existing HTML file is kept unless `--rewrite` is passed."""
    html_path = project_with_search / f"{SEARCH_ID}__results.html"
    html_path.write_text("existing", encoding="utf-8")

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID])

    assert result.exit_code == 0, result.output
    assert "Warning" in result.output
    assert html_path.read_text(encoding="utf-8") == "existing"

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID, "--rewrite"])

    assert result.exit_code == 0, result.output
    assert html_path.read_text(encoding="utf-8") != "existing"


def test_export_strips_markup_from_values(projects_dir, mock_no_dotenv, project_with_search):
    """Markup is stripped out of values, never injected and never shown as tags."""
    _write_csv(
        str(project_with_search),
        f"{SEARCH_ID}__results.csv",
        [
            _row(
                search_engine_internal_id="p1",
                title="<script>alert(1)</script>",
                abstract="<p><strong>Intro:</strong> body</p>",
            )
        ],
    )

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID])

    assert result.exit_code == 0, result.output
    html = (project_with_search / f"{SEARCH_ID}__results.html").read_text(encoding="utf-8")
    assert ">alert(1)<" in html
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" not in html
    assert ">Intro: body<" in html


def test_export_missing_csv_raises(projects_dir, mock_no_dotenv, project_with_search):
    """Passing a CSV filename that does not exist raises."""
    result = runner.invoke(app, ["search", "export", "--pid", PID, "--csv-file", "nope.csv"])

    assert isinstance(result.exception, LLMExerException)


def test_export_without_results_prints_hint(projects_dir, mock_no_dotenv, monkeypatch):
    """A search with no result CSV prints the `search run` hint instead of failing."""
    searches_path = projects_dir / PID / "searches"
    os.makedirs(searches_path)
    monkeypatch.setenv("PROJECT_ID", PID)
    (searches_path / f"{SEARCH_ID}.yaml").write_text(
        yaml.dump({"query": "q", "year": "2020-2025", "onlyOpenAccess": False}),
        encoding="utf-8",
    )

    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID])

    assert result.exit_code == 0, result.output
    assert "search run" in result.output


@pytest.mark.parametrize(
    "values,expected",
    [
        (["True", "False", ""], "bool"),
        (["1", "0"], "bool"),
        (["2023", "1999"], "number"),
        (["12.5", ""], "number"),
        (["a" * (LONG_TEXT_THRESHOLD + 1), "short"], "long"),
        (["short", "also short"], "text"),
        (["", ""], "text"),
    ],
)
def test_classify_column(values, expected):
    """Column kinds drive badge, alignment and more/less rendering."""
    assert _classify_column(values) == expected


def _export_html(project_with_search):
    """Export the sample search and return the rendered HTML."""
    result = runner.invoke(app, ["search", "export", "--pid", PID, "--file", SEARCH_ID])
    assert result.exit_code == 0, result.output
    return (project_with_search / f"{SEARCH_ID}__results.html").read_text(encoding="utf-8")


def test_export_links_doi(projects_dir, mock_no_dotenv, project_with_search):
    """DOIs render as doi.org links opening in a new tab."""
    html = _export_html(project_with_search)

    assert 'href="https://doi.org/10.1000/xyz%20123"' in html
    assert 'target="_blank"' in html
    assert 'rel="noopener noreferrer"' in html


def test_export_keeps_doi_url_as_is(projects_dir, mock_no_dotenv, project_with_search):
    """A DOI already stored as a URL is linked verbatim, not double-prefixed."""
    _write_csv(
        str(project_with_search),
        f"{SEARCH_ID}__results.csv",
        [_row(search_engine_internal_id="p1", doi="https://doi.org/10.5/abc")],
    )

    html = _export_html(project_with_search)

    assert 'href="https://doi.org/10.5/abc"' in html
    assert "https://doi.org/https" not in html


def test_export_copy_buttons(projects_dir, mock_no_dotenv, project_with_search):
    """Only the copyable columns get a copy button, and the icon sprite is defined once."""
    html = _export_html(project_with_search)

    assert html.count('id="icon-copy"') == 1
    assert html.count('id="icon-check"') == 1

    # title, abstract, authors, doi and pdf_filename are filled on row 1; row 2 leaves
    # doi and pdf_filename empty, so it contributes three buttons.
    assert html.count("align-baseline copy-btn") == 8


def test_export_has_no_sticky_header(projects_dir, mock_no_dotenv, project_with_search):
    """The table must not scroll vertically or pin its header over the rows."""
    html = _export_html(project_with_search)

    assert "position: sticky" not in html
    assert "max-height" not in html


def test_export_renames_column_labels(projects_dir, mock_no_dotenv, project_with_search):
    """Verbose column headers are shown under shorter labels."""
    html = _export_html(project_with_search)

    for label in ("lang", "citations", "references", "OpenAccess"):
        assert f"\n          {label}<span" in html

    # The CSV column names survive as the sort/filter keys.
    assert 'data-col="language"' in html
    assert 'data-col="citationCount"' in html


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("<p><strong>Intro:</strong> body</p>", "Intro: body"),
        ("&lt;b&gt;bold&lt;/b&gt;", "bold"),
        ("a&amp;b", "a&b"),
        ("line\n\n\nbreak", "line break"),
        ("tab\tsep", "tab sep"),
        ("null\x00byte", "null byte"),
        ("  padded  ", "padded"),
        ("plain text", "plain text"),
    ],
)
def test_sanitize(raw, expected):
    """Values are stripped of markup, control characters and whitespace runs."""
    assert _sanitize(raw) == expected


def test_export_column_order(projects_dir, mock_no_dotenv, project_with_search):
    """`doi` follows `abstract` and `pdf_downloaded` follows `isOpenAccess`."""
    html = _export_html(project_with_search)

    keys = re.findall(r'<th scope="col" data-col="([^"]+)"', html)

    assert keys[keys.index("abstract") + 1] == "doi"
    assert keys[keys.index("isOpenAccess") + 1] == "pdf_downloaded"


def test_order_columns_ignores_missing_anchors():
    """A column whose anchor is absent keeps its original position."""
    assert _order_columns(["year", "doi", "title"]) == ["year", "doi", "title"]
    assert _order_columns(["abstract", "year", "doi"]) == ["abstract", "doi", "year"]


def test_export_row_counters(projects_dir, mock_no_dotenv, project_with_search):
    """The header shows a total badge and a live 'showing' badge with a percentage."""
    html = _export_html(project_with_search)

    assert 'Total: <span class="count-value">2</span>' in html
    assert 'id="shown-count"' in html
    assert 'id="shown-percent"' in html
