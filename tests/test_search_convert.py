"""Tests for `search convert` and the per-engine conversions it writes into `queries:`."""

import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

import llmexer.base.search_arxiv as search_arxiv
import llmexer.base.search_core as search_core
import llmexer.base.search_europepmc as search_europepmc
import llmexer.constants as constants
from llmexer.base.search_engines import CONVERTERS, ENGINES, convert_query
from llmexer.base.search_query import to_pubmed_query
from llmexer.cli import app
from llmexer.exceptions import (
    SearchQueryNotationException,
    UnexpectedCLIParamsException,
)

runner = CliRunner()
PID = "convert-exp"
QUERY = '("air quality" | smog) + monitoring -indoor'


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))
    monkeypatch.setenv("PROJECT_ID", PID)
    monkeypatch.setenv("COLUMNS", "250")
    os.makedirs(tmp_path / PID / "searches")
    return tmp_path / PID


def _search_file(project, text, name="s1.yaml"):
    path = project / "searches" / name
    path.write_text(text, encoding="utf-8")
    return path


def _convert(*options):
    return runner.invoke(app, ["search", "convert", *options])


# ---------------------------------------------------------------------------
# Conversions
# ---------------------------------------------------------------------------


def test_every_engine_has_a_converter():
    assert set(CONVERTERS) == set(ENGINES)


def test_to_pubmed_query_tags_every_term():
    assert to_pubmed_query(QUERY) == '("air quality"[tiab] OR smog[tiab]) AND monitoring[tiab] NOT indoor[tiab]'
    with pytest.raises(SearchQueryNotationException, match="PubMed"):
        to_pubmed_query("NOT a")


@pytest.mark.parametrize(
    "key, build, args",
    [
        ("core", search_core.build_core_query, ("2020-2025",)),
        ("europepmc", search_europepmc.build_europepmc_query, ("2020-2025", True)),
        ("arxiv", search_arxiv.build_arxiv_query, ("2020-2025",)),
    ],
)
def test_converted_own_string_sends_the_same_query(key, build, args):
    """A converted string written to `queries:` must not change what the engine is asked.

    Compared without brackets: an own string gets one extra pair around it.
    """

    automatic = build(QUERY, *args)
    own = build("ignored", *args, convert_query(QUERY, key))

    assert own.replace("(", "").replace(")", "") == automatic.replace("(", "").replace(")", "")


# ---------------------------------------------------------------------------
# search convert --query
# ---------------------------------------------------------------------------


def test_convert_query_prints_every_engine(project):
    result = _convert("--query", QUERY)

    assert result.exit_code == 0, result.output
    for key in ENGINES:
        assert key in result.output
    assert "monitoring[tiab]" in result.output  # brackets survive the console
    assert "ANDNOT" in result.output
    assert not any((project / "searches").iterdir())


def test_convert_query_shows_an_engine_it_cannot_serve(project):
    result = _convert("--query", "NOT llm")

    assert result.exit_code == 0, result.output
    assert "arXiv can only exclude terms" in result.output


@pytest.mark.parametrize("options", [[], ["--query", "a", "--file", "s1.yaml"]])
def test_convert_needs_exactly_one_input(project, options):
    result = _convert(*options)

    assert isinstance(result.exception, UnexpectedCLIParamsException)


# ---------------------------------------------------------------------------
# search convert --file
# ---------------------------------------------------------------------------


def test_convert_file_writes_queries_and_keeps_own_strings(project):
    path = _search_file(
        project,
        f"query: '{QUERY}'\nyear: 2020-2025\nonlyOpenAccess: false\nengines: [semantic_scholar, pubmed, core]\n"
        "queries:\n  pubmed: 'asthma[mh]'\n  arxiv: 'ti:x'\n",
    )

    result = _convert("--file", "s1.yaml")

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert list(saved) == ["query", "year", "onlyOpenAccess", "engines", "queries"]
    assert saved["query"] == QUERY
    assert saved["queries"] == {
        "semantic_scholar": convert_query(QUERY, "semantic_scholar"),
        "pubmed": "asthma[mh]",
        "core": convert_query(QUERY, "core"),
        "arxiv": "ti:x",  # an engine the search does not use: left as it is
    }
    assert "kept: own string already in queries:" in result.output
    assert path.read_text(encoding="utf-8").startswith("query: '")


def test_convert_file_then_run_uses_the_written_strings(project, monkeypatch):
    """The written `queries:` reach the engines as own strings."""

    import dataclasses

    received = {}

    def fake_runner(*args, custom_query=None, key=None):
        received[key] = custom_query
        return []

    for key in ("europepmc", "arxiv"):
        monkeypatch.setitem(
            ENGINES,
            key,
            dataclasses.replace(ENGINES[key], runner=lambda *a, key=key, **k: fake_runner(*a, key=key, **k)),
        )
    _search_file(project, f"query: '{QUERY}'\nengines: [europepmc, arxiv]\n")

    assert _convert("--file", "s1.yaml").exit_code == 0
    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert received == {key: convert_query(QUERY, key) for key in ("europepmc", "arxiv")}


def test_convert_file_dry_run_writes_nothing(project):
    path = _search_file(project, f"query: '{QUERY}'\nengines: [core]\n")
    before = path.read_text(encoding="utf-8")

    result = runner.invoke(app, ["--dry-run", "search", "convert", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert "would update" in result.output
    assert path.read_text(encoding="utf-8") == before


def test_convert_file_leaves_out_an_engine_it_cannot_serve(project):
    path = _search_file(project, "query: 'NOT llm'\nengines: [core, arxiv]\n")

    result = _convert("--file", "s1.yaml")

    assert result.exit_code == 0, result.output
    assert yaml.safe_load(path.read_text(encoding="utf-8"))["queries"] == {"core": convert_query("NOT llm", "core")}
    assert "not written" in result.output


def test_convert_file_rejects_an_unknown_engine_in_queries(project):
    _search_file(project, f"query: '{QUERY}'\nengines: [core]\nqueries:\n  scopus: 'x'\n")

    result = _convert("--file", "s1.yaml")

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "scopus" in str(result.exception)
