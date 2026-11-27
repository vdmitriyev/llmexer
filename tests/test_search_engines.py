"""Tests for choosing the search engines of `search run` and running them in order."""

import dataclasses
import json
import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

import llmexer.base.analysis.prisma as prisma
import llmexer.constants as constants
from llmexer.base.search_engines import ENGINES, parse_engine_keys, resolve_engines
from llmexer.cli import app
from llmexer.exceptions import UnexpectedCLIParamsException
from llmexer.paths import SEARCH_ENGINES_ENV_VAR, resolve_paths

runner = CliRunner()
PID = "engines-exp"


def _paper(title, doi, source):
    return {"title": title, "doi": doi, "year": 2024, "entry_source": source}


@pytest.fixture()
def project(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))
    monkeypatch.setenv("PROJECT_ID", PID)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("CORE_API_KEY", raising=False)
    os.makedirs(tmp_path / PID)
    return tmp_path / PID


# Engine key -> the own search string (custom_query) the fake engine received.
received: dict = {}


@pytest.fixture()
def fake_engines(monkeypatch):
    """Replace every engine's caller with a fake that returns fixed papers and records its call."""

    found = {
        "semantic_scholar": [_paper("A", "10.1/a", "Semantic Scholar")],
        "openalex": [_paper("A", "10.1/a", "OpenAlex"), _paper("B", "10.1/b", "OpenAlex")],
        "core": [_paper("C", "10.1/c", "CORE")],
        "europepmc": [_paper("B", "10.1/b", "Europe PMC"), _paper("D", "10.1/d", "Europe PMC")],
        "pubmed": [_paper("D", "10.1/d", "PubMed")],
        "arxiv": [_paper("E", "10.48550/arXiv.1", "arXiv")],
    }
    calls = []
    received.clear()
    for key, papers in found.items():

        def runner_for(*args, key=key, papers=papers, custom_query=None):
            calls.append(key)
            received[key] = custom_query
            with open(args[5], "w", encoding="utf-8") as fh:  # raw_path
                json.dump([], fh)
            return list(papers)

        monkeypatch.setitem(ENGINES, key, dataclasses.replace(ENGINES[key], runner=runner_for))
    return calls


def _run(*options):
    return runner.invoke(app, ["search", "run", "--query", "sensor", *options])


def _with_default(monkeypatch, engines: str):
    """The engine list LLMEXER_SEARCH_ENGINES / llmexer.ini would give."""
    monkeypatch.setattr(constants, "SEARCH_ENGINES", engines)


def _search_file(project, name: str, text: str):
    searches = project / "searches"
    searches.mkdir(exist_ok=True)
    (searches / name).write_text(text, encoding="utf-8")


def _in_output(text: str, result) -> bool:
    """True when ``text`` is in the output. Rich wraps long lines (e.g. Windows temp paths), so whitespace is ignored."""
    return "".join(text.split()) in "".join(result.output.split())


def _counts(project):
    return json.loads(next((project / "searches" / "logs").glob("*__counts.json")).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Which engines
# ---------------------------------------------------------------------------


def test_parse_engine_keys():
    assert parse_engine_keys(["PubMed, arxiv", "pubmed", " europepmc "]) == ["pubmed", "arxiv", "europepmc"]
    assert parse_engine_keys(None) == []


def test_unknown_engine_lists_the_known_ones():
    with pytest.raises(UnexpectedCLIParamsException, match="Unknown search engine.*scopus.*Known: semantic_scholar"):
        parse_engine_keys("pubmed,scopus")


def test_resolution_order(monkeypatch):
    monkeypatch.setattr(constants, "SEARCH_ENGINES", "arxiv")

    assert resolve_engines(["europepmc"]) == ["europepmc"]
    assert resolve_engines("pubmed, arxiv") == ["pubmed", "arxiv"]
    assert resolve_engines(None) == ["arxiv"]
    assert resolve_engines([]) == ["arxiv"]
    monkeypatch.setattr(constants, "SEARCH_ENGINES", None)
    assert resolve_engines() == ["semantic_scholar", "openalex", "core"]


def test_ini_and_env_set_the_default(isolated_llmexer_home, monkeypatch):
    resolve_paths()  # creates the ini
    ini = isolated_llmexer_home / constants.INI_FILE_NAME
    assert "search_engines = semantic_scholar, openalex, core" in ini.read_text(encoding="utf-8")

    ini.write_text(ini.read_text(encoding="utf-8").replace("openalex, core", "pubmed"), encoding="utf-8")
    assert resolve_paths().search_engines == "semantic_scholar, pubmed"

    monkeypatch.setenv(SEARCH_ENGINES_ENV_VAR, "arxiv")
    assert resolve_paths().search_engines == "arxiv"


def test_an_old_ini_without_the_setting_uses_the_default(isolated_llmexer_home):
    isolated_llmexer_home.mkdir(parents=True, exist_ok=True)
    (isolated_llmexer_home / constants.INI_FILE_NAME).write_text(
        "[llmexer]\nprojects_dir = .projects\n", encoding="utf-8"
    )

    assert resolve_paths().search_engines == constants.DEFAULT_SEARCH_ENGINES


# ---------------------------------------------------------------------------
# search run / search create
# ---------------------------------------------------------------------------


def test_default_engines_skip_the_ones_without_a_key(project, fake_engines):
    result = _run()

    assert result.exit_code == 0, result.output
    assert fake_engines == ["semantic_scholar"]
    assert _in_output("OpenAlex skipped", result) and _in_output("CORE skipped", result)
    assert _counts(project) == {
        "semantic_scholar": 1,
        "openalex": 0,
        "openalex_duplicates": 0,
        "core": 0,
        "core_duplicates": 0,
    }


def test_key_engines_run_when_their_key_is_set(project, fake_engines, monkeypatch):
    monkeypatch.setenv("OPENALEX_API_KEY", "k1")
    monkeypatch.setenv("CORE_API_KEY", "k2")

    result = _run()

    assert result.exit_code == 0, result.output
    assert fake_engines == ["semantic_scholar", "openalex", "core"]
    assert _counts(project)["openalex_duplicates"] == 1


def test_engines_run_in_order_and_add_only_new_papers(project, fake_engines, monkeypatch):
    _with_default(monkeypatch, "europepmc, pubmed, arxiv")

    result = _run()

    assert result.exit_code == 0, result.output
    assert fake_engines == ["europepmc", "pubmed", "arxiv"]
    assert _counts(project) == {"europepmc": 2, "pubmed": 1, "pubmed_duplicates": 1, "arxiv": 1, "arxiv_duplicates": 0}
    csv_text = next((project / "searches").glob("*__results.csv")).read_text(encoding="utf-8")
    assert csv_text.count("Europe PMC") == 2 and "arXiv" in csv_text and "PubMed" not in csv_text
    saved = yaml.safe_load(next((project / "searches").glob("*.yaml")).read_text(encoding="utf-8"))
    assert saved["engines"] == ["europepmc", "pubmed", "arxiv"]
    raw_files = sorted(path.name.split("__", 1)[1] for path in (project / "searches" / "jsons").iterdir())
    assert raw_files == ["arxiv_results_raw.json", "europepmc_results_raw.json", "pubmed_results_raw.json"]


def test_search_file_engines_win_over_the_default(project, fake_engines, monkeypatch):
    _with_default(monkeypatch, "arxiv")
    _search_file(project, "s1.yaml", "query: sensor\nyear: 2020-2025\nengines: [pubmed, europepmc]\n")

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert fake_engines == ["pubmed", "europepmc"]


def test_engine_option_is_gone(project, fake_engines):
    result = _run("--engine", "arxiv")

    assert result.exit_code != 0
    assert _in_output("No such option", result)
    assert fake_engines == []


def test_old_search_file_without_engines_runs_as_before(project, fake_engines):
    _search_file(project, "old.yaml", "query: sensor\nyear: 2020-2025\nonlyOpenAccess: false\n")

    result = runner.invoke(app, ["search", "run", "--file", "old.yaml"])

    assert result.exit_code == 0, result.output
    assert fake_engines == ["semantic_scholar"]


def test_unknown_engine_in_the_search_file(project, fake_engines):
    _search_file(project, "s1.yaml", "query: sensor\nengines: [pubmed, scopus]\n")

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "scopus" in str(result.exception)
    assert fake_engines == []


def test_unknown_engine_in_the_default(project, fake_engines, monkeypatch):
    _with_default(monkeypatch, "semantic_scholar, scopus")

    result = _run()

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert fake_engines == []


def test_dry_run_names_every_raw_file(project, fake_engines, monkeypatch):
    _with_default(monkeypatch, "pubmed, arxiv")

    result = runner.invoke(app, ["--dry-run", "search", "run", "--query", "x"])

    assert result.exit_code == 0, result.output
    assert _in_output("__pubmed_results_raw.json", result) and _in_output("__arxiv_results_raw.json", result)
    assert fake_engines == []


def test_search_create_saves_the_default_engines(project, monkeypatch):
    _with_default(monkeypatch, "arxiv")

    result = runner.invoke(app, ["search", "create", "--query", "x"])

    assert result.exit_code == 0, result.output
    saved = yaml.safe_load(next((project / "searches").glob("*.yaml")).read_text(encoding="utf-8"))
    assert saved["engines"] == ["arxiv"]


def test_rename_moves_every_raw_file(project):
    jsons = project / "searches" / "jsons"
    jsons.mkdir(parents=True)
    (project / "searches" / "old.yaml").write_text("query: x\n", encoding="utf-8")
    for engine in ENGINES.values():
        (jsons / f"old{engine.raw_suffix}").write_text("[]", encoding="utf-8")

    result = runner.invoke(app, ["search", "rename", "--old-id", "old", "--new-id", "new"])

    assert result.exit_code == 0, result.output
    assert sorted(path.name for path in jsons.iterdir()) == sorted(f"new{e.raw_suffix}" for e in ENGINES.values())


# ---------------------------------------------------------------------------
# PRISMA reads every engine
# ---------------------------------------------------------------------------


def test_prisma_labels_match_the_engines():
    assert prisma.ENGINE_LABELS == {key: engine.label for key, engine in ENGINES.items()}


def test_prisma_counts_every_engine(project, fake_engines, monkeypatch):
    _with_default(monkeypatch, "europepmc, pubmed, arxiv")
    assert _run().exit_code == 0
    search_id = next((project / "searches").glob("*.yaml")).stem

    by_database, _ = prisma.identified_by_database(project, [search_id])
    duplicates, _ = prisma.duplicates_removed(project, [search_id])

    assert by_database == {"Europe PMC": 2, "PubMed": 1, "arXiv": 1}
    assert duplicates == 1


# ---------------------------------------------------------------------------
# Own search string per engine: `queries:` in the search file
# ---------------------------------------------------------------------------


def test_own_strings_reach_their_engines_only(project, fake_engines):
    _search_file(
        project,
        "s1.yaml",
        "query: sensor\nengines: [europepmc, pubmed, arxiv]\n"
        "queries:\n  pubmed: 'sensor[tiab] AND asthma[mh]'\n  arxiv: ''\n",
    )

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert received == {"europepmc": None, "pubmed": "sensor[tiab] AND asthma[mh]", "arxiv": None}


def test_unknown_engine_in_queries_stops_the_run(project, fake_engines):
    _search_file(project, "s1.yaml", "query: sensor\nengines: [pubmed]\nqueries:\n  pubmed_x: 'a'\n")

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert "pubmed_x" in str(result.exception)
    assert fake_engines == []


@pytest.mark.parametrize("queries", ["queries: 'just a string'\n", "queries:\n  pubmed: [a, b]\n"])
def test_malformed_queries_stop_the_run(project, fake_engines, queries):
    _search_file(project, "s1.yaml", "query: sensor\nengines: [pubmed]\n" + queries)

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert isinstance(result.exception, UnexpectedCLIParamsException)
    assert fake_engines == []


def test_own_string_for_an_unused_engine_is_a_warning(project, fake_engines):
    _search_file(project, "s1.yaml", "query: sensor\nengines: [pubmed]\nqueries:\n  arxiv: 'ti:x'\n")

    result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert _in_output("queries: arxiv:", result) and _in_output("not used", result)
    assert fake_engines == ["pubmed"]


def test_run_logs_the_search_and_its_own_strings(project, fake_engines, caplog):
    _search_file(project, "s1.yaml", "query: sensor\nengines: [europepmc, pubmed]\nqueries:\n  pubmed: 'x[mh]'\n")

    with caplog.at_level("INFO", logger="llmexer"):
        result = runner.invoke(app, ["search", "run", "--file", "s1.yaml"])

    assert result.exit_code == 0, result.output
    assert "Search 's1'" in caplog.text
    assert "engines europepmc, pubmed; own strings from queries: pubmed" in caplog.text


def test_new_search_file_carries_a_commented_queries_example(project, monkeypatch):
    _with_default(monkeypatch, "semantic_scholar, pubmed")

    assert runner.invoke(app, ["search", "create", "--query", "x"]).exit_code == 0

    text = next((project / "searches").glob("*.yaml")).read_text(encoding="utf-8")
    assert "# queries:\n#   semantic_scholar: '...'\n#   pubmed: '...'" in text
    assert "queries" not in yaml.safe_load(text)
