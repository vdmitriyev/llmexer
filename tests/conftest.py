"""Keep every test run away from the real `.llmexer` home and its projects folder."""

import pytest

import llmexer.constants as constants
import llmexer.paths as paths


@pytest.fixture(autouse=True)
def isolated_llmexer_home(tmp_path_factory, monkeypatch):
    """Point the `.llmexer` home and the basedir into a fresh temporary folder.

    The path constants are reset to `None`, so each CLI call resolves them again.
    A test fixture that sets `constants.PROJECTS_PATH` afterwards still wins.
    """

    basedir = tmp_path_factory.mktemp("llmexer-basedir")
    home = basedir / constants.LLMEXER_HOME_DIR
    monkeypatch.setattr(paths, "llmexer_home", lambda: home)
    monkeypatch.setenv(paths.BASEDIR_ENV_VAR, str(basedir))
    monkeypatch.delenv(paths.PROJECTS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(paths.EXPERIMENTS_DIR_ENV_VAR, raising=False)
    monkeypatch.delenv(paths.SEARCH_ENGINES_ENV_VAR, raising=False)
    # Keys and caps a `.env` loaded by an earlier test may have left behind: without the
    # keys no test can reach a key-based search engine.
    for name in (
        "OPENALEX_API_KEY",
        "CORE_API_KEY",
        "NCBI_API_KEY",
        "ZENODO_API_KEY",
        "OPEN_ALEX_MAX_RESPONSES",
        "MAX_OPEN_ALEX_RESPONSES",
        "CORE_MAX_RESPONSES",
        "SEARCH_MAX_RESPONSES",
    ):
        monkeypatch.delenv(name, raising=False)
    for name in (
        "BASEDIR",
        "PROJECTS_DIR",
        "PROJECTS_PATH",
        "EXPERIMENTS_DIR",
        "SEARCH_ENGINES",
        "TEMP_PATH",
        "LOG_FILE_PATH",
    ):
        monkeypatch.setattr(constants, name, None)
    return home


@pytest.fixture(autouse=True)
def no_search_engine_network(monkeypatch):
    """Fail fast if a search caller (CORE, Europe PMC, PubMed, arXiv) is not mocked."""

    def refuse():
        raise RuntimeError("A test tried to call a real search engine; mock its HTTP session.")

    monkeypatch.setattr("llmexer.base.search_record.make_http_session", refuse)
