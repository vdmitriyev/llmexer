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
    for name in ("BASEDIR", "PROJECTS_DIR", "PROJECTS_PATH", "EXPERIMENTS_DIR", "TEMP_PATH", "LOG_FILE_PATH"):
        monkeypatch.setattr(constants, name, None)
    return home
