"""Tests for `llmexer.paths`: the `.llmexer` home, its ini file and the basedir."""

import configparser
import os
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import llmexer.constants as constants
import llmexer.paths as paths
from llmexer.cli import app
from llmexer.exceptions import ConfigException

runner = CliRunner()

# Captured at import, before the autouse fixture in conftest.py replaces it.
REAL_LLMEXER_HOME = paths.llmexer_home


@pytest.fixture()
def user_mode(tmp_path, monkeypatch):
    """No virtualenv; the home folder is `tmp_path / "user"`."""
    user_home = tmp_path / "user"
    user_home.mkdir()
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: user_home))
    monkeypatch.setattr(paths, "llmexer_home", lambda: user_home / constants.LLMEXER_HOME_DIR)
    monkeypatch.delenv(paths.BASEDIR_ENV_VAR, raising=False)
    return user_home


@pytest.fixture()
def venv_mode(tmp_path, monkeypatch):
    """A virtualenv at `tmp_path / "work" / ".venv"`."""
    work = tmp_path / "work"
    venv = work / ".venv"
    venv.mkdir(parents=True)
    monkeypatch.setattr(sys, "prefix", str(venv))
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path / "python"))
    monkeypatch.setattr(paths, "llmexer_home", lambda: work / constants.LLMEXER_HOME_DIR)
    monkeypatch.delenv(paths.BASEDIR_ENV_VAR, raising=False)
    return work


def _write_ini(home: Path, **values) -> None:
    home.mkdir(parents=True, exist_ok=True)
    parser = configparser.ConfigParser()
    parser[paths.INI_SECTION] = values
    with open(home / constants.INI_FILE_NAME, "w", encoding="utf-8") as f:
        parser.write(f)


def test_venv_home_is_next_to_the_venv(venv_mode, monkeypatch):
    monkeypatch.setattr(paths, "llmexer_home", REAL_LLMEXER_HOME)
    assert paths.in_virtualenv()
    assert paths.llmexer_home() == (venv_mode / constants.LLMEXER_HOME_DIR).resolve()


def test_user_home_without_venv(user_mode, monkeypatch):
    monkeypatch.setattr(paths, "llmexer_home", REAL_LLMEXER_HOME)
    assert not paths.in_virtualenv()
    assert paths.llmexer_home() == user_mode / constants.LLMEXER_HOME_DIR


def test_venv_defaults(venv_mode):
    resolved = paths.resolve_paths()

    ini_text = resolved.ini_file.read_text(encoding="utf-8")
    assert "basedir = ." in ini_text
    assert resolved.venv
    assert resolved.basedir == venv_mode.resolve()
    assert resolved.basedir_source == "ini"
    assert resolved.projects_path == venv_mode.resolve() / ".projects"
    assert resolved.log_file_path == venv_mode / constants.LLMEXER_HOME_DIR / constants.LOG_FILE_NAME


def test_user_defaults_write_absolute_basedir(user_mode):
    resolved = paths.resolve_paths()

    ini_text = resolved.ini_file.read_text(encoding="utf-8")
    assert f"basedir = {user_mode.resolve()}" in ini_text
    assert not resolved.venv
    assert resolved.basedir == user_mode.resolve()
    assert resolved.projects_path == user_mode.resolve() / ".projects"


def test_ini_values_are_used(venv_mode, tmp_path):
    target = tmp_path / "data"
    _write_ini(venv_mode / constants.LLMEXER_HOME_DIR, basedir=str(target), projects_dir="exps")

    resolved = paths.resolve_paths()

    assert resolved.basedir == target.resolve()
    assert resolved.projects_path == target.resolve() / "exps"


def test_relative_basedir_in_venv_follows_the_venv_parent(venv_mode):
    _write_ini(venv_mode / constants.LLMEXER_HOME_DIR, basedir="sub")
    assert paths.resolve_paths().basedir == (venv_mode / "sub").resolve()


def test_relative_basedir_without_venv_is_an_error(user_mode):
    _write_ini(user_mode / constants.LLMEXER_HOME_DIR, basedir="sub")
    with pytest.raises(ConfigException, match="absolute"):
        paths.resolve_paths()


def test_ini_without_section_is_an_error(user_mode):
    home = user_mode / constants.LLMEXER_HOME_DIR
    home.mkdir()
    (home / constants.INI_FILE_NAME).write_text("basedir = /x\n", encoding="utf-8")
    with pytest.raises(ConfigException):
        paths.resolve_paths()


def test_env_overrides_ini(venv_mode, tmp_path, monkeypatch):
    _write_ini(venv_mode / constants.LLMEXER_HOME_DIR, basedir=str(tmp_path / "ini"), projects_dir="ini-projects")
    monkeypatch.setenv(paths.BASEDIR_ENV_VAR, str(tmp_path / "env"))
    monkeypatch.setenv(paths.PROJECTS_DIR_ENV_VAR, "env-projects")

    resolved = paths.resolve_paths()

    assert resolved.basedir_source == "env"
    assert resolved.projects_path == (tmp_path / "env").resolve() / "env-projects"


def test_apply_paths_keeps_values_already_set(venv_mode, monkeypatch):
    monkeypatch.setattr(constants, "PROJECTS_PATH", "/kept")
    paths.apply_paths(paths.resolve_paths())
    assert constants.PROJECTS_PATH == "/kept"
    assert constants.PROJECTS_DIR == ".projects"


def test_cli_writes_nothing_into_the_cwd(venv_mode, tmp_path, monkeypatch):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    result = runner.invoke(app, ["self", "version"])

    assert result.exit_code == 0, result.output
    assert list(cwd.iterdir()) == []
    assert (venv_mode / constants.LLMEXER_HOME_DIR / constants.LOG_FILE_NAME).exists()
    assert (venv_mode / ".projects").is_dir()


def test_cli_reads_basedir_from_cwd_env_file(venv_mode, tmp_path, monkeypatch):
    cwd = tmp_path / "elsewhere"
    cwd.mkdir()
    target = tmp_path / "from-env"
    (cwd / ".env").write_text(f"{paths.BASEDIR_ENV_VAR}={target}\n", encoding="utf-8")
    monkeypatch.chdir(cwd)

    try:
        result = runner.invoke(app, ["self", "version"])
    finally:
        os.environ.pop(paths.BASEDIR_ENV_VAR, None)  # load_dotenv() writes to os.environ directly

    assert result.exit_code == 0, result.output
    assert (target / ".projects").is_dir()


def test_self_paths_shows_the_ini_file(venv_mode, monkeypatch):
    monkeypatch.setenv("COLUMNS", "300")
    result = runner.invoke(app, ["self", "paths"])
    assert result.exit_code == 0, result.output
    assert constants.INI_FILE_NAME in result.output
    assert "virtualenv" in result.output


def test_experiments_dir_default_env_and_ini(venv_mode, monkeypatch):
    assert paths.resolve_paths().experiments_dir == "experiments"

    _write_ini(venv_mode / constants.LLMEXER_HOME_DIR, basedir=".", experiments_dir="exps")
    assert paths.resolve_paths().experiments_dir == "exps"

    monkeypatch.setenv(paths.EXPERIMENTS_DIR_ENV_VAR, "experiment")
    assert paths.resolve_paths().experiments_dir == "experiment"


@pytest.mark.parametrize("value", ["a/b", "..", "."])
def test_experiments_dir_must_be_a_folder_name(venv_mode, monkeypatch, value):
    monkeypatch.setenv(paths.EXPERIMENTS_DIR_ENV_VAR, value)
    with pytest.raises(ConfigException, match="experiments_dir"):
        paths.resolve_paths()


def test_new_ini_lists_experiments_dir(user_mode):
    resolved = paths.resolve_paths()
    assert "experiments_dir = experiments" in resolved.ini_file.read_text(encoding="utf-8")
