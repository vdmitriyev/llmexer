"""Where llmexer keeps its data: the `.llmexer` home, its ini file and the basedir.

Inside a virtualenv the home is `<folder holding the venv>/.llmexer`; otherwise it is
`~/.llmexer`. The ini file in it sets the basedir (where the projects folder lives)
and the projects folder name. `LLMEXER_BASEDIR` and `LLMEXER_PROJECTS_DIR` override it.
"""

import configparser
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import llmexer.constants as _constants
from llmexer.exceptions import ConfigException

INI_SECTION = "llmexer"
BASEDIR_ENV_VAR = "LLMEXER_BASEDIR"
PROJECTS_DIR_ENV_VAR = "LLMEXER_PROJECTS_DIR"


@dataclass
class RuntimePaths:
    """Every path llmexer writes to, resolved once per CLI call."""

    venv: bool
    home: Path
    ini_file: Path
    basedir: Path
    basedir_source: str
    projects_dir: str
    projects_path: Path
    temp_path: Path
    log_file_path: Path


def in_virtualenv() -> bool:
    """True when running inside a venv / virtualenv / uv environment."""
    return sys.prefix != sys.base_prefix


def venv_root() -> Path:
    """The folder that holds the active virtualenv."""
    return Path(sys.prefix).resolve().parent


def llmexer_home() -> Path:
    """The `.llmexer` folder: next to the venv, or in the user's home folder."""
    if in_virtualenv():
        return venv_root() / _constants.LLMEXER_HOME_DIR
    return Path.home() / _constants.LLMEXER_HOME_DIR


def _default_basedir_value(venv: bool) -> str:
    if venv:
        return "."
    return str(Path.home().resolve())


def ensure_ini(home: Path, venv: bool) -> Path:
    """Create `<home>/llmexer.ini` with default values if it does not exist yet."""

    ini_file = home / _constants.INI_FILE_NAME
    if ini_file.exists():
        return ini_file

    home.mkdir(parents=True, exist_ok=True)
    if venv:
        basedir_comment = "; Relative paths are resolved against the folder that holds the virtualenv."
    else:
        basedir_comment = "; Must be an absolute path."
    content = (
        f"[{INI_SECTION}]\n"
        f"{basedir_comment}\n"
        f"basedir = {_default_basedir_value(venv)}\n"
        "; Name of the folder (inside basedir) that holds the projects.\n"
        f"projects_dir = {_constants.DEFAULT_PROJECTS_DIR}\n"
    )
    ini_file.write_text(content, encoding="utf-8")
    return ini_file


def _read_ini(ini_file: Path) -> configparser.SectionProxy:
    parser = configparser.ConfigParser()
    try:
        parser.read(ini_file, encoding="utf-8")
    except (configparser.Error, OSError) as e:
        raise ConfigException(f"Could not read '{ini_file}': {e}") from e
    if not parser.has_section(INI_SECTION):
        raise ConfigException(f"'{ini_file}' has no [{INI_SECTION}] section.")
    return parser[INI_SECTION]


def _ini_basedir(value: str, venv: bool, ini_file: Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if not venv:
        raise ConfigException(f"'basedir' in '{ini_file}' must be an absolute path, got '{value}'.")
    return (venv_root() / path).resolve()


def resolve_paths() -> RuntimePaths:
    """Resolve the basedir and projects folder from the environment and the ini file."""

    venv = in_virtualenv()
    home = llmexer_home()
    ini_file = ensure_ini(home, venv)
    ini = _read_ini(ini_file)

    env_basedir = os.getenv(BASEDIR_ENV_VAR, "").strip()
    ini_basedir = ini.get("basedir", "").strip()
    if env_basedir:
        basedir = Path(env_basedir).expanduser().resolve()
        basedir_source = "env"
    elif ini_basedir:
        basedir = _ini_basedir(ini_basedir, venv, ini_file)
        basedir_source = "ini"
    else:
        basedir = _ini_basedir(_default_basedir_value(venv), venv, ini_file)
        basedir_source = "default"

    env_projects_dir = os.getenv(PROJECTS_DIR_ENV_VAR, "").strip()
    ini_projects_dir = ini.get("projects_dir", "").strip()
    projects_dir = env_projects_dir or ini_projects_dir or _constants.DEFAULT_PROJECTS_DIR

    return RuntimePaths(
        venv=venv,
        home=home,
        ini_file=ini_file,
        basedir=basedir,
        basedir_source=basedir_source,
        projects_dir=projects_dir,
        projects_path=basedir / projects_dir,
        temp_path=basedir / _constants.TEMP_DIR,
        log_file_path=home / _constants.LOG_FILE_NAME,
    )


def apply_paths(paths: RuntimePaths) -> None:
    """Publish the resolved paths on `llmexer.constants`, keeping any value already set."""

    values = {
        "BASEDIR": str(paths.basedir),
        "PROJECTS_DIR": paths.projects_dir,
        "PROJECTS_PATH": str(paths.projects_path),
        "TEMP_PATH": str(paths.temp_path),
        "LOG_FILE_PATH": str(paths.log_file_path),
    }
    for name, value in values.items():
        if getattr(_constants, name, None) is None:
            setattr(_constants, name, value)
