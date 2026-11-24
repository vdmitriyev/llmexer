"""Where an experiment's files live: one `experiment/` folder, or several sub-experiments.

A project uses one of two layouts:

- ``single`` (no ``experiments.yaml``): every file sits in ``experiments/``.
- ``multiple``: ``experiments/experiments.yaml`` lists named sub-experiments. They share
  ``llm-params.csv`` and ``prompts/`` in ``experiments/``, and each one keeps its own
  ``data.csv``, ``mapping.csv``, ``llms-for-experiment.csv``, databases and
  ``responses/`` in ``experiments/<name>/``.

The folder name comes from ``experiments_dir`` in ``.llmexer/llmexer.ini`` (default
``experiments``; it was ``experiment`` before 0.6.0).
"""

import os
import re
from dataclasses import dataclass, field
from enum import Enum

import yaml

import llmexer.constants as _constants
from llmexer.base.experiment import (
    DIR_PROMPTS,
    DIR_RESPONSES,
    FILE_DATA,
    FILE_LLM_PARAMS,
    FILE_LLMS_FOR_EXPERIMENT,
    FILE_MAPPING,
)
from llmexer.exceptions import (
    ExperimentLayoutException,
    LLMExerException,
    ProjectNotExistsException,
)

EXPERIMENTS_YAML = "experiments.yaml"
EXP_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
RESERVED_NAMES = {DIR_PROMPTS, DIR_RESPONSES}
FILE_KEYS = {"data": FILE_DATA, "mapping": FILE_MAPPING, "llms": FILE_LLMS_FOR_EXPERIMENT}


class Layout(str, Enum):
    """How a project's ``experiment/`` folder is organised."""

    single = "single"
    multiple = "multiple"


@dataclass
class Manifest:
    """The content of ``experiments.yaml``."""

    default: str | None = None
    experiments: dict[str, dict[str, str]] = field(default_factory=dict)

    def names(self) -> list[str]:
        return list(self.experiments)


@dataclass(frozen=True)
class ExperimentPaths:
    """Every path one (sub-)experiment reads or writes."""

    pid: str
    name: str | None
    root: str
    work_dir: str
    data: str
    mapping: str
    llms: str

    @property
    def layout(self) -> Layout:
        return Layout.single if self.name is None else Layout.multiple

    @property
    def llm_params(self) -> str:
        return os.path.join(self.root, FILE_LLM_PARAMS)

    @property
    def prompts_dir(self) -> str:
        return os.path.join(self.root, DIR_PROMPTS)

    @property
    def responses_dir(self) -> str:
        return os.path.join(self.work_dir, DIR_RESPONSES)

    def exp_args(self) -> list[str]:
        """``["--exp", name]`` for commands printed for the user; empty in a single project."""
        return [] if self.name is None else ["--exp", self.name]

    def exp_flag(self) -> str:
        """`` --exp <name>`` as a string to put into a printed command, or ``""``."""
        return "" if self.name is None else f" --exp {self.name}"

    def label(self) -> str:
        """``pid`` or ``pid / name``."""
        return self.pid if self.name is None else f"{self.pid} / {self.name}"


def experiments_dir_name() -> str:
    """The configured experiments folder name (``experiments`` unless the ini or env say otherwise)."""
    return _constants.EXPERIMENTS_DIR or _constants.DEFAULT_EXPERIMENTS_DIR


def experiment_root(project_path: str) -> str:
    return os.path.join(project_path, experiments_dir_name())


def legacy_folder_hint(project_path: str) -> str | None:
    """How to fix a project that still has the pre-0.6.0 ``experiment/`` folder, or ``None``."""

    legacy = _constants.LEGACY_EXPERIMENTS_DIR
    current = experiments_dir_name()
    if current == legacy or os.path.exists(experiment_root(project_path)):
        return None
    if not os.path.isdir(os.path.join(project_path, legacy)):
        return None
    old_path = os.path.join(project_path, legacy)
    new_path = experiment_root(project_path)
    return (
        f"This project still has the old '{legacy}/' folder. llmexer from version 0.6.0 now reads '{current}/'.\n"
        f"Rename it or set 'experiments_dir = {legacy}' in .llmexer/llmexer.ini "
        f"(LLMEXER_EXPERIMENTS_DIR={legacy}).\n"
        "Command for the rename:\n"
        f"mv {old_path} {new_path}"
    )


def validate_name(name: str) -> str:
    """Return ``name`` if it can be a sub-experiment folder name, else raise."""

    if not name or not EXP_NAME_PATTERN.match(name) or name in RESERVED_NAMES:
        raise ExperimentLayoutException(
            f"Invalid experiment name '{name}'. Use letters, digits, '_' and '-', "
            f"starting with a letter or digit (not {', '.join(sorted(RESERVED_NAMES))})."
        )
    return name


def default_files() -> dict[str, str]:
    return dict(FILE_KEYS)


def manifest_path(root: str) -> str:
    return os.path.join(root, EXPERIMENTS_YAML)


def load_manifest(root: str) -> Manifest | None:
    """Read ``experiments.yaml``; ``None`` when the project uses the single layout."""

    path = manifest_path(root)
    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ExperimentLayoutException(f"Could not read '{path}': {e}") from e
    if not isinstance(raw, dict):
        raise ExperimentLayoutException(f"'{path}' must hold a mapping with an 'experiments' key.")

    unknown_keys = set(raw) - {"layout", "default", "experiments"}
    if unknown_keys:
        raise ExperimentLayoutException(f"Unknown key(s) in '{path}': {', '.join(sorted(unknown_keys))}.")

    raw_experiments = raw.get("experiments") or {}
    if not isinstance(raw_experiments, dict) or not raw_experiments:
        raise ExperimentLayoutException(f"'{path}' lists no experiments.")

    experiments = {}
    for name, files in raw_experiments.items():
        validate_name(str(name))
        files = files or {}
        if not isinstance(files, dict):
            raise ExperimentLayoutException(f"Experiment '{name}' in '{path}' must be a mapping of files.")
        unknown_files = set(files) - set(FILE_KEYS)
        if unknown_files:
            raise ExperimentLayoutException(
                f"Unknown file key(s) for experiment '{name}' in '{path}': {', '.join(sorted(unknown_files))}. "
                f"Allowed: {', '.join(FILE_KEYS)}."
            )
        resolved_files = default_files()
        resolved_files.update({key: str(value) for key, value in files.items() if value})
        experiments[str(name)] = resolved_files

    default = raw.get("default")
    if default is not None and str(default) not in experiments:
        raise ExperimentLayoutException(
            f"The default experiment '{default}' in '{path}' is not listed. " f"Experiments: {', '.join(experiments)}."
        )

    return Manifest(default=None if default is None else str(default), experiments=experiments)


def save_manifest(root: str, manifest: Manifest) -> str:
    """Write ``experiments.yaml`` and return its path."""

    content = {
        "layout": Layout.multiple.value,
        "default": manifest.default,
        "experiments": manifest.experiments,
    }
    path = manifest_path(root)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False, allow_unicode=True)
    return path


def _project_path(pid: str) -> str:
    project_path = os.path.join(_constants.PROJECTS_PATH, pid)
    if not os.path.exists(project_path):
        raise ProjectNotExistsException(f"Project '{pid}' does not exist.")
    return project_path


def _paths_for(pid: str, root: str, name: str | None, files: dict[str, str]) -> ExperimentPaths:
    work_dir = root if name is None else os.path.join(root, name)
    return ExperimentPaths(
        pid=pid,
        name=name,
        root=root,
        work_dir=work_dir,
        data=os.path.join(work_dir, files["data"]),
        mapping=os.path.join(work_dir, files["mapping"]),
        llms=os.path.join(work_dir, files["llms"]),
    )


def _pick_name(manifest: Manifest, exp: str | None, root: str) -> str:
    names = ", ".join(manifest.names())
    if exp is not None:
        if exp not in manifest.experiments:
            raise ExperimentLayoutException(f"Unknown experiment '{exp}'. Experiments: {names}.")
        return exp
    if manifest.default:
        return manifest.default
    if len(manifest.experiments) == 1:
        return manifest.names()[0]
    raise ExperimentLayoutException(
        f"Several experiments and no default in '{manifest_path(root)}'. " f"Pass --exp with one of: {names}."
    )


def resolve_experiment(pid: str, exp: str | None = None, must_exist: bool = True) -> ExperimentPaths:
    """Resolve the files of project ``pid``'s experiment ``exp`` (``None`` = the default).

    Raises:
        LLMExerException: the project has no experiments folder (``must_exist``).
        ExperimentLayoutException: ``exp`` is unknown, is passed to a single project,
            or is missing while several experiments have no default.
    """

    project_path = _project_path(pid)
    root = experiment_root(project_path)
    if must_exist and not os.path.exists(root):
        hint = legacy_folder_hint(project_path)
        if hint:
            raise ExperimentLayoutException(hint)
        raise LLMExerException(f"Project '{pid}' has not been initialised. Run `experiment init --pid {pid}` first.")

    manifest = load_manifest(root)
    if manifest is None:
        if exp is not None:
            raise ExperimentLayoutException(
                f"Project '{pid}' uses the single layout, so --exp '{exp}' does not apply. "
                f"Run `experiment convert --pid {pid} --exp <name>` to switch to several experiments."
            )
        return _paths_for(pid, root, None, default_files())

    name = _pick_name(manifest, exp, root)
    return _paths_for(pid, root, name, manifest.experiments[name])


def list_sub_experiments(project_path: str) -> list[ExperimentPaths]:
    """All (sub-)experiments of a project; one entry for a single project, none if not initialised."""

    pid = os.path.basename(os.path.normpath(project_path))
    root = experiment_root(project_path)
    if not os.path.exists(root):
        return []
    manifest = load_manifest(root)
    if manifest is None:
        return [_paths_for(pid, root, None, default_files())]
    return [_paths_for(pid, root, name, files) for name, files in manifest.experiments.items()]


def default_name(project_path: str) -> str | None:
    """The sub-experiment used without ``--exp``; ``None`` for a single project or no default."""
    manifest = load_manifest(experiment_root(project_path))
    if manifest is None:
        return None
    if manifest.default:
        return manifest.default
    return manifest.names()[0] if len(manifest.experiments) == 1 else None
