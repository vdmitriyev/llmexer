"""Tests for `llmexer.base.experiment_layout`: experiments.yaml and resolving sub-experiments."""

import os

import pytest

from llmexer.base.experiment_layout import (
    EXPERIMENTS_YAML,
    Layout,
    Manifest,
    default_files,
    default_name,
    list_sub_experiments,
    load_manifest,
    resolve_experiment,
    save_manifest,
    validate_name,
)
from llmexer.exceptions import ExperimentLayoutException, LLMExerException

PID = "layout-test"


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture()
def root(projects_dir):
    path = projects_dir / PID / "experiments"
    path.mkdir(parents=True)
    return path


def _write_yaml(root, text):
    (root / EXPERIMENTS_YAML).write_text(text, encoding="utf-8")


def _multi(root, *names, default=None):
    manifest = Manifest(default=default, experiments={name: default_files() for name in names})
    save_manifest(str(root), manifest)


def test_single_layout_without_yaml(root):
    paths = resolve_experiment(PID)

    assert paths.name is None
    assert paths.layout == Layout.single
    assert paths.work_dir == str(root)
    assert paths.data == str(root / "data.csv")
    assert paths.llm_params == str(root / "llm-params.csv")
    assert paths.responses_dir == str(root / "responses")
    assert paths.exp_args() == [] and paths.exp_flag() == ""
    assert paths.label() == PID


def test_exp_on_single_layout_is_rejected(root):
    with pytest.raises(ExperimentLayoutException, match="single layout"):
        resolve_experiment(PID, "abstracts")


def test_not_initialised_project(projects_dir):
    (projects_dir / PID).mkdir()
    with pytest.raises(LLMExerException, match="not been initialised"):
        resolve_experiment(PID)


def test_multiple_layout_paths(root):
    _multi(root, "abstracts", "fulltext", default="abstracts")

    paths = resolve_experiment(PID, "fulltext")

    assert paths.layout == Layout.multiple
    assert paths.work_dir == str(root / "fulltext")
    assert paths.data == str(root / "fulltext" / "data.csv")
    assert paths.mapping == str(root / "fulltext" / "mapping.csv")
    assert paths.llms == str(root / "fulltext" / "llms-for-experiment.csv")
    # Shared files stay in experiment/.
    assert paths.llm_params == str(root / "llm-params.csv")
    assert paths.prompts_dir == str(root / "prompts")
    assert paths.responses_dir == str(root / "fulltext" / "responses")
    assert paths.exp_args() == ["--exp", "fulltext"]
    assert paths.label() == f"{PID} / fulltext"


def test_default_is_used_without_exp(root):
    _multi(root, "abstracts", "fulltext", default="fulltext")
    assert resolve_experiment(PID).name == "fulltext"


def test_only_experiment_is_used_without_default(root):
    _multi(root, "abstracts")
    assert resolve_experiment(PID).name == "abstracts"


def test_several_without_default_need_exp(root):
    _multi(root, "abstracts", "fulltext")
    with pytest.raises(ExperimentLayoutException, match="abstracts, fulltext"):
        resolve_experiment(PID)


def test_unknown_exp_lists_names(root):
    _multi(root, "abstracts", "fulltext", default="abstracts")
    with pytest.raises(ExperimentLayoutException, match="Unknown experiment 'nope'"):
        resolve_experiment(PID, "nope")


def test_custom_file_names_in_yaml(root):
    _write_yaml(
        root,
        "layout: multiple\ndefault: a\nexperiments:\n  a:\n    data: data_fulltext.csv\n",
    )

    paths = resolve_experiment(PID)

    assert paths.data == str(root / "a" / "data_fulltext.csv")
    assert paths.mapping == str(root / "a" / "mapping.csv")


@pytest.mark.parametrize(
    "text, match",
    [
        ("layout: multiple\nexperiments: {}\n", "no experiments"),
        ("experiments:\n  a: {}\nextra: 1\n", "Unknown key"),
        ("experiments:\n  a:\n    prompts: p\n", "Unknown file key"),
        ("default: b\nexperiments:\n  a: {}\n", "default experiment 'b'"),
        ("experiments:\n  'bad name': {}\n", "Invalid experiment name"),
        ("experiments: [a, b\n", "Could not read"),
        ("- a\n- b\n", "must hold a mapping"),
    ],
)
def test_invalid_yaml(root, text, match):
    _write_yaml(root, text)
    with pytest.raises(ExperimentLayoutException, match=match):
        load_manifest(str(root))


def test_save_and_load_round_trip(root):
    _multi(root, "a", "b", default="b")

    manifest = load_manifest(str(root))

    assert manifest.default == "b"
    assert manifest.names() == ["a", "b"]
    assert manifest.experiments["a"] == default_files()


@pytest.mark.parametrize("name", ["abstracts", "full_text", "v2-run", "1st"])
def test_valid_names(name):
    assert validate_name(name) == name


@pytest.mark.parametrize("name", ["", "-x", "_x", "a b", "a/b", "prompts", "responses", "ü"])
def test_invalid_names(name):
    with pytest.raises(ExperimentLayoutException):
        validate_name(name)


def test_list_sub_experiments_and_default(projects_dir, root):
    project_path = str(projects_dir / PID)
    assert [p.name for p in list_sub_experiments(project_path)] == [None]
    assert default_name(project_path) is None

    _multi(root, "a", "b", default="b")

    assert [p.name for p in list_sub_experiments(project_path)] == ["a", "b"]
    assert default_name(project_path) == "b"


def test_list_sub_experiments_not_initialised(projects_dir):
    os.makedirs(projects_dir / "empty")
    assert list_sub_experiments(str(projects_dir / "empty")) == []
