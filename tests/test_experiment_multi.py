"""Tests for projects with several sub-experiments (`experiment init --layout multiple`)."""

import json
import os
from unittest.mock import Mock

import pytest
import yaml
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import (
    ExperimentLayoutException,
    LLMExerException,
    UnexpectedCLIParamsException,
)
from tests.db_helpers import list_dbs, read_experiment_df

runner = CliRunner()

PID = "multi-exp"

_OLLAMA_ONLY_LLMS = "provider;model_name;profile_name;notes\nollama;phi4:14b;ollama-phi4-creative;local model\n"
_CUSTOM_DATA = "ID;Title;Abstract\nX01;Custom one;First custom.\nX02;Custom two;Second custom.\nX03;Three;Third.\n"
_CUSTOM_MAPPING = "data_id;prompt_id\nX01;prompt01\nX02;prompt01\nX03;prompt01\n"


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path / "projects"))
    os.makedirs(tmp_path / "projects" / PID)
    monkeypatch.setenv("COLUMNS", "300")
    monkeypatch.delenv("PROJECT_ID", raising=False)
    monkeypatch.setattr("llmexer.cli.load_dotenv", Mock(return_value=True))
    return tmp_path / "projects"


@pytest.fixture()
def root(projects_dir):
    return projects_dir / PID / "experiments"


@pytest.fixture()
def custom_files(tmp_path):
    files = {}
    for key, content in (("llms", _OLLAMA_ONLY_LLMS), ("data", _CUSTOM_DATA), ("mapping", _CUSTOM_MAPPING)):
        path = tmp_path / f"custom_{key}.csv"
        path.write_text(content, encoding="utf-8")
        files[key] = str(path)
    return files


@pytest.fixture()
def mock_ollama(monkeypatch):
    import llmexer.base.llm_provider as llm_module
    from llmexer.base.llm_provider import CallerState, ProviderResponse

    class FakeOllamaProvider:
        def __init__(self, provider, auth=None, base_url=None, **kwargs):
            self.state = CallerState.FINISHED

        def execute(self, prompt, row):
            self.state = CallerState.FINISHED
            return ProviderResponse(text="mocked response", total_tokens=42)

    monkeypatch.setattr(llm_module, "OllamaProvider", FakeOllamaProvider)


def _invoke(*args):
    return runner.invoke(app, list(args))


def _ok(*args):
    result = _invoke(*args)
    assert result.exit_code == 0, (result.output, result.exception)
    return result


def _manifest(root):
    return yaml.safe_load((root / "experiments.yaml").read_text(encoding="utf-8"))


def _init_multi(custom_files=None, *names):
    args = ["experiment", "init", "--pid", PID, "--layout", "multiple"]
    for name in names or ("abstracts", "fulltext"):
        args += ["--exp", name]
    if custom_files:
        args += ["--llms", custom_files["llms"]]
    return _ok(*args)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


def test_init_multiple_creates_shared_and_per_experiment_files(root):
    _init_multi()

    assert (root / "llm-params.csv").is_file()
    assert (root / "prompts" / "prompt01.txt").is_file()
    assert not (root / "data.csv").exists()
    for name in ("abstracts", "fulltext"):
        for filename in ("data.csv", "mapping.csv", "llms-for-experiment.csv"):
            assert (root / name / filename).is_file()
    manifest = _manifest(root)
    assert manifest["default"] == "abstracts"
    assert list(manifest["experiments"]) == ["abstracts", "fulltext"]


def test_init_single_is_unchanged(root):
    _ok("experiment", "init", "--pid", PID)

    assert (root / "data.csv").is_file()
    assert not (root / "experiments.yaml").exists()


def test_init_copies_custom_files(root, custom_files):
    _ok(
        "experiment",
        "init",
        "--pid",
        PID,
        "--layout",
        "multiple",
        "--exp",
        "a",
        "--data",
        custom_files["data"],
        "--mapping",
        custom_files["mapping"],
        "--llms",
        custom_files["llms"],
    )

    assert (root / "a" / "data.csv").read_text(encoding="utf-8") == _CUSTOM_DATA
    assert (root / "a" / "mapping.csv").read_text(encoding="utf-8") == _CUSTOM_MAPPING
    assert (root / "a" / "llms-for-experiment.csv").read_text(encoding="utf-8") == _OLLAMA_ONLY_LLMS


def test_init_single_accepts_custom_data(root, custom_files):
    _ok("experiment", "init", "--pid", PID, "--data", custom_files["data"])
    assert (root / "data.csv").read_text(encoding="utf-8") == _CUSTOM_DATA


@pytest.mark.parametrize(
    "extra, exception",
    [
        (["--layout", "multiple"], UnexpectedCLIParamsException),
        (["--exp", "a"], UnexpectedCLIParamsException),
        (["--layout", "multiple", "--exp", "a", "--exp", "a"], UnexpectedCLIParamsException),
        (["--layout", "multiple", "--exp", "bad name"], ExperimentLayoutException),
    ],
)
def test_init_rejects_bad_layout_params(root, extra, exception):
    result = _invoke("experiment", "init", "--pid", PID, *extra)

    assert isinstance(result.exception, exception)
    assert not root.exists()


def test_init_rejects_custom_file_with_wrong_columns(root, tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")

    result = _invoke("experiment", "init", "--pid", PID, "--data", str(bad))

    assert isinstance(result.exception, LLMExerException)
    assert "missing column" in str(result.exception)
    assert not root.exists()


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def test_add_from_templates(root):
    _init_multi()

    _ok("experiment", "add", "papers", "--pid", PID)

    assert (root / "papers" / "data.csv").is_file()
    assert list(_manifest(root)["experiments"]) == ["abstracts", "fulltext", "papers"]
    assert _manifest(root)["default"] == "abstracts"


def test_add_from_other_experiment_and_override(root, custom_files):
    _init_multi(custom_files)
    (root / "abstracts" / "mapping.csv").write_text("data_id;prompt_id\nD01;prompt01\n", encoding="utf-8")

    _ok("experiment", "add", "copy", "--pid", PID, "--from", "abstracts", "--data", custom_files["data"], "--default")

    assert (root / "copy" / "mapping.csv").read_text(encoding="utf-8") == "data_id;prompt_id\nD01;prompt01\n"
    assert (root / "copy" / "data.csv").read_text(encoding="utf-8") == _CUSTOM_DATA
    assert (root / "copy" / "llms-for-experiment.csv").read_text(encoding="utf-8") == _OLLAMA_ONLY_LLMS
    assert _manifest(root)["default"] == "copy"


def test_add_existing_name_fails(root):
    _init_multi()
    result = _invoke("experiment", "add", "fulltext", "--pid", PID)
    assert isinstance(result.exception, ExperimentLayoutException)


def test_add_on_single_layout_fails(root):
    _ok("experiment", "init", "--pid", PID)
    result = _invoke("experiment", "add", "x", "--pid", PID)
    assert isinstance(result.exception, ExperimentLayoutException)
    assert "convert" in str(result.exception)


# ---------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------


def _single_with_results(root):
    _ok("experiment", "init", "--pid", PID)
    (root / "data_backup_20260101_01.csv").write_text("ID\n", encoding="utf-8")
    (root / "llm-params_backup_20260101_01.csv").write_text("x\n", encoding="utf-8")
    (root / "experiment_20260101_01.db").write_bytes(b"")
    (root / "experiment_20260101_01.html").write_text("<html/>", encoding="utf-8")
    (root / "responses").mkdir()
    (root / "responses" / "r.json").write_text("{}", encoding="utf-8")


def test_convert_moves_experiment_files(root):
    _single_with_results(root)

    _ok("experiment", "convert", "--pid", PID, "--exp", "main")

    moved = sorted(os.listdir(root / "main"))
    assert moved == [
        "data.csv",
        "data_backup_20260101_01.csv",
        "experiment_20260101_01.db",
        "experiment_20260101_01.html",
        "llms-for-experiment.csv",
        "mapping.csv",
        "responses",
    ]
    assert sorted(os.listdir(root)) == [
        "experiments.yaml",
        "llm-params.csv",
        "llm-params_backup_20260101_01.csv",
        "main",
        "prompts",
    ]
    assert _manifest(root)["default"] == "main"


def test_convert_dry_run_moves_nothing(root):
    _single_with_results(root)
    before = sorted(os.listdir(root))

    _ok("--dry-run", "experiment", "convert", "--pid", PID, "--exp", "main")

    assert sorted(os.listdir(root)) == before


def test_convert_twice_fails(root):
    _single_with_results(root)
    _ok("experiment", "convert", "--pid", PID, "--exp", "main")

    result = _invoke("experiment", "convert", "--pid", PID, "--exp", "other")

    assert isinstance(result.exception, ExperimentLayoutException)


# ---------------------------------------------------------------------------
# generate -> run -> stats -> export -> try, per sub-experiment
# ---------------------------------------------------------------------------


def test_generate_and_run_stay_in_their_sub_experiment(root, custom_files, mock_ollama):
    _init_multi(custom_files)

    _ok("experiment", "generate", "--pid", PID, "--exp", "fulltext")
    _ok("experiment", "generate", "--pid", PID)  # the default: abstracts

    assert len(list_dbs(root / "fulltext")) == 1
    assert len(list_dbs(root / "abstracts")) == 1
    assert list_dbs(root) == []

    result = _ok("experiment", "run", "--pid", PID, "--exp", "fulltext")
    assert f"{PID} / fulltext" in result.output

    fulltext_db = root / "fulltext" / list_dbs(root / "fulltext")[0]
    abstracts_db = root / "abstracts" / list_dbs(root / "abstracts")[0]
    assert (read_experiment_df(fulltext_db)["status"] == "success").all()
    assert read_experiment_df(abstracts_db)["status"].isna().all()
    assert len(os.listdir(root / "fulltext" / "responses")) == 2
    assert not (root / "abstracts" / "responses").exists()
    assert not (root / "responses").exists()

    stats = _ok("experiment", "stats", "--pid", PID, "--exp", "fulltext")
    assert "Experiment stats" in stats.output


def test_export_try_column_carries_exp(root, custom_files):
    _init_multi(custom_files)
    _ok("experiment", "generate", "--pid", PID, "--exp", "fulltext")

    _ok("experiment", "export", "--pid", PID, "--exp", "fulltext")

    html_files = [name for name in os.listdir(root / "fulltext") if name.endswith(".html")]
    assert len(html_files) == 1
    html = (root / "fulltext" / html_files[0]).read_text(encoding="utf-8")
    assert "--exp fulltext" in html


def test_try_suggest_carries_exp(root, custom_files):
    _init_multi(custom_files)

    result = _ok("experiment", "try", "--pid", PID, "--exp", "fulltext", "--suggest")

    assert f"experiment try --pid {PID} --exp fulltext" in result.output


def test_try_writes_responses_into_sub_experiment(root, custom_files, mock_ollama):
    _init_multi(custom_files)

    _ok(
        "experiment",
        "try",
        "--pid",
        PID,
        "--exp",
        "fulltext",
        "--prompt",
        "prompt01",
        "--profile",
        "ollama-phi4-creative",
        "--data-id",
        "D01",
    )

    assert len(os.listdir(root / "fulltext" / "responses")) == 1
    assert not (root / "abstracts" / "responses").exists()


def test_copy_search_and_map_write_into_sub_experiment(projects_dir, root):
    _init_multi()
    searches = projects_dir / PID / "searches"
    searches.mkdir()
    (searches / "s.csv").write_text(
        "title;abstract;year;doi;authors\nT1;A1;2024;10.1/x;Ann\nT2;A2;2025;10.1/y;Bob\n", encoding="utf-8"
    )

    _ok("experiment", "copy-search", "--pid", PID, "--exp", "fulltext", "--file", "s.csv")
    _ok("experiment", "map", "--pid", PID, "--exp", "fulltext")

    assert "S02" in (root / "fulltext" / "data.csv").read_text(encoding="utf-8")
    assert "S02;prompt01" in (root / "fulltext" / "mapping.csv").read_text(encoding="utf-8")
    assert len([n for n in os.listdir(root / "fulltext") if n.startswith("data_backup_")]) == 1
    assert "S02" not in (root / "abstracts" / "data.csv").read_text(encoding="utf-8")


def test_several_without_default_ask_for_exp(root):
    _init_multi()
    manifest = _manifest(root)
    manifest["default"] = None
    (root / "experiments.yaml").write_text(yaml.safe_dump(manifest), encoding="utf-8")

    result = _invoke("experiment", "generate", "--pid", PID)

    assert isinstance(result.exception, ExperimentLayoutException)
    assert "abstracts, fulltext" in str(result.exception)


def test_exp_on_single_project_fails(root):
    _ok("experiment", "init", "--pid", PID)
    result = _invoke("experiment", "generate", "--pid", PID, "--exp", "a")
    assert isinstance(result.exception, ExperimentLayoutException)


# ---------------------------------------------------------------------------
# view, list, bare, analysis
# ---------------------------------------------------------------------------


def test_view_reads_sub_experiment_and_shared_files(root, custom_files):
    _init_multi(custom_files)

    llms = _ok("experiment", "view", "--pid", PID, "--exp", "abstracts", "--llms")
    params = _ok("experiment", "view", "--pid", PID, "--exp", "abstracts")

    assert f"nano {root / 'abstracts' / 'llms-for-experiment.csv'}" in llms.output
    assert "litellm" not in llms.output
    assert f"nano {root / 'llm-params.csv'}" in params.output


def test_list_shows_each_sub_experiment(root, monkeypatch):
    _init_multi()
    monkeypatch.setenv("PROJECT_ID", PID)

    result = _ok("experiment", "list")

    assert f"{PID} / abstracts *" in result.output
    assert f"{PID} / fulltext" in result.output


def test_bare_experiment_lists_sub_experiments(root, monkeypatch):
    _init_multi()
    monkeypatch.setenv("PROJECT_ID", PID)

    result = _ok("experiment")

    assert "* abstracts" in result.output
    assert "fulltext" in result.output


def test_analysis_init_points_at_sub_experiment(projects_dir, root, custom_files):
    _init_multi(custom_files)
    _ok("experiment", "generate", "--pid", PID, "--exp", "fulltext")

    _ok("analysis", "init", "--pid", PID, "--exp", "fulltext")

    notebook = projects_dir / PID / "analysis" / "fulltext__analyse_experiment.ipynb"
    source = "".join("".join(cell["source"]) for cell in json.loads(notebook.read_text(encoding="utf-8"))["cells"])
    assert "PROJECT_DIR / 'experiments' / 'fulltext'" in source
    assert "fulltext__" in source
    assert (projects_dir / PID / "analysis" / "analyse_searches.ipynb").is_file()


# ---------------------------------------------------------------------------
# experiments folder name
# ---------------------------------------------------------------------------


def test_old_experiment_folder_is_not_read(projects_dir):
    legacy = projects_dir / PID / "experiment"
    legacy.mkdir()
    (legacy / "data.csv").write_text("ID\n", encoding="utf-8")

    result = _invoke("experiment", "generate", "--pid", PID)

    assert isinstance(result.exception, ExperimentLayoutException)
    assert "experiments_dir = experiment" in str(result.exception)
    assert legacy.is_dir() and not (projects_dir / PID / "experiments").exists()


def test_old_experiment_folder_is_reported_by_list(projects_dir):
    (projects_dir / PID / "experiment").mkdir()

    result = _ok("experiment", "list")

    assert "old 'experiment/' folder" in result.output
    assert PID in result.output


def test_experiments_dir_can_be_configured(projects_dir, monkeypatch):
    monkeypatch.setenv("LLMEXER_EXPERIMENTS_DIR", "experiment")

    _ok("experiment", "init", "--pid", PID)
    view = _ok("experiment", "view", "--pid", PID)

    assert (projects_dir / PID / "experiment" / "llm-params.csv").is_file()
    assert not (projects_dir / PID / "experiments").exists()
    assert str(projects_dir / PID / "experiment" / "llm-params.csv") in view.output


def test_old_folder_message_wording(projects_dir):
    (projects_dir / PID / "experiment").mkdir()
    project_path = projects_dir / PID

    result = _invoke("experiment", "generate", "--pid", PID)

    assert str(result.exception) == (
        "This project still has the old 'experiment/' folder. llmexer from version 0.6.0 now reads 'experiments/'.\n"
        "Rename it or set 'experiments_dir = experiment' in .llmexer/llmexer.ini (LLMEXER_EXPERIMENTS_DIR=experiment).\n"
        "Command for the rename:\n"
        f"mv {os.path.join(project_path, 'experiment')} {os.path.join(project_path, 'experiments')}"
    )
