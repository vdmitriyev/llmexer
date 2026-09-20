"""Tests for the `project create` command."""

import os

import pytest
from typer.testing import CliRunner

from llmexer.cli import app
from llmexer.exceptions import ProjectAlreadyExistsException

runner = CliRunner()


@pytest.fixture()
def projects_dir(tmp_path, monkeypatch):
    """Redirect PROJECTS_PATH to a temporary directory for each test."""
    import llmexer.commands.project as project_module
    import llmexer.constants as constants

    monkeypatch.setattr(constants, "PROJECTS_PATH", str(tmp_path))
    monkeypatch.setattr(project_module, "PROJECTS_PATH", str(tmp_path))
    return tmp_path


def test_create_auto_generated_id(projects_dir):
    """Running `project create` without --id should create a YYYYMMDD-XXXX folder."""
    result = runner.invoke(app, ["project", "create"])
    assert result.exit_code == 0
    folders = list(projects_dir.iterdir())
    assert len(folders) == 1
    name = folders[0].name
    assert len(name) == 17  # YYYYMMDD-8chars
    assert name[8] == "-"


def test_create_with_custom_id(projects_dir):
    """Running `project create --id my-exp` should create a folder named my-exp."""
    result = runner.invoke(app, ["project", "create", "--id", "my-exp"])
    assert result.exit_code == 0
    assert (projects_dir / "my-exp").is_dir()
    assert "my-exp" in result.output


def test_create_duplicate_id_raises(projects_dir):
    """Running `project create --id` twice with the same ID should raise ProjectAlreadyExistsException."""
    runner.invoke(app, ["project", "create", "--id", "duplicate-exp"])
    result = runner.invoke(app, ["project", "create", "--id", "duplicate-exp"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ProjectAlreadyExistsException)


def test_create_duplicate_id_error_message(projects_dir):
    """The exception message should mention the duplicate project ID."""
    exp_id = "dup-exp"
    runner.invoke(app, ["project", "create", "--id", exp_id])
    result = runner.invoke(app, ["project", "create", "--id", exp_id])
    assert exp_id in str(result.exception)


def test_create_writes_gitignore(projects_dir):
    """A new project folder should carry a .gitignore covering the generated artefacts."""
    result = runner.invoke(app, ["project", "create", "--id", "gi-exp"])
    assert result.exit_code == 0

    gitignore = projects_dir / "gi-exp" / ".gitignore"
    assert gitignore.is_file()

    content = gitignore.read_text(encoding="utf-8")
    for pattern in (
        "*.html",
        "*.db",
        "experiment/data_backup_*.csv",
        "experiment/mapping_backup_*.csv",
        "searches/jsons/*",
        "papers/*",
        "experiment/responses/*",
        "analysis/.backup/*",
    ):
        assert pattern in content


def test_create_gitignore_matches_template(projects_dir):
    """The written .gitignore should be the shipped file, byte for byte."""
    from llmexer.base.project import SOURCE_GITIGNORE, read_project_template

    runner.invoke(app, ["project", "create", "--id", "gi-template-exp"])

    gitignore = projects_dir / "gi-template-exp" / ".gitignore"
    assert gitignore.read_text(encoding="utf-8") == read_project_template(SOURCE_GITIGNORE)


def test_create_writes_readme(projects_dir):
    """A new project folder should carry a README describing what it holds."""
    result = runner.invoke(app, ["project", "create", "--id", "readme-exp"])
    assert result.exit_code == 0

    readme = projects_dir / "readme-exp" / "README.md"
    assert readme.is_file()

    content = readme.read_text(encoding="utf-8")
    for section in ("### About", "### Structure", "### Software"):
        assert section in content
    for folder in ("experiment", "papers", "searches", "analysis"):
        assert f"[{folder}]({folder})" in content
    assert "https://pypi.org/project/llmexer/" in content


def test_create_readme_carries_the_package_version(projects_dir):
    """The README names the version that created the project."""
    from llmexer.version import package_version

    runner.invoke(app, ["project", "create", "--id", "readme-version-exp"])

    readme = projects_dir / "readme-version-exp" / "README.md"
    assert f"* Version: {package_version()}" in readme.read_text(encoding="utf-8")


def test_create_readme_matches_template(projects_dir):
    """The written README should be the shipped file with the version filled in."""
    from llmexer.base.project import (
        README_VERSION_PLACEHOLDER,
        SOURCE_README,
        read_project_template,
    )
    from llmexer.version import package_version

    runner.invoke(app, ["project", "create", "--id", "readme-template-exp"])

    readme = projects_dir / "readme-template-exp" / "README.md"
    expected = read_project_template(SOURCE_README).replace(README_VERSION_PLACEHOLDER, package_version())
    assert readme.read_text(encoding="utf-8") == expected


def test_the_bundled_project_templates_ship_with_the_package():
    """A missing data file is an install problem, so it is checked on its own."""
    from llmexer.base.project import (
        PACKAGE_PROJECT_DATA_PATH,
        SOURCE_GITIGNORE,
        SOURCE_README,
    )

    assert (PACKAGE_PROJECT_DATA_PATH / SOURCE_GITIGNORE).is_file()
    assert (PACKAGE_PROJECT_DATA_PATH / SOURCE_README).is_file()
