import logging
import os
import re
import sys
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from llmexer import constants
from llmexer.commands import analysis, experiment, papers, project, search
from llmexer.commands import self as self_module
from llmexer.common import ensure_directory_exists
from llmexer.configs import console, cprint, logger, settings
from llmexer.exceptions import LLMExerException
from llmexer.logger import attach_file_handler
from llmexer.paths import apply_paths, resolve_paths
from llmexer.version import package_summary, package_version

app = typer.Typer(
    help="`llmexer` is a framework and CLI utility to plan, design, run and control various LLM experiments."
)

app.add_typer(project.app, name="project")
app.add_typer(project.app, name="proj", hidden=True)

app.add_typer(experiment.app, name="experiment")
app.add_typer(experiment.app, name="exp", hidden=True)

app.add_typer(analysis.app, name="analysis")
app.add_typer(analysis.app, name="analyse", hidden=True)
app.add_typer(analysis.app, name="analyze", hidden=True)

app.add_typer(search.app, name="search")
app.add_typer(papers.app, name="papers")
app.add_typer(self_module.app, name="self")


def _load_env(env_file: Path = None) -> None:
    """Load `env_file`, or else the `.env` file in the current directory if there is one."""

    if env_file:
        cprint(
            "Loading environment variables from:",
            Text(f"{env_file}", style="bold blue"),
        )
        success = load_dotenv(env_file, override=True)
        if not success:
            typer.echo(f"Warning: Could not load variables from {env_file}", err=True)
        return

    cwd_env_file = Path.cwd() / ".env"
    if cwd_env_file.is_file():
        load_dotenv(cwd_env_file)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            "-d",
            help="Simulate execution without making changes.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Enable verbose outputs.",
        ),
    ] = False,
    env_file: Annotated[
        Path,
        typer.Option(
            "--env-file",
            "-e",
            help="Specify a path to a .env file to load environment variables.",
            exists=True,
            file_okay=True,
            dir_okay=False,
            writable=False,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show version.",
        ),
    ] = False,
) -> None:
    """
    This function runs *before* any other command (subcommand)
    or when the app is called without a subcommand.
    """
    settings.dry_run = dry_run
    settings.verbose = verbose

    if settings.verbose:
        cprint(
            Text("✅", style="bold green"),
            "Verbose mode:",
            Text("enabled", style="bold green"),
        )
        cprint(
            Text("✅", style="bold green"),
            "Dry run mode:",
            Text(
                f"{settings.dry_run}",
                style=f'bold {"red" if not settings.dry_run else "green"}',
            ),
        )
    elif settings.dry_run:
        cprint(
            Text("✅", style="bold green"),
            "Dry run mode:",
            Text("enabled", style="bold green"),
        )

    _load_env(env_file)

    project_id = os.getenv("PROJECT_ID")
    if project_id:
        settings.project_id = project_id
        if settings.verbose:
            cprint(
                Text("✅", style="bold green"),
                "Current project:",
                Text(f"{project_id}", style="bold yellow"),
            )

    runtime_paths = resolve_paths()
    apply_paths(runtime_paths)
    attach_file_handler(constants.LOG_FILE_PATH, os.getenv("APP_LOG_LEVEL", "INFO"))
    ensure_directory_exists(constants.PROJECTS_PATH)

    if version:
        if settings.verbose:
            table = Table()
            table.add_column("Field", justify="right", style="cyan", no_wrap=True)
            table.add_column("Value", justify="left", style="yellow", no_wrap=True)
            summary = package_summary()
            for item in summary:
                table.add_row(item["field"], item["value"])
            console.print(table)
        else:
            cprint(f"{package_version()}", style="yellow")
        exit(0)

    # If a subcommand was provided, don't exit; continue to the subcommand.
    # Otherwise, Typer will handle exiting or showing the help page.
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())


# Lines of an error message that are a command to copy, shown as one.
_COMMAND_LINE_PREFIXES = ("llmexer ", "mv ", "move ")
# `command` in backticks, or a 'value' in single quotes (not the apostrophe in "project's").
_HIGHLIGHT_PATTERN = re.compile(r"(`[^`\n]+`|(?<!\w)'[^'\n]+'(?!\w))")

error_console = Console(stderr=True)


def _style_error_line(line: str) -> str:
    """Rich markup for one line of an error message: commands yellow, quoted values cyan."""

    stripped = line.strip()
    if stripped.startswith(_COMMAND_LINE_PREFIXES):
        return f"  [bold yellow]{escape(stripped)}[/bold yellow]"

    styled_parts = []
    for part in _HIGHLIGHT_PATTERN.split(line):
        if part.startswith("`") and part.endswith("`") and len(part) > 1:
            styled_parts.append(f"[bold yellow]{escape(part[1:-1])}[/bold yellow]")
        elif part.startswith("'") and part.endswith("'") and len(part) > 1:
            styled_parts.append(f"[cyan]{escape(part)}[/cyan]")
        else:
            styled_parts.append(escape(part))
    return "".join(styled_parts)


def print_error(error: LLMExerException) -> None:
    """Print an llmexer error as a short colored message instead of a traceback."""

    lines = str(error).splitlines() or [type(error).__name__]
    # soft_wrap: the terminal wraps long lines itself, so a command stays copyable.
    error_console.print(f"[bold red]Error:[/bold red] {_style_error_line(lines[0])}", soft_wrap=True)
    for line in lines[1:]:
        error_console.print(_style_error_line(line), soft_wrap=True)

    details = f"{type(error).__name__}"
    if constants.LOG_FILE_PATH:
        details += f"; traceback in {constants.LOG_FILE_PATH}"
    error_console.print(f"[dim]({escape(details)}; use --verbose to show it here)[/dim]", soft_wrap=True)


def _has_log_file() -> bool:
    return any(isinstance(handler, logging.FileHandler) for handler in logger.handlers)


def run() -> None:
    """Console entry point: llmexer's own errors are shown as a message, not a traceback.

    Tests call ``app`` directly and still see the exception itself. ``--verbose``
    keeps the full traceback, for debugging.
    """

    try:
        app()
    except LLMExerException as error:
        # Only into the log file: without one, logging would print the traceback to stderr.
        if _has_log_file():
            logger.error("%s: %s", type(error).__name__, error, exc_info=True)
        if settings.verbose:
            raise
        print_error(error)
        sys.exit(1)


if __name__ == "__main__":
    run()
