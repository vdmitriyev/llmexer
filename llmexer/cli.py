from enum import Enum
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.table import Table
from rich.text import Text
from typing_extensions import Annotated

from llmexer.configs import console, settings
from llmexer.version import package_summary, package_version

app = typer.Typer(
    help="`llmexer` is a framework and CLI utility to plan, design, run and control various LLM experiments."
)


class MigrationConditionChoice(str, Enum):
    due_date = "due-date"


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
        console.print(
            Text("✅", style="bold green"),
            "Verbose mode:",
            Text("enabled", style="bold green"),
        )
        console.print(
            Text("✅", style="bold green"),
            "Dry run mode:",
            Text(
                f"{settings.dry_run}",
                style=f"bold {"red" if not settings.dry_run else "green"}",
            ),
        )
    elif settings.dry_run:
        console.print(
            Text("✅", style="bold green"),
            "Dry run mode:",
            Text("enabled", style="bold green"),
        )

    if env_file:
        console.print(
            "Loading environment variables from:",
            Text(f"{env_file}", style="bold blue"),
        )
        success = load_dotenv(env_file, override=True)
        if not success:
            typer.echo(f"Warning: Could not load variables from {env_file}", err=True)
    else:
        load_dotenv()

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
            console.print(f"{package_version()}", style="yellow")
        exit(0)

    # If a subcommand was provided, don't exit; continue to the subcommand.
    # Otherwise, Typer will handle exiting or showing the help page.
    if ctx.invoked_subcommand is not None:
        return


if __name__ == "__main__":
    app()
