"""Self group commands of the CLI interface."""

import os

import typer
from rich.table import Table
from rich.text import Text

from llmexer.common import get_user_agent
from llmexer.configs import console, cprint
from llmexer.version import package_version

app = typer.Typer(help="Helps with the self-manage of the llmexer CLI.")

LLMEXER_ENV_VARS = [
    ("EXPERIMENT_ID", "bold cyan", False),
    ("UNPAYWALL_EMAIL", None, False),
    ("DOCLING_URL", None, False),
    ("DOCLING_USER", None, False),
    ("DOCLING_PASSWORD", None, True),
]


@app.command()
def version():
    """Print the current llmexer version"""
    cprint(package_version())


@app.command()
def user_agent():
    """Print the User-Agent string used by llmexer for HTTP requests"""
    cprint("User-Agent:", Text(get_user_agent(), style="bold green"))


@app.command()
def envs():
    """Print llmexer-relevant environment variables as a table"""
    table = Table(title="Environment Variables", border_style="bright_blue")
    table.add_column("Variable", style="white", no_wrap=True)
    table.add_column("Value", style="cyan")
    for key, style, secret in LLMEXER_ENV_VARS:
        value = os.environ.get(key, "")
        display = (
            Text("********", style="dim")
            if (secret and value)
            else (value or Text("<not set>", style=f"dim {style}" if style else "dim"))
        )
        table.add_row(
            key,
            (
                Text(str(display), style=style)
                if (style and not secret and value)
                else display
            ),
        )
    console.print(table)
