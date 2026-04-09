import os

import typer
from rich.table import Table
from rich.text import Text

from llmexer.configs import console
from llmexer.version import package_version

app = typer.Typer(help="Introspection commands for the llmexer CLI.")

LLMEXER_ENV_VARS = [
    ("EXPERIMENT_ID", "bold cyan", False),
    ("UNPAYWALL_EMAIL", None, False),
    ("DOCLING_URL", None, False),
    ("DOCLING_USER", None, False),
    ("DOCLING_PASSWORD", None, True),
]


@app.command()
def version():
    """Print the current llmexer version."""
    console.print(package_version())


@app.command()
def envs():
    """Print llmexer-relevant environment variables as a table."""
    table = Table("Variable", "Value", title="Environment Variables")
    for key, style, secret in LLMEXER_ENV_VARS:
        value = os.environ.get(key, "")
        display = (
            Text("********", style="dim")
            if (secret and value)
            else (value or Text("<not set>", style=f"dim {style}" if style else "dim"))
        )
        table.add_row(
            Text(key, style=style) if style else key,
            (
                Text(str(display), style=style)
                if (style and not secret and value)
                else display
            ),
        )
    console.print(table)
