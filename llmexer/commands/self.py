"""Self group commands of the CLI interface."""

import os
from typing import List, NamedTuple, Optional

import typer
from rich.table import Table
from rich.text import Text

from llmexer.base.budget import MAX_SPEND_ENV_VAR
from llmexer.base.llm_provider import URL_MAP
from llmexer.common import get_user_agent
from llmexer.configs import console, cprint
from llmexer.constants import (
    DEFAULT_DOCLING_URL,
    DEFAULT_MAX_OPENALEX_RESPONSES,
    DEFAULT_OPENROUTER_MAX_SPEND_USD,
)
from llmexer.version import package_version

app = typer.Typer(help="Helps with the self-manage of the llmexer CLI.")

SECRET_MASK = "********"


class EnvVar(NamedTuple):
    """One environment variable shown by ``self envs``."""

    name: str
    default: Optional[str] = None
    secret: bool = False
    style: Optional[str] = None


GENERAL_ENV_VARS = [
    EnvVar("PROJECT_ID", style="bold cyan"),
    EnvVar("APP_LOG_LEVEL", default="INFO"),
    EnvVar("LLMEXER_BASEDIR", default="<current directory>"),
]

PAPERS_SEARCH_ENV_VARS = [
    EnvVar("UNPAYWALL_EMAIL"),
    EnvVar("OPENALEX_API_KEY", secret=True),
    EnvVar("MAX_OPEN_ALEX_RESPONSES", default=str(DEFAULT_MAX_OPENALEX_RESPONSES)),
    EnvVar("DOCLING_URL", default=DEFAULT_DOCLING_URL),
    EnvVar("DOCLING_USER"),
    EnvVar("DOCLING_PASSWORD", secret=True),
]


def _provider_env_vars(provider: str) -> List[EnvVar]:
    """The URL and key variables of one provider, plus the spend cap for openrouter."""

    provider_upper = provider.upper()
    env_vars = [
        EnvVar(f"PROVIDER_{provider_upper}_URL", default=URL_MAP.get(provider)),
        EnvVar(f"PROVIDER_{provider_upper}_KEY", secret=True),
    ]
    if provider == "openrouter":
        env_vars.append(EnvVar(MAX_SPEND_ENV_VAR, default=str(DEFAULT_OPENROUTER_MAX_SPEND_USD)))
    return env_vars


def _custom_providers() -> List[str]:
    """Providers outside ``URL_MAP`` that have a ``PROVIDER_<NAME>_URL`` set."""

    custom = []
    for key in sorted(os.environ):
        if not (key.startswith("PROVIDER_") and key.endswith("_URL")):
            continue
        provider = key.removeprefix("PROVIDER_").removesuffix("_URL").lower()
        if provider and provider not in URL_MAP:
            custom.append(provider)
    return custom


def _provider_sections() -> List[List[EnvVar]]:
    """One section per built-in provider, then one per custom provider."""

    providers = list(URL_MAP) + _custom_providers()
    sections = []
    for provider in providers:
        sections.append(_provider_env_vars(provider))
    return sections


def _display_value(env_var: EnvVar) -> Text:
    """The value of ``env_var`` as shown in the table: masked when secret."""

    value = os.environ.get(env_var.name, "")
    if not value:
        return Text("<not set>", style="dim")
    if env_var.secret:
        return Text(SECRET_MASK, style="dim")
    return Text(value, style=env_var.style or "cyan")


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
    table.add_column("Value")
    table.add_column("Default", style="dim")

    sections = [GENERAL_ENV_VARS, PAPERS_SEARCH_ENV_VARS] + _provider_sections()
    for section in sections:
        for env_var in section:
            table.add_row(env_var.name, _display_value(env_var), env_var.default or "")
        table.add_section()
    console.print(table)
