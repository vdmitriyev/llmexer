import os

from rich.console import Console

from llmexer.logger import get_logger


class GlobalFlags:
    """Class to hold global configuration state."""

    dry_run: bool = False
    verbose: bool = False
    experiment_id: str = None


logger = get_logger()
console = Console()
settings = GlobalFlags()
