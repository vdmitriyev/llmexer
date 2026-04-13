import os

from rich.console import Console

from llmexer.constants import APP_LOG_LEVEL
from llmexer.logger import get_logger


class GlobalFlags:
    """Class to hold global configuration state."""

    dry_run: bool = False
    verbose: bool = False
    experiment_id: str = None


logger = get_logger()
console = Console()
settings = GlobalFlags()


def cprint(*args, log_level: str = "debug", **kwargs) -> None:
    """Print to the Rich console and, if verbose or DEBUG level, also log to file.

    Args:
        *args: Positional arguments forwarded to console.print().
        log_level: Logger method to use for file output ("debug", "info", "warning", "error").
        **kwargs: Keyword arguments forwarded to console.print().
    """
    console.print(*args, **kwargs)
    if APP_LOG_LEVEL == "DEBUG" or settings.verbose:
        message = " ".join(str(a) for a in args)
        getattr(logger, log_level, logger.debug)(message)
