import logging

from llmexer.constants import APP_LOG_LEVEL, LOGGER_NAME

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def get_logger(logger_name: str = None, logging_level: str = APP_LOG_LEVEL) -> logging.Logger:
    """Return the named logger. The log file is attached later by `attach_file_handler()`.

    Args:
        logger_name (str): name of the logger, `llmexer` by default
        logging_level (str): level of the logging
    """

    logger = logging.getLogger(logger_name or LOGGER_NAME)
    if logging_level is not None:
        logger.setLevel(logging_level)
    return logger


def attach_file_handler(log_file_path: str, logging_level: str = None) -> logging.Logger:
    """Write the llmexer log to `log_file_path`, replacing any log file attached before."""

    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(file_handler)

    if logging_level:
        logger.setLevel(logging_level.upper())
    return logger
