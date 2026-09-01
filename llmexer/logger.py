import logging

from llmexer.constants import APP_LOG_LEVEL, LOG_FILE_NAME, LOGGER_NAME


def get_logger(logger_name: str = None, logging_level: str = APP_LOG_LEVEL) -> None:
    """Gets a configured logger for the class

    Args:
        logging_level (str): level of the logging
    """

    logger_name = logger_name or LOGGER_NAME
    logger = logging.getLogger(logger_name)

    if not logger.handlers:

        # Sets up a logger that logs to both a file and the console.
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        file_handler = logging.FileHandler(LOG_FILE_NAME)
        file_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)

    if logging_level is not None:
        logger.setLevel(logging_level)

    return logger
