"""Provides module specific exceptions."""


class LLMExerException(Exception):
    """Generic exception for LLMExerException."""


class ExperimentAlreadyExistsException(LLMExerException):
    """Raised when an experiment with the given ID already exists."""


class ExperimentNotExistsException(LLMExerException):
    """Raised when an experiment with the given ID does not exists."""


class ExperimentIDRequiredException(LLMExerException):
    """Raised an experiment ID has not been passed."""


class UnexpectedCLIParamsException(LLMExerException):
    """Raised then unexpected CLI params are passed ."""


class PaperAlreadyExistsException(LLMExerException):
    """Raised when a paper with the same filename already exists in the papers directory."""


class PaperAddException(LLMExerException):
    """Raised when a paper cannot be added (e.g. not a PDF, download failure)."""
