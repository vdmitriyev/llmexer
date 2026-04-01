"""Provides module specific exceptions."""


class LLMExerException(Exception):
    """Generic exception for LLMExerException."""


class ExperimentAlreadyExistsException(LLMExerException):
    """Raised when an experiment with the given ID already exists."""


class ExperimentNotExistsException(LLMExerException):
    """Raised when an experiment with the given ID does not exists."""


class ExperimentIDRequiredException(LLMExerException):
    """Raised an experiment ID has not been passed."""
