"""Provides module specific exceptions."""


class LLMExerException(Exception):
    """Generic exception for LLMExerException."""


class ExperimentAlreadyExistsException(LLMExerException):
    """Raised when an experiment with the given ID already exists."""
