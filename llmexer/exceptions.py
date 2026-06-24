"""Provides module specific exceptions."""


class LLMExerException(Exception):
    """Generic exception for LLMExerException."""


class ProjectAlreadyExistsException(LLMExerException):
    """Raised when a project with the given ID already exists."""


class ProjectNotExistsException(LLMExerException):
    """Raised when a project with the given ID does not exists."""


class ProjectIDRequiredException(LLMExerException):
    """Raised a project ID has not been passed."""


class UnexpectedCLIParamsException(LLMExerException):
    """Raised then unexpected CLI params are passed ."""


class PaperAlreadyExistsException(LLMExerException):
    """Raised when a paper with the same filename already exists in the papers directory."""


class PaperAddException(LLMExerException):
    """Raised when a paper cannot be added (e.g. not a PDF, download failure)."""


class PaperDownloadException(LLMExerException):
    """Raised when a DOI cannot be resolved to an open-access PDF via Unpaywall,
    or when the Unpaywall API call itself fails."""


class PaperExtractException(LLMExerException):
    """Raised when a paper cannot be extracted."""


class SearchResultsAlreadyExistException(LLMExerException):
    """Raised when search result files already exist and --force-rewrite is not set."""
