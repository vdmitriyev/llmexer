"""Render, validate and write the scaffolded analysis notebooks.

Package-only. The notebooks are Jinja2 templates of raw notebook JSON living in
``llmexer/data/`` (shipped via ``[tool.setuptools.package-data]``), rendered with
an environment of this module's own rather than
:func:`llmexer.base.html_export.render_template`: that one turns autoescaping
*on*, which is right for an HTML page and would corrupt notebook JSON here.

Every injected value goes through the ``tojson`` filter, so the rendered text is
valid JSON whatever the value contains - a Windows path, a quote or a newline.
The result is then parsed and validated with ``nbformat`` before it reaches the
disk, so a broken template can never leave a corrupt ``.ipynb`` behind.
"""

import os
import pprint
import shutil

import nbformat
from jinja2 import Environment, FileSystemLoader
from nbformat import ValidationError

from llmexer.base.analysis import COPIED_MODULES
from llmexer.constants import PACKAGE_DATA_PATH
from llmexer.exceptions import LLMExerException
from llmexer.logger import get_logger

logger = get_logger()

# Bundled Jinja2 templates (see `llmexer/data/`), keyed by notebook kind.
TEMPLATES = {
    "experiment": "analyse_experiment.ipynb.j2",
    "searches": "analyse_searches.ipynb.j2",
}

# Output file per kind, written into <project>/analysis/.
NOTEBOOKS = {
    "experiment": "analyse_experiment.ipynb",
    "searches": "analyse_searches.ipynb",
}

# Where the copied modules come from: this package's own directory.
_MODULES_PATH = os.path.dirname(os.path.abspath(__file__))


def _python_literal(value) -> str:
    """Render ``value`` as Python source, for injection into a code cell.

    ``tojson`` alone is not enough for a value that lands inside a code cell:
    JSON spells them ``null`` / ``true`` / ``false``, which are not Python. This
    produces ``None`` / ``True`` / ``False`` and quotes strings the way Python
    does; the result is then embedded into the notebook JSON by ``tojson`` as
    usual.
    """

    return pprint.pformat(value, width=88, sort_dicts=True)


def render_notebook(kind: str, context: dict) -> str:
    """Render one bundled notebook template to a JSON string.

    ``autoescape`` is off on purpose - the template is JSON, not markup, and the
    ``tojson`` filter does the escaping that matters.
    """

    if kind not in TEMPLATES:
        raise LLMExerException(f"Unknown notebook kind '{kind}'. Expected one of: {sorted(TEMPLATES)}.")

    env = Environment(
        loader=FileSystemLoader(str(PACKAGE_DATA_PATH)),
        autoescape=False,  # nosec B701 - JSON output; values are escaped by `tojson`
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["pyliteral"] = _python_literal

    return env.get_template(TEMPLATES[kind]).render(**context)


def validate_notebook(source: str, kind: str):
    """Parse and validate a rendered notebook, raising on anything malformed."""

    try:
        notebook = nbformat.reads(source, as_version=4)
    except Exception as exc:  # nbformat raises several unrelated types here
        raise LLMExerException(f"Rendered '{kind}' notebook is not valid JSON/ipynb: {exc}") from exc

    try:
        nbformat.validate(notebook)
    except ValidationError as exc:
        raise LLMExerException(f"Rendered '{kind}' notebook failed nbformat validation: {exc}") from exc

    return notebook


def write_notebook(kind: str, context: dict, path: str) -> str:
    """Render, validate and write one notebook. Returns the written path."""

    notebook = validate_notebook(render_notebook(kind, context), kind)
    with open(path, "w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    logger.info(f"Analysis notebook written: {path}")

    return path


def module_source_path(filename: str) -> str:
    """Absolute path of one of the modules copied into a project."""

    return os.path.join(_MODULES_PATH, filename)


def copy_analysis_modules(target_dir: str) -> list:
    """Copy the analysis modules into ``target_dir``, returning their new paths.

    Only :data:`llmexer.base.analysis.COPIED_MODULES` travel: this module stays
    in the package, because it knows about ``llmexer`` and the copies must not.
    """

    written = []
    for filename in COPIED_MODULES:
        destination = os.path.join(target_dir, filename)
        shutil.copy2(module_source_path(filename), destination)
        written.append(destination)

    return written
