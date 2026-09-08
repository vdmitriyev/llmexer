"""Analysis helpers behind the ``analysis`` CLI category.

The package is split in two on purpose:

* :mod:`~llmexer.base.analysis.transform`, :mod:`~llmexer.base.analysis.stats`
  and :mod:`~llmexer.base.analysis.plots` are **copied** into a project's
  ``analysis/`` folder by ``analysis init`` and imported there as top-level
  modules (``import stats``). They must therefore stay self-contained: no
  ``llmexer`` imports, and no imports of each other. ``transform`` carries the
  loaders too, so a scaffolded ``analysis/`` folder needs nothing but pandas.
* :mod:`~llmexer.base.analysis.notebook` is package-only: it renders the bundled
  templates and copies the modules above, both of which are the CLI's job.

Nothing is imported here: ``plots`` pulls in matplotlib, which lives in the
optional ``analysis`` dependency group, and ``notebook`` is imported by the CLI
on a plain install.
"""

# Copied verbatim into <project>/analysis/ by ``analysis init``. An explicit
# tuple rather than a glob, so `notebook.py` is never copied.
COPIED_MODULES = ("transform.py", "stats.py", "plots.py")
