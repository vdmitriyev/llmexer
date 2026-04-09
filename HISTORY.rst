=======
History
=======

0.1.12 (2026-04-09)
-------------------

* Add ``--rewrite`` flag to ``papers extract`` (default: ``False``) to force re-extraction of already-extracted files; without it, papers with an existing ``.txt`` or ``.md`` file are skipped
* Add ``language`` column to search CSV output, detected offline from title + abstract using ``langdetect`` (ISO code, e.g. ``en``; falls back to ``unknown`` when text is empty or detection fails)
* Rename all search-related files to use a consistent ``<ID>``-only naming scheme (no ``search_`` prefix): ``<ID>.yaml``, ``<ID>_results_raw.json``, ``<ID>_results.csv``
* Rename ``--force-overwrite`` to ``--rewrite`` in ``search run`` for consistency
* Rename failed-downloads CSV from ``<stem>_failed.csv`` to ``<stem>_download_failed.csv``; add ``desired_filename`` and ``downloaded`` (always ``False``) columns

0.1.11 (2026-04-08)
-------------------

* Add ``--processor`` option to ``papers extract`` with ``pypdf`` (default, saves ``.txt``) and ``docling`` (saves ``.md``) backends
* Add docling backend support: uploads PDFs to a remote docling-serve instance via HTTP Basic Auth; reads ``DOCLING_URL`` (default ``http://localhost:5001/``), ``DOCLING_USER``, and ``DOCLING_PASSWORD`` from ``.env``
* Add ``--docling-url``, ``--docling-user``, ``--docling-password`` options to ``papers extract`` to override ``.env`` values at runtime
* Add per-file spinner to ``papers extract`` to indicate progress during extraction

0.1.10 (2026-04-02)
-------------------

* Add ``papers download`` command with ``--doi`` (repeatable) and ``--email`` / ``UNPAYWALL_EMAIL`` env var support for Unpaywall API
* Add ``--search-file`` option to ``papers download`` to read DOIs from a search result CSV (``searches/<FILE>``) and download all papers by DOI via Unpaywall
* Rename downloaded PDFs from ``--search-file`` using structured ``YEAR_TITLE_DOI.pdf`` scheme
* Save a ``<search_file_stem>_download_failed.csv`` (columns: ``doi``, ``url``, ``title``, ``desired_filename``, ``downloaded``) next to the source CSV when any downloads fail or are skipped
* Extract ``_download_pdf_from_url`` helper (shared by ``add --url`` and ``download``) supporting both ``fallback_name`` and ``forced_name`` filename modes

0.1.9 (2026-04-02)
------------------

* Rename ``search new`` subcommand to ``search create`` for consistency

0.1.8 (2026-04-02)
------------------

* Add `stats` command to `search` category to display statistics for a completed search result CSV
* Add `year` field (publication year) as the first column in search result CSV and raw JSON output
* `stats` reads `<SEARCH_ID>_results.csv`, displays two tables side-by-side: publications per year (descending) and open access breakdown
* Both tables include a `%` column (percentage of total, rounded to 1 decimal) computed in pandas
* Extract `read_search_params()` helper to reuse YAML loading and `search_id` derivation across `run` and `stats`
* Replace plain `print_search_header()` prints with a borderless Rich table for cleaner aligned output
* Tables rendered side-by-side in an invisible Rich grid with equal widths (`expand=True`, `ratio=1`)

0.1.7 (2026-04-02)
------------------

* Implement Semantic Scholar bulk search API in `search run` using direct HTTP requests (replaces `semanticscholar` SDK)
* Add `--batch` parameter to `search run` to control the number of papers fetched per API request (default: 1000)
* Change `--limit` parameter default from 100 to unlimited (`None`)
* Add `--force-overwrite` flag to `search run` to allow overwriting existing result files
* Save raw API response as `<SEARCH_ID>_results_raw.json` and flattened results as `<SEARCH_ID>_results.csv` (semicolon-separated) in the experiment's `searches/` directory
* Raise `SearchResultsAlreadyExistException` if result files already exist and `--force-overwrite` is not set
* Add `SearchResultsAlreadyExistException` exception class
* Fix `search run --file` to correctly derive search ID from the YAML filename and support absolute file paths
* Fix tests for `search run`, `papers extract`, `search eid`, `search new`
* Remove dependency: `semanticscholar`

0.1.6 (2026-04-02)
------------------

* Add `extract` command to `papers` category to extract full text from PDFs and save as `.txt` and `.md` files
* Use `pypdf` (transitive dependency) for PDF text extraction
* Respect `--dry-run` flag; skip unreadable PDFs with a warning instead of failing
* Add `PaperExtractException` exception class
* Add tests for `papers extract` command

0.1.5 (2026-04-02)
------------------

* Add `add` command to `papers` category to add PDF papers to an experiment's `papers/` subdirectory
* Support `--file` parameter to copy a single PDF file
* Support `--directory` parameter to recursively copy all PDFs from a directory
* Support `--url` parameter to download a PDF from a URL
* Raise `UnexpectedCLIParamsException` if more than one input source is provided
* Raise `PaperAlreadyExistsException` if a PDF with the same name already exists
* Add `requests` as a dependency for URL downloading
* Fix tests

0.1.4 (2026-04-01)
------------------

* Update `search` and `papers` commands to use current experiment ID as default for `--eid` parameter
* Rename `search` command to `run` in the `search` category
* Add `new` command to `search` category to create search configuration YAML files
* Update `run` command to accept `--file` parameter for loading search parameters from YAML files
* Implement Semantic Scholar API integration for paper search
* Export search results to CSV with fields: DOI, TITLE, AUTHORS, ABSTRACT, IsOpenAccess, Year, PaperId

0.1.3 (2026-04-01)
------------------

* Add categories such as `papers` and `search` to the CLI interface
* Add `rename` command to rename existing experiments with validation
* Add `current` command to `experiment` category to display the current experiment ID
* Add `EXPERIMENT_ID` loading from `.env` file on startup
* Add further tests

0.1.2 (2026-03-30)
------------------

* Modify command that crates experiments, so it would accept custom ID
* Add command to `list` experiments
* Add tests

0.1.1 (2026-03-30)
------------------

* Add command to create experiment folder under ID


0.1.0 (2026-03-27)
------------------

* First release of the basic CLI on Github
