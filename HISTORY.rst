=======
History
=======

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
