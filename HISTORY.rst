=======
History
=======

0.2.34 (2026-07-03)
-------------------

* ``search filter``: ``--file`` is now optional. When omitted, the given exclusion filters are applied to every search in the project (each search's ``__filtered.csv`` rewritten), and each applied filter for each search is recorded in ``searches/logs/filters-applied.log``

0.2.33 (2026-07-03)
-------------------

* ``search merge``: the merged CSVs (``<pid>__merged_results.csv`` / ``<pid>__merged_filtered.csv``) are now sorted by ``year`` descending (newest first) before being saved; rows with a missing/blank year are placed last

0.2.32 (2026-07-03)
-------------------

* ``search sync``: new ``--existing-only`` flag that checks only the files listed in existing rows and does not append new rows for PDFs found in ``papers/`` that are not already listed
* ``papers download --search-file``: the automatic post-download sync now runs in existing-only mode, so it updates ``pdf_downloaded`` for the downloaded rows without inventing new rows for unrelated PDFs sitting in ``papers/``

0.2.31 (2026-07-03)
-------------------

* ``papers download``: the ``Failed list saved to:`` message now shows the ``logs/`` subfolder location instead of just the bare filename
* ``papers download --search-file``: after all downloads finish, the search is automatically reconciled against the ``papers/`` folder (same as ``search sync`` — updates ``pdf_downloaded`` and picks up text/markdown companions and newly present PDFs), replacing the previous DOI-only ``pdf_downloaded`` update

0.2.30 (2026-07-03)
-------------------

* ``search filter`` reworked into a chainable **exclusion** filter: it reads the existing ``<id>__filtered.csv`` if present (else ``<id>__results.csv``) and rewrites ``<id>__filtered.csv``. Combinable criteria applied in order — ``--language`` / ``--source`` / ``--doi`` drop rows equal to the given value, ``--downloaded`` drops rows not yet downloaded. Was: a single keep-matching language filter
* Each applied filter is recorded to ``searches/logs/filters-applied.log`` (``<datetime> filter applied: <filter> ; input rows: <n>; output rows: <n>``)
* ``papers download`` now saves the ``<stem>_download_failed.csv`` into the ``searches/logs/`` subdirectory (was: directly in ``searches/``); ``search rename`` moves it accordingly

0.2.29 (2026-07-03)
-------------------

* Search files now use ``__`` as the delimiter between search id and role: ``<id>__results.csv``, ``<id>__filtered.csv``, ``<id>__results_raw.json`` (was single ``_``). Raw JSON responses are written to a ``searches/jsons/`` subdirectory. Affects ``search run`` / ``filter`` / ``sync`` / ``merge`` / ``list`` / ``rename``
* ``search merge``: ``duplicates_counter`` now holds the number of duplicate occurrences (one less than the number of searches a publication was found in; ``0`` for a single-search publication)

0.2.28 (2026-07-03)
-------------------

* Add ``search merge``: combine a project's search CSVs into two deduplicated files — ``<pid>__merged_results.csv`` (from ``*_results.csv``) and ``<pid>__merged_filtered.csv`` (from ``*_filtered.csv``). Publications are deduplicated by DOI (falling back to title). Each source search adds a ``0/1`` column named after its YAML id, plus a ``duplicates_counter`` column counting how many searches contained the publication. Guarded by ``--rewrite``; respects ``--dry-run``
* ``search stats``: when ``--file`` is omitted, fall back to the merged file(s) (``<pid>__merged_results.csv`` / ``<pid>__merged_filtered.csv``) if present

0.2.27 (2026-06-25)
-------------------

* ``experiment stats``: extend the per-model breakdown table with new columns — ``finished`` (successfully finished requests), ``open`` (pending/unrun requests), ``time total`` (elapsed time over the model's finished requests, formatted ``HH:MM:SS``), ``average time`` (mean elapsed per finished request, ``HH:MM:SS``), and ``tokens`` (summed over the model's finished requests); the request-count column (in both the Providers and Models tables) is renamed from ``Count`` to ``requests``
* ``ExperimentDAO.stats()`` now returns ``models`` as a per-model aggregate dict (``count`` / ``finished`` / ``open`` / ``tokens`` / ``elapsed_seconds``) instead of a flat name→count map; ``providers`` is unchanged

0.2.26 (2026-06-25)
-------------------

* ``experiment run`` now persists the **complete raw backend response** under a ``raw_response`` key in both the per-call ``experiment/responses/*.json`` files and the database ``response_json`` column, instead of only the response text and total token count; this captures all provider fields (e.g. ``finish_reason``, per-token ``usage``, and ollama extras such as ``eval_count`` / ``*_duration``)
* Applies to all providers: add ``serialize_response`` (``llmexer/base/llm_provider.py``) which dumps an OpenAI SDK response (``extra="allow"``, so provider-specific extras survive); ``LLMRunResult`` gains a ``raw`` field and ``Experiment`` gains a ``raw_response`` field carried into ``build_response_payload``

0.2.25 (2026-06-25)
-------------------

* Move the ``list`` command from the ``project`` group to the ``experiment`` group (now ``llmexer experiment list``); it still lists all projects with their initialization state and generated experiment databases, with ``--sort-by`` / ``--desc``
* ``experiment list`` now reports generated experiments as the SQLite databases (``experiment_*.db``) instead of the obsolete generated ``*.csv`` files; the listed databases are sorted, so the "example to run" hint points at the latest one
* ``README.md``: sync the ``experiment`` CLI group docs with the SQLite-backed workflow — ``generate`` writes ``experiment_<YYYYMMDD>_<NN>.db`` (one table per provider), ``run`` writes results back into that database in place (no separate ``*_results.csv``), and ``stats`` reports ``total`` / ``finished`` / ``running`` / ``errors``

0.2.24 (2026-06-25)
-------------------

* Reorder ``llm-params.csv`` identity columns so they lead with ``provider``, ``model_name``, ``profile_name`` (was ``profile_name``, ``model_name``, ``provider``)
* Rename the ``models.csv`` input config file to ``llm-models.csv`` (``experiment init`` template and ``experiment generate`` input)
* ``experiment init`` defaults: change the default model from ``llama3.3:latest`` to ``gemma4:31b`` and trim the ``llm-models.csv`` template to two models — ``gemma4:31b`` and ``phi4:14b``; the ``llm-params.csv`` example profiles now use ``gemma4:31b`` (``ollama-default``) and ``phi4:14b`` (``ollama-creative``)
* ``README.md``: describe the ``experiment generate`` cartesian product as ``data row × prompt × LLM models × LLM parameters``

0.2.23 (2026-06-25)
-------------------

* Add ``experiment copy-papers``: copy parsed papers (``.md``/``.txt``) from the project's ``papers/`` folder into ``experiment/data.csv`` as rows ``ID;filename;content`` with IDs ``P01``, ``P02``, … ordered alphabetically by filename (``.md`` preferred over ``.txt`` when both exist)
* Add ``experiment copy-search``: copy a search results CSV (``--file``, absolute or relative to ``searches/``) into ``experiment/data.csv`` as rows ``ID;Title;Abstract;doi;authors`` with IDs ``S01``, ``S02``, … preserving the source file's row order
* Both commands back up an existing ``data.csv`` to ``data_backup_<YYYYMMDD>_<NN>.csv`` before overwriting

0.2.22 (2026-06-25)
-------------------

* ``experiment run`` now skips rows already in the ``success`` state (no LLM call, result left untouched) and prints the skip to the console
* Rename ``CallerState.SUCCESS`` to ``CallerState.FINISHED`` (value ``"success"`` → ``"finished"``); the per-row ``state`` column now stores ``"finished"`` on success (the ``status`` column still uses ``"success"`` / ``"Error: …"``)
* ``experiment stats``: drop the ``pending`` metric and rename ``completed`` → ``finished``; metrics are now total, finished, running, errors, total_tokens (plus per-provider and per-model breakdowns)

0.2.21 (2026-06-25)
-------------------

* Refine the per-provider experiment-table schema introduced in 0.2.20: drop the duplicate ``param_model_name`` / ``param_provider`` columns, since ``model_name`` / ``provider_name`` already carry the model and provider
* Move the SHA-256 hash columns (``prompt_hash``, ``original_data_hash``) to the end of each provider table
* ``README.md``: add a hint that the generated experiment database (``experiment/experiment_*.db``) can be opened and edited with an external SQLite tool such as `DBeaver <https://dbeaver.io/>`_

0.2.20 (2026-06-24)
-------------------

* **Breaking:** store generated experiments and their results in a per-generation **SQLite database** (via SQLAlchemy) instead of CSV files; ``experiment generate`` now writes ``experiment/experiment_<YYYYMMDD>_<NN>.db`` (``<NN>`` is a zero-padded counter starting at ``01``). Input config (``models.csv``, ``data.csv``, ``mapping.csv``, ``llm-params.csv``, ``prompts/*.txt``) stays as CSV/text
* Use **one table per LLM provider** (e.g. ``experiment_ollama``, ``experiment_openai``), each holding only its own parameter columns plus the generated rows and their results (status, tokens, timestamps, and the call payload as ``response_json``)
* Add a data access layer ``llmexer/base/dao.py`` (``ExperimentDAO``) that isolates all database access; ``experiment run`` writes results back into the database in place and ``experiment stats`` aggregates across provider tables
* Add ``SQLAlchemy>=2.0.36`` as a dependency

0.2.19 (2026-06-24)
-------------------

* **Breaking:** rename the top-level container from "experiment" to "project". Artifacts now live under ``.projects/<PROJECT_ID>/`` (was ``.experiments/<EXPERIMENT_ID>/``); the CLI flag ``--eid`` becomes ``--pid`` and the env var ``EXPERIMENT_ID`` becomes ``PROJECT_ID`` across all commands
* Add a new ``project`` command group (alias ``proj``) that manages the project lifecycle — ``create``, ``list``, ``rename``, ``current`` — moved out of the ``experiment`` group
* The ``experiment`` group now contains the experiment setup & execution commands: ``init``, ``generate``, ``run``, ``stats`` (all operate on a project via ``--pid``)
* Rename internal symbols for consistency: ``EXPERIMENTS_PATH``→``PROJECTS_PATH``, ``get_proper_eid``→``get_proper_pid``, ``get_experiment_directory_path``→``get_project_directory_path`` (plus new ``get_experiment_subdir_path``), ``generate_experiment_id``→``generate_project_id``, ``settings.experiment_id``→``settings.project_id``, and exceptions ``Experiment{IDRequired,NotExists,AlreadyExists}Exception``→``Project{...}Exception``

0.2.18 (2026-06-14)
-------------------

* Consolidate experiment results into a single, stable file named after the generated input file: ``experiment run --file experiment_<NAME>.csv`` now writes ``experiment/experiment_<NAME>_results.csv`` next to it (no per-run timestamp) instead of a new ``experiment_<eid>_results_<TIMESTAMP>.csv`` on every run; each generated file therefore keeps just two CSVs — the original and its results (per-call JSON responses under ``experiment/responses/`` are unchanged)
* Persistence is owned by ``ExperimentsManager``: new ``results_path()`` (derives ``<stem>_results.csv`` from the loaded file, idempotent), ``merge_results(file)`` (copies result columns from an existing results file onto matching rows, keyed by ``ID``/``code``), and ``save_results(file)`` (writes the whole DataFrame to the single results file)
* ``experiment run`` now merges in place across runs: a ``--filter-provider`` or ``--id`` run updates only the rows it executed while preserving results from earlier runs, and always persists the full row set to the one results file
* Fix ``_get_generated_experiment_files`` so ``*_results.csv`` files are excluded from the ``experiment list`` "Experiments" column
* ``experiment stats`` auto-discovers the experiment's ``*_results.csv`` when ``--file`` is omitted (raising a clear error pointing to ``experiment run`` if none exists, or asking for ``--file`` when several exist); ``--file`` still overrides to inspect any CSV

0.2.17 (2026-06-14)
-------------------

* Add ``Experiment`` dataclass and ``ExperimentsManager`` mapper in ``llmexer/base/llm_manager.py``: ``Experiment`` represents a single generated-CSV combination (prompt, model, provider, params, plus result/provider-state fields) and serialises via ``to_json(file)`` (``indent=4``) and ``to_yaml(file)`` — both default to a filename derived from ``experiment_id`` when ``file`` is omitted; ``ExperimentsManager`` owns a generated ``experiment_*.csv`` as a pandas DataFrame with ``load(file)`` / ``unload(file)`` / ``sync(file)``, runs a single combination by id via ``run(id_experiment)`` (resolving the right provider and copying ``CallerState`` / ``CallerStats`` back into the row), and reports aggregate ``stats()`` (completed, running, errors, pending, total tokens, per-provider and per-model counts)
* Add ``experiment stats`` command: loads a generated ``experiment_*.csv`` (or a results CSV) and renders aggregate statistics as Rich tables (totals plus provider/model breakdowns)
* Add ``--id`` option to ``experiment run`` to execute only a single combination by its ``ID`` (or ``code``) instead of all rows; ``experiment run`` now delegates execution to ``ExperimentsManager``
* Refactor ``llmexer/base`` module layout: split the former ``llm.py`` into ``llm_core.py`` (the ``LLMRunResult`` result contract) and ``llm_provider.py`` (provider configuration ``URL_MAP`` / ``resolve_provider_config`` plus the ``LLMRequestsMapper`` and ``OllamaProvider`` implementations); the abstract ``LLMProviderBase`` and supporting types were renamed from ``provider.py`` to ``llm_provider.py``
* Add ``PyYAML`` as an explicit dependency (previously only transitive) to back ``Experiment.to_yaml``

0.2.16 (2026-05-10)
-------------------

* Add ``OllamaProvider`` in ``llmexer/base/llm.py`` as the first concrete ``LLMProviderBase`` implementation; translates CSV row parameters to ollama-specific OpenAI-compatible API calls (``num_ctx``, ``num_predict``, ``repeat_penalty`` via ``extra_body``), tracks per-call timing and cumulative stats, and manages lifecycle state transitions (``STARTED → RUNNING → SUCCESS/ERROR``); ``experiment run`` now routes ollama rows through ``OllamaProvider`` instead of ``LLMRequestsMapper``

0.2.15 (2026-05-10)
-------------------

* Add ``tokens_estimate`` column to ``experiment generate`` output: computed as ``len(rendered_prompt) // 4`` (1 token ≈ 4 characters); column appears between ``prompt`` and ``original_data`` in the 21-column output CSV
* Add ``LLMProviderBase`` abstract base class in ``llmexer/base/provider.py`` to define the shared contract for all future LLM provider implementations; introduces ``CallerState`` enum (``started``, ``running``, ``success``, ``error``), ``ProviderAuth`` (``api_key``, ``extra_headers``), ``ProviderRequest`` (``model``, ``prompt``, ``params``), ``ProviderResponse`` (``text``, ``usage_tokens``, ``raw``), and ``CallerStats`` (``call_count``, ``total_tokens``, ``elapsed_seconds``) dataclasses; base class exposes ``data``, ``session``, ``state``, ``stats``, ``auth``, and ``timeout`` fields plus three abstract methods: ``build_session()``, ``build_request()``, and ``execute()``

0.2.14 (2026-04-27)
-------------------

* Change the behaviour of the docling extraction - no it removes images from the mardown output


0.2.13 (2026-04-21)
-------------------

* Improve documentation in README.md
* Add an example how to run an experiment in CLI into the `experiment list` command

0.2.12 (2026-04-21)
-------------------
* Tiny refactor of the code by moving functions around for better readability

0.2.11 (2026-04-20)
-------------------

* Refactor: move utility functions from ``commands/papers.py`` to ``base/papers.py`` to separate CLI logic from reusable business logic
* Moved to ``base/papers.py``: ``extract_via_docling()``, ``download_pdf_from_url()``, ``resolve_unpaywall_pdf_url()``, ``get_first_author_last_name()``, ``make_structured_filename()``, ``PDFProcessor`` enum, ``DOCLING_TIMEOUT`` and ``UNPAYWALL_EMAIL`` constants
* Update ``commands/search.py`` to import ``get_first_author_last_name`` and ``make_structured_filename`` from ``base/papers.py``
* Add ``base/__init__.py`` to establish ``llmexer.base`` as a proper Python package

0.2.10 (2026-04-16)
-------------------

* Fix behaviour of creating folders when running the experiment

0.2.9 (2026-04-16)
-------------------

* Modify outputs of the `experiment run`, so it handles status updates better

0.2.8 (2026-04-16)
-------------------

* Normalise CLI command help text: all command docstrings now end with a period for consistency across ``search``, ``experiment``, ``papers``, and ``self`` command groups

0.2.7 (2026-04-16)
-------------------

* Add ``search rename`` command: renames a search ID and all its associated files (``<id>.yaml``, ``<id>_results.csv``, ``<id>_results_raw.json``, ``<id>_filtered.csv``, ``<id>_results_download_failed.csv``); accepts ``--old-id`` and ``--new-id``; raises ``LLMExerException`` if the source does not exist or the target already exists
* Add ``search list`` command: lists all YAML search configs in the experiment's ``searches/`` folder as a Rich table (columns: ``#``, ``Name``, ``Query``, ``Year``, ``Created``, ``Results``); prints a next-step hint below the table referencing the latest search file (e.g. ``llmexer search stats --file <latest>.yaml``)
* Improve ``search stats`` Stats Breakdown table colors: ``Count`` and ``%`` columns now inherit the color of their stat row (``bold green`` for ``existing``, ``bold red`` for ``missing``, ``magenta`` for Open Access / Entry Source / Language); default column color changed from fixed green/yellow to ``cyan``

0.2.6 (2026-04-16)
-------------------

* Enhance ``experiment list`` table styling: highlight the current experiment (when ``EXPERIMENT_ID`` is set) with bold yellow text and underline only on the counter (#); change ``Name`` and ``Created`` columns to cyan; rename ``Generated Files`` column to ``Experiments``; use space separator instead of comma for generated files list
* Fix test ``test_run_provider_key_falls_back_to_llm_api_key_env``: rename to ``test_run_provider_key_defaults_to_na_when_absent`` and update assertion to reflect the current implementation where ``api_key`` defaults to ``"na"`` when ``PROVIDER_<PROVIDER>_KEY`` is absent (the ``LLM_API_KEY`` fallback was removed)

0.2.5 (2026-04-15)
-------------------

* Remove ``papers rename`` stub command: the command had no implementation and is no longer exposed in the CLI, documentation, or tests

0.2.4 (2026-04-14)
-------------------

* Refactor: add ``cprint()`` utility function to ``configs.py``: always prints to the Rich console and additionally writes a plain-text entry to the log file when ``APP_LOG_LEVEL=DEBUG`` or ``--verbose`` is passed; accepts the same ``*args``/``**kwargs`` as ``console.print()`` plus an optional ``log_level`` parameter (default: ``"debug"``)
* Remove ``StreamHandler`` from the logger so that ``logger.*()`` calls write only to ``llmexer.log`` (no duplicate unformatted lines in the terminal)
* Replace all notification-style ``console.print()`` calls across ``experiment.py``, ``papers.py``, ``search.py``, ``self.py``, and ``cli.py`` with ``cprint()``; structural Rich ``Table`` and layout renders remain as ``console.print()``
* Fix ``test_experiment_generate.py``: update ``llm-params.csv`` fixtures in tests that used ``model-a``/``model-b``/``m`` model names to include matching param profile rows so the generate command produces output

0.2.3 (2026-04-13)
-------------------

* Fix mapping between LLM models and LLM params

0.2.2 (2026-04-13)
-------------------

* Add ``experiment run`` command: executes all prompt × LLM-parameter-profile combinations from ``experiment_YYYYMMDD-GUID.csv`` (produced by ``experiment generate``) and ``llm-params.csv``; writes a ``experiment_<ID>_results_<TIMESTAMP>.csv`` results file and one JSON response file per call under ``experiment/responses/``
* Extend ``experiment init`` to also create ``experiment/llm-params.csv`` with 12 columns (``profile_name``, ``model_name``, ``provider``, ``temperature``, ``top_p``, ``max_tokens``, ``context_window``, ``seed``, ``repeat_penalty``, ``min_p``, ``best_of``, ``thinking_level``) and five example profiles covering ``ollama``, ``openai``, ``vllm``, and ``gemini`` providers
* Results CSV columns are the union of the experiment CSV columns (``ID``, ``code``, ``prompt``, ``original_data``, ``model_name``, ``provider_name``, ``prompt_hash``, ``original_data_hash``), the params CSV columns (renamed ``param_model_name`` and ``param_provider`` to avoid collision), and four computed fields: ``response_text``, ``usage_tokens``, ``status``, ``timestamp``
* ``experiment run`` supports provider-specific parameter mapping via ``llmexer/llm.py``: ``ollama`` uses ``num_ctx``/``num_predict``/``repeat_penalty`` in ``extra_body``; ``vllm`` uses ``min_p``/``best_of``; ``gemini`` uses ``thinking_level``; ``openai`` uses ``max_completion_tokens``/``seed``; all providers share ``temperature`` and ``top_p``
* All LLM calls use the OpenAI Python SDK with provider-specific ``base_url`` values (``ollama``: ``localhost:11434/v1``, ``vllm``: ``localhost:8000/v1``, ``openai``: default, ``gemini``: Google endpoint); ``openai`` is a lazy optional dependency — a clear install hint is raised if absent
* ``experiment run`` supports ``--dry-run``, ``--file`` (override auto-detected prompt CSV), ``--output`` (override results file path), ``--filter-provider`` (only run rows for a specific provider)
* API key read from ``LLM_API_KEY`` or ``PROVIDER_<PROVIDER_UPPER>_KEY`` env vars; base URL overridable via ``PROVIDER_<PROVIDER_UPPER>_URL`` env vars (fall back to built-in defaults)
* Individual call failures are recorded as ``status="Error: ..."`` rows in the results CSV; the batch continues rather than aborting
* Extract LLM call logic into new ``llmexer/llm.py`` module (``LLMRunResult`` dataclass, ``LLMRequestsMapper`` class, ``URL_MAP``) to isolate the optional ``openai`` dependency from the rest of the CLI
* Rename provider-specific columns in ``llm-params.csv`` to carry an explicit provider prefix: ``context_window`` → ``ollama_context_window``, ``repeat_penalty`` → ``ollama_repeat_penalty``, ``seed`` → ``openai_seed``, ``min_p`` → ``vllm_min_p``, ``best_of`` → ``vllm_best_of``, ``thinking_level`` → ``gemini_thinking_level``; universal columns (``temperature``, ``top_p``, ``max_tokens``) unchanged
* Reorder ``llm-params.csv`` columns to group by provider: universal → ollama → vllm → openai → gemini
* ``experiment generate`` now performs the full cartesian product (data × prompts × models × parameter profiles) and embeds all 12 param columns directly into the output CSV, producing a self-contained 20-column file; ``experiment run`` no longer reads a separate ``llm-params.csv``
* ``code`` field extended to ``DATAID_PROMPTID_MODELNAME_PROFILENAME``

0.2.1 (2026-04-13)
-------------------

* Remove ``json_params`` column from ``experiment generate`` output; the ``notes`` field in ``models.csv`` is no longer parsed as JSON

0.2.0 (2026-04-10)
-------------------

* Add ``experiment init`` command: initialises an existing experiment with a standard folder structure (``experiment/``, ``experiment/prompts/``) and four template files — ``models.csv`` (name/provider/notes), ``data.csv`` (ID/Title/Abstract), ``mapping.csv`` (data_id/prompt_id), and ``prompts/prompt-01.txt`` (``{title}``/``{abstract}`` template)
* Raise ``LLMExerException`` when ``experiment init`` is called on an already-initialised experiment; raise ``ExperimentNotExistsException`` when the experiment folder does not exist
* Extract ``get_proper_eid()`` and ``get_experiment_directory_path()`` helpers into ``common.py`` to centralise ``--eid`` resolution and experiment-path validation across command modules
* Add ``experiment generate`` command: renders all (data row, prompt, model) combinations defined in ``experiment/`` into a single ``experiment_YYYYMMDD-GUID.csv`` output file; columns are ``ID``, ``code``, ``prompt``, ``original_data``, ``model_name``, ``provider_name``, ``prompt_hash``, ``original_data_hash``
* ``ID`` column is a 1-based integer counter; rows are sorted by model order as listed in ``models.csv`` (all data rows for the first model first, then all for the second, etc.)
* ``code`` field encodes the combination as ``DATAID_PROMPTID_MODELNAME`` for easy cross-referencing
* ``prompt_hash`` and ``original_data_hash`` are SHA-256 hex digests of the rendered prompt and the serialised original data row respectively
* ``experiment generate`` supports ``--dry-run``: prints the row count and output path without writing the file; skips missing ``data_id`` or prompt file entries with a warning instead of failing
* Switch prompt template syntax from Python single-brace ``{title}`` to Jinja2 double-brace ``{{title}}`` in ``experiment init`` scaffold and all template rendering
* Extend ``experiment init`` default ``models.csv`` template with four pre-filled models: ``llama3.3:latest``, ``phi4:14b``, ``gemma3:12b``, ``gemma3:27b`` (all provider ``ollama``)
* Extend ``experiment init`` default ``mapping.csv`` template with two example rows: ``D01;prompt01`` and ``D02;prompt01``


0.1.20 (2026-04-10)
-------------------

* Add a ``Total: papers`` row at the top of the ``Stats Breakdown`` table in ``search stats``, showing the total paper count and ``100%``
* Extract first-author last-name logic from ``search run`` into a standalone ``_get_first_author_last_name`` helper in ``papers.py``, co-located with ``_make_structured_filename``

0.1.19 (2026-04-10)
-------------------

* Extend ``Stats Breakdown`` table in ``search stats`` with three new sections: per-value ``Entry Source`` counts, ``TXT existing/missing`` counts, and ``Markdown existing/missing`` counts
* Rename ``Publications per Year`` table to ``Papers by Year``; narrow Year and Count columns using ``min_width`` instead of padded column name strings
* Stack results and filtered stats tables vertically (results on top, filtered below) instead of side by side; filtered section is omitted when no filtered CSV exists
* Change ``Downloaded`` rows in ``Stats Breakdown`` to ``PDF: existing/missing`` pattern, matching TXT and Markdown
* Add ``search sync`` command: reconciles ``<ID>_results.csv`` (and ``<ID>_filtered.csv`` if present) against the experiment's ``papers/`` folder
* Sync sets ``pdf_downloaded=True`` for rows whose ``pdf_filename`` is found in ``papers/``, fills ``txt_filename`` when a matching ``.txt`` file exists, and fills ``markdown_filename`` when a matching ``.md`` file exists
* Appends new rows for PDFs in ``papers/`` not listed in the CSV; new rows have ``entry_source="manually added"``, ``pdf_downloaded=True``, and ``txt_filename``/``markdown_filename`` set if the companion files exist; all other fields are left blank
* Respects ``--dry-run``: prints what would be written without modifying any files

0.1.18 (2026-04-10)
-------------------

* Rename ``desired_filename`` column to ``pdf_filename`` in search CSV output, papers download failed CSV, and all related code
* Rename ``downloaded`` column to ``pdf_downloaded`` in search CSV output, papers download failed CSV, and all related code
* Add ``entry_source``, ``txt_filename``, and ``markdown_filename`` columns to search CSV output; ``entry_source`` is set to ``Semantic Scholar`` at search time, the other two are left blank
* Print ``Query saved to: <yaml_filename>`` notification in ``search run`` when ``--query`` is passed, before the search begins

0.1.17 (2026-04-09)
-------------------

* Change ``_detect_language`` to detect title and abstract separately; returns ``"unclear"`` when they disagree, when text is missing, or when detection fails (replaces ``"unknown"``)
* Replace ``Open Access Breakdown`` table in ``search stats`` with a ``Stats Breakdown`` table combining: open access (True only), per-language counts, and downloaded (True only)
* Colour ``Stats Breakdown`` "Stat" column so label text is white and the parameter value (e.g. ``True``, ``en``, ``unclear``) is magenta

0.1.16 (2026-04-09)
-------------------

* Add ``desired_filename`` and ``downloaded`` columns to ``_results.csv``; ``desired_filename`` is pre-computed using ``_make_structured_filename`` at search time, ``downloaded`` is initialised to ``False``
* Rename ``s2_paper_id`` column to ``sem_scholar_paper_id``
* Request additional Semantic Scholar fields: ``referenceCount``, ``citationCount``, ``fieldsOfStudy``, ``citationStyles``, ``publicationTypes``; all five are stored in ``_results_raw.json``
* Fileds ``referenceCount`` and ``citationCount`` are added as CSV columns
* Update ``papers download --search-file`` to read ``desired_filename`` from the CSV instead of recomputing it, and to write ``downloaded=True`` back to ``_results.csv`` for each successfully downloaded paper
* Fix name collision: rename CLI command function ``filter`` to ``filter_results``
* Change ``_make_structured_filename`` output format from ``YEAR_TITLE_DOI.pdf`` to ``YEAR_AUTHOR_TITLE_DOI.pdf``. Title and DOI are lowercased (title is truncated to 60 characters)

0.1.15 (2026-04-09)
-------------------

* Add ``self user-agent`` command to print the HTTP User-Agent string used by llmexer for API requests (example: llmexer/<version> (python-request/<version>))

0.1.14 (2026-04-09)
-------------------

* Add ``self`` command group with two subcommands: ``self version`` (prints the current package version)
* ``self envs`` (displays llmexer-relevant environment variables as a Rich table)

0.1.13 (2026-04-09)
-------------------

* Add ``search filter`` command: filters ``<ID>_results.csv`` by ``--language`` (default: ``en``) and saves matching rows to ``<ID>_filtered.csv``; prints total, filtered-out, and remaining counts on separate lines
* Modify ``search stats`` to respect filtered CSV

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
