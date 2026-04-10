## About

> `llmexer` `llmexer` is a framework and CLI utility to plan, design, run and control various LLM experiments


## Installation

* (option 1) Clone this repo and follow `development setup`

##  Development Setup

This guide walks through setting up the project for local development using `uv`.

1. Create a new virtual environment in a `.venv` directory and activates it.
    ```bash
    uv venv
    ```
1. Activate the environment (macOS/Linux):
   ```
   source .venv/bin/activate
   ```
1. Activate the environment (Windows):
    ```
    call .venv/Scripts/activate.bat
    ```
1.  Install package in **editable mode** with **dev** dependencies
    Installing the package in **editable mode** (`-e`) is the key to development. It links the `llmexer` command in your environment directly to your source code.
    ```bash
    uv pip install -e . --group dev
    ```

## ⚙️ Configuration

This tool requires access to local or remote running LLMS. It uses a `.env` file to securely load your API credentials, and also set the current experiment.

1.  Create a `.env` file in the root of the project
2.  Set the current experiment ID (optional):
    ```
    EXPERIMENT_ID=20260330-3a9adf70
    ```
3.  Configure the docling backend (optional, used by `papers extract --processor docling`):
    ```
    DOCLING_URL=http://localhost:5001/
    DOCLING_USER=myuser
    DOCLING_PASSWORD=mypassword
    ```
4. You could also pass a custom file as `.env` to the CLI:
    ```
    llmexer --env-file custom.env
    ```

## 🚀 Getting Started


Because it's a typer CLI application, you can explore all its commands and options by simply running.
```bash
llmexer --help
```

A typical end-to-end workflow for collecting and processing papers inside an experiment:

**1. Create a new experiment**
```bash
llmexer experiment create
# Output: created experiment '20260402-a1b2c3d4'
```

**2. Give it a meaningful name**
```bash
llmexer experiment rename --old-id 20260402-a1b2c3d4 --new-id llm-survey-2026
```

**3. Initialise the experiment structure (optional)**

Scaffold a standard `experiment/` subfolder with template CSVs and a prompt file:
```bash
llmexer experiment init --eid llm-survey-2026
```
This creates:
- `experiment/models.csv` — list of models to use (name, provider, notes); pre-filled with `llama3.3:latest`, `phi4:14b`, `gemma3:12b`, `gemma3:27b`
- `experiment/data.csv` — input data rows (ID, Title, Abstract)
- `experiment/mapping.csv` — maps data IDs to prompt IDs; pre-filled with `D01;prompt01` and `D02;prompt01`
- `experiment/prompts/prompt01.txt` — a starter Jinja2 prompt template using `{{title}}` and `{{abstract}}`

**4. Add papers to the experiment**

From a local file:
```bash
llmexer papers add --eid llm-survey-2026 --file ~/Downloads/attention-is-all-you-need.pdf
```
From a directory of PDFs:
```bash
llmexer papers add --eid llm-survey-2026 --directory ~/Downloads/papers/
```
From a URL:
```bash
llmexer papers add --eid llm-survey-2026 --url https://arxiv.org/pdf/1706.03762
```

**5. Run a literature search**

Create a search configuration and run it:
```bash
llmexer search create --eid llm-survey-2026 --query "large language models"
llmexer search run --eid llm-survey-2026 --file 20260401-bfdd863d.yaml
```

Or run directly from a query string:
```bash
llmexer search run --eid llm-survey-2026 --query "large language models" --limit 500
```

Results are saved as `<ID>_results.csv` and `<ID>_results_raw.json` in `searches/`.

**6. Filter search results by language (optional)**

Filter to English-only papers before downloading (default language is `en`):
```bash
llmexer search filter --eid llm-survey-2026 --file 20260401-bfdd863d.yaml
```
This produces `20260401-bfdd863d_filtered.csv` with only the matching rows.

**7. Download open-access papers by DOI via Unpaywall**

By DOI (one or more):
```bash
llmexer papers download --eid llm-survey-2026 --doi 10.1038/nature12373 --email you@example.com
```
From a full search result CSV (downloads all papers with a DOI, names each file `YEAR_AUTHOR_TITLE_DOI.pdf`):
```bash
llmexer papers download --eid llm-survey-2026 --search-file 20260401-bfdd863d_results.csv
```
Or from a filtered CSV to download only the papers that passed the language filter:
```bash
llmexer papers download --eid llm-survey-2026 --search-file 20260401-bfdd863d_filtered.csv
```
Failed downloads are saved automatically as `20260401-bfdd863d_results_download_failed.csv` (columns: `doi`, `url`, `title`, `desired_filename`, `downloaded`) next to the source CSV.

**8. Extract text from all added papers**

Using the default `pypdf` backend (saves `.txt` files):
```bash
llmexer papers extract --eid llm-survey-2026
```

Using the `docling` backend for richer Markdown output (saves `.md` files), reading connection details from `.env`:
```bash
llmexer papers extract --eid llm-survey-2026 --processor docling
```

Override `.env` connection settings at runtime:
```bash
llmexer papers extract --eid llm-survey-2026 --processor docling \
  --docling-url http://myserver:5001/ \
  --docling-user admin \
  --docling-password secret
```

By default, papers that already have an extracted file are skipped. Use `--rewrite` to force re-extraction:
```bash
llmexer papers extract --eid llm-survey-2026 --rewrite
```

**9. Generate rendered prompts for all data-model combinations**

After filling in `experiment/models.csv`, `experiment/data.csv`, `experiment/mapping.csv`, and your Jinja2 prompt templates:
```bash
llmexer experiment generate --eid llm-survey-2026
```
This renders every (data row, prompt, model) combination and writes the results to `experiment/experiment_YYYYMMDD-GUID.csv`. Use `--dry-run` to preview the row count without writing:
```bash
llmexer --dry-run experiment generate --eid llm-survey-2026
```

## 🧪 CLI category: experiment

The `experiment` (alias: `exp`) category provides commands for managing LLM experiments:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `create` | Create a new experiment folder under `.experiments/` using format `YYYYMMDD-GUID`. Accepts an optional custom ID. | `llmexer experiment create --id my-experiment` |
| `init` | Initialise an existing experiment with a standard folder structure (`experiment/`, `experiment/prompts/`) and template files: `models.csv` (pre-filled with 4 ollama models), `data.csv`, `mapping.csv` (pre-filled with D01 and D02 rows), `prompts/prompt01.txt` (Jinja2 template using `{{title}}` and `{{abstract}}`). Raises an error if already initialised. | `llmexer experiment init --eid my-experiment` |
| `generate` | Render all (data row × prompt × model) combinations defined in `experiment/` and write the result to `experiment/experiment_YYYYMMDD-GUID.csv`. Output columns: `ID`, `code`, `prompt`, `original_data`, `model_name`, `provider_name`, `prompt_hash`, `original_data_hash`, `json_params`. Rows are sorted by model order from `models.csv`. Supports `--dry-run`. | `llmexer experiment generate --eid my-experiment` |
| `list` | List all experiments with optional sorting by name or date. | `llmexer experiment list --sort-by date --desc` |
| `current` | Display the current experiment ID loaded from `.env`. | `llmexer experiment current` |
| `rename` | Rename an existing experiment. Uses `EXPERIMENT_ID` from `.env` if `--old-id` is omitted. | `llmexer experiment rename --old-id old-name --new-id new-name` |

#### Using Current Experiment ID

Many commands support the `--eid` parameter to specify which experiment to work with. If you set `EXPERIMENT_ID` in your `.env` file, you can omit this parameter and the commands will use the current experiment automatically:

```bash
# Set in .env
EXPERIMENT_ID=my-experiment

# These commands will use my-experiment automatically
llmexer search run --query "machine learning"
llmexer papers rename
```

You can still override the current experiment by explicitly providing `--eid`:
```bash
llmexer search run --eid different-experiment --query "deep learning"
```

## 📑 CLI category: papers

The `papers` category provides commands for managing PDF papers within an experiment:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `add --file` | Copy a single PDF into the experiment's `papers/` folder. | `llmexer papers add --file /path/to/paper.pdf` |
| `add --directory` | Recursively copy all PDFs from a directory. Already-existing papers are skipped. | `llmexer papers add --directory /path/to/folder` |
| `add --url` | Download a PDF from a URL into the experiment's `papers/` folder. | `llmexer papers add --url https://example.com/paper.pdf` |
| `download --doi` | Download one or more open-access PDFs by DOI using the Unpaywall API. Email required via `--email` or `UNPAYWALL_EMAIL` env var. | `llmexer papers download --doi 10.1038/nature12373 --email you@example.com` |
| `download --search-file` | Download all papers from a search result CSV (inside `searches/`), including filtered CSVs (`_filtered.csv`). Files are named `YEAR_AUTHOR_TITLE_DOI.pdf`. On success, updates `downloaded=True` in the source CSV. Failures saved as `<stem>_download_failed.csv`. | `llmexer papers download --search-file 20260401-abc123_filtered.csv` |
| `extract` | Extract text from all PDFs in `papers/`. Default `pypdf` backend saves `.txt`; `docling` backend sends PDFs to a remote docling-serve instance and saves `.md`. Connection details (`DOCLING_URL`, `DOCLING_USER`, `DOCLING_PASSWORD`) read from `.env`; overridable via `--docling-url`, `--docling-user`, `--docling-password`. Already-extracted files are skipped unless `--rewrite` is passed. | `llmexer papers extract --eid my-experiment --processor docling` |

## 🔍 CLI category: search

The `search` category provides commands for managing and running literature searches using the Semantic Scholar bulk API:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `create` | Create a search configuration YAML file in the experiment's `searches/` folder. | `llmexer search create --query "machine learning"` |
| `run --query` | Run a search directly from a query string. Saves `<ID>_results_raw.json` and `<ID>_results.csv` to `searches/`. CSV columns include: `sem_scholar_paper_id`, `year`, `title`, `authors`, `abstract`, `isOpenAccess`, `doi`, `language`, `referenceCount`, `citationCount`, `entry_source`, `pdf_filename`, `txt_filename`, `markdown_filename`, `pdf_downloaded`. Raw JSON also contains `fieldsOfStudy`, `citationStyles`, `publicationTypes`. | `llmexer search run --query "neural networks" --limit 200` |
| `run --file` | Run a search loading parameters from an existing YAML config. Use `--rewrite` to overwrite existing result files. | `llmexer search run --file 20260401-abc123.yaml` |
| `stats` | Display statistics for a completed search: papers per year and a stats breakdown (open access, language, downloaded, entry source, txt/markdown presence), stacked for results and filtered CSVs. | `llmexer search stats --file 20260401-abc123.yaml` |
| `filter` | Filter `<ID>_results.csv` by language (default: `en`) and save matching rows to `<ID>_filtered.csv`. Prints total entries, filtered-out count, and remaining count. | `llmexer search filter --file 20260401-abc123.yaml --language en` |
| `sync` | Reconcile `<ID>_results.csv` (and `<ID>_filtered.csv` if present) against the experiment's `papers/` folder. Updates `pdf_downloaded`, `txt_filename`, and `markdown_filename` for existing rows; appends new rows for PDFs not yet listed (marked `entry_source="manually added"`). Respects `--dry-run`. | `llmexer search sync --file 20260401-abc123.yaml` |

Semantic Scholar API Documentation: [Paper bulk search](https://api.semanticscholar.org/api-docs/#tag/Paper-Data/operation/get_graph_paper_bulk_search) -> this can be used to formulate more sophisticated query string

## 🔎 CLI category: self

The `self` category provides introspection commands for the llmexer CLI itself:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `version` | Print the current llmexer package version. | `llmexer self version` |
| `envs` | Display all llmexer-relevant environment variables as a table. `EXPERIMENT_ID` is highlighted in bold cyan; `DOCLING_PASSWORD` is masked as `********` when set. | `llmexer self envs` |

## CLI UI
![alt text](docs/cli-ui.png)


## Documentation

[Documentation](https://vdmitriyev.github.io/llmexer/)

## License

[MIT](https://github.com/vdmitriyev/llmexer/blob/main/LICENSE)
