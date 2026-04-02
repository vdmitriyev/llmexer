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
3. You could also pass custom file as `.env`to the CLI:
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

**3. Add papers to the experiment**

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

**4. Extract text from all added papers**
```bash
llmexer papers extract --eid llm-survey-2026
# Creates .txt and files next to each PDF inside .experiments/llm-survey-2026/papers/
```

## 🧪 CLI category: experiment

The `experiment` (alias: `exp`) category provides commands for managing LLM experiments:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `create` | Create a new experiment folder under `.experiments/` using format `YYYYMMDD-GUID`. Accepts an optional custom ID. | `llmexer experiment create --id my-experiment` |
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
| `extract` | Extract text from all PDFs in `papers/` and save as `.txt` files. Skips unreadable PDFs with a warning. | `llmexer papers extract --eid my-experiment` |

## 🔍 CLI category: search

The `search` category provides commands for managing and running literature searches using the Semantic Scholar bulk API:

| Shortname | Description | Command Example |
|-----------|-------------|-----------------|
| `new` | Create a search configuration YAML file in the experiment's `searches/` folder. | `llmexer search new --query "machine learning"` |
| `run --query` | Run a search directly from a query string. Saves `<ID>_results_raw.json` and `<ID>_results.csv` to `searches/`. | `llmexer search run --query "neural networks" --limit 200` |
| `run --file` | Run a search loading parameters from an existing YAML config. Use `--force-overwrite` to overwrite existing result files. | `llmexer search run --file search_20260401-abc123.yaml` |
| `stats` | Display statistics for a completed search: publications per year and open access breakdown, shown as side-by-side tables. | `llmexer search stats --file search_20260401-abc123.yaml` |

## CLI UI
![alt text](docs/cli-ui.png)


## Documentation

[Documentation](https://vdmitriyev.github.io/llmexer/)

## License

[MIT](https://github.com/vdmitriyev/llmexer/blob/main/LICENSE)
