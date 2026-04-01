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

##  Configuration

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

## Usage

Once installed, the `llmexer` command is available directly in your terminal.

Because it's a typer application, you can explore all its commands and options by simply running:

* Help
    ```
    llmexer --help
    ```

Here is a list of examples demonstrating the core features of the utility:

* List version
    ```
    llmexer version
    ```

## Getting Started

* Create a new experiment — generates a uniquely named folder under `.experiments/` using the format `YYYYMMDD-GUID`:
    ```
    llmexer experiment create
    ```
    The alias `exp` can be used as a shorthand:
    ```
    llmexer exp create
    ```

### CLI category: experiment

The `experiment` (alias: `exp`) category provides commands for managing LLM experiments:

* **Create an experiment** — generates a uniquely named folder under `.experiments/` using the format `YYYYMMDD-GUID`:
    ```bash
    llmexer experiment create
    ```
    Or with a custom ID:
    ```bash
    llmexer experiment create --id my-custom-experiment
    ```

* **List experiments** — displays all experiments with sorting options:
    ```bash
    llmexer experiment list
    ```
    Sort by date (newest first):
    ```bash
    llmexer experiment list --sort-by date --desc
    ```

* **Show current experiment** — displays the current experiment ID from `.env`:
    ```bash
    llmexer experiment current
    ```

* **Rename an experiment** — changes the ID of an existing experiment:
    ```bash
    llmexer experiment rename --old-id old-experiment-name --new-id new-experiment-name
    ```
    If `EXPERIMENT_ID` is set in `.env`, you can omit `--old-id`:
    ```bash
    llmexer experiment rename --new-id new-experiment-name
    ```

### Using Current Experiment ID

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

### CLI category: search

The `search` category provides commands for managing and running literature searches using the Semantic Scholar API:

* **Create a search configuration** — generates a YAML file with search parameters:
    ```bash
    llmexer search new --query "machine learning"
    ```
    This creates a file like `search_20260401-abc123.yaml` in the experiment's `searches` folder with:
    ```yaml
    query: machine learning
    year: 2020-2026
    onlyOpenAccess: false
    ```

* **Run a search with direct query**:
    ```bash
    llmexer search run --query "neural networks" --limit 200
    ```
    This queries Semantic Scholar and saves results to CSV files in the experiment's `searches` folder.

* **Run a search from a config file**:
    ```bash
    llmexer search run --file search_20260401-abc123.yaml --limit 500
    ```

* **Search results**:
    - Papers are fetched in batches of 100 (API limit)
    - Results are saved incrementally every 100 papers: `search_YYYYMMDD-GUID_partial_100.csv`, `search_YYYYMMDD-GUID_partial_200.csv`, etc.
    - Final results saved to: `search_YYYYMMDD-GUID_final.csv`
    - CSV fields: `DOI`, `TITLE`, `AUTHORS`, `ABSTRACT`, `IsOpenAccess`, `Year`, `PaperId`

## CLI UI
![alt text](docs/cli-ui.png)


## Documentation

[Documentation](https://vdmitriyev.github.io/llmexer/)

## License

[MIT](https://github.com/vdmitriyev/llmexer/blob/main/LICENSE)
