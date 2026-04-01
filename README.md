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

## CLI UI
![alt text](docs/cli-ui.png)


## Documentation

[Documentation](https://vdmitriyev.github.io/llmexer/)

## License

[MIT](https://github.com/vdmitriyev/llmexer/blob/main/LICENSE)
