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

This tool requires access to your GitLab repository (*scopes*: `api`, `read_api`). It uses a `.env` file to securely load your API credentials.

1.  Create a `.env` file in the root of the project

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

## CLI UI

![alt text](https://raw.githubusercontent.com/vdmitriyev/llmexer/main/docs/cli-ui.png)

## Documentation

[Documentation](https://vdmitriyev.github.io/llmexer/)

## License

[MIT](https://github.com/vdmitriyev/llmexer/blob/main/LICENSE)
