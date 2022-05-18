# `fact` User Guide

!!! info

    For more information on how this was built and deployed, as well as other Python best
    practices, see [`python-blueprint`](https://github.com/johnthagen/python-blueprint).

!!! info

    This user guide is purely an illustrative example that shows off several features of 
    [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and included Markdown
    extensions.

## Installation

First, [install Poetry](https://python-poetry.org/docs/#installation):

=== "Linux/macOS"

    ```bash
    curl -sSL https://install.python-poetry.org | python3 -
    ```

=== "Windows"

    ```powershell
    (Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -
    ```

Then install the `fact` package and its dependencies:

```bash
poetry install
```

Activate the virtual environment created automatically by Poetry:

```bash
poetry shell
```

## Quick Start

To use `fact` within your project, import the `factorial` function and execute it like:

```python
from fact.lib import factorial

# (1)
assert factorial(3) == 6
```

1. This assertion will be `True`

!!! tip

    Within PyCharm, use ++tab++ to auto-complete suggested imports while typing.

### Expected Results

<div class="center-table" markdown>

| Input | Output |
|:-----:|:------:|
|   1   |   1    |
|   2   |   2    |
|   3   |   6    |
|   4   |   24   | 

</div>
