---
icon: material/math-integral
status: new
---

# `fact` User Guide

??? info "`python-blueprint` Project"

    For more information on how this was built and deployed, as well as other Python best
    practices, see [`python-blueprint`](https://github.com/johnthagen/python-blueprint).

!!! info

    This user guide is purely an illustrative example that shows off several features of 
    [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) and included Markdown
    extensions[^1].

[^1]: See `python-blueprint`'s `mkdocs.yml` for how to enable these features.

## Installation

First, [install pipx](https://pypa.github.io/pipx/):

=== "Linux"

    ```bash
    python3 -m pip install --user pipx
    python3 -m pipx ensurepath
    ```

=== "Windows"

    > If you installed Python using Microsoft Store, replace `py` with `python3` in the next line.

    ```powershell
    py -m pip install --user pipx
    ```

    If you receive

    ```
    WARNING: The script pipx.exe is installed in `<USER folder>\AppData\Roaming\Python\Python3x\Scripts` which is not on PATH
    ```

    Go into the mentioned folder and run

    ```
    pipx ensurepath
    ```

=== "macOS"

    ```bash
    brew install pipx
    pipx ensurepath
    ```

Next, [install Poetry](https://python-poetry.org/docs/#installation):

```bash
pipx install poetry
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

To use `fact` within your project, import the `factorial` function and execute the API like:

*[API]: Application Programming Interface

```python
from fact.lib import factorial

assert factorial(3) == 6 # (1)!
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
