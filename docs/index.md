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

First, [install `uv`](https://docs.astral.sh/uv/getting-started/installation):

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

Then install the `fact` package and its dependencies:

```bash
uv sync
```

## Quick Start

To run the included CLI:

```bash
uv run fact 3
```

To use `fact` as a library within your project, import the `factorial` function and execute the
API like:

*[API]: Application Programming Interface

```python
from fact.lib import factorial

assert factorial(3) == 6  # (1)!
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
