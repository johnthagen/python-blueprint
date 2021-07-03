# `fact` User Guide

!!! info

    For more information on how this was built and deployed, as well as other Python best
    practices, see [`python-blueprint`](https://github.com/johnthagen/python-blueprint).

!!! info

    This user guide is purely an illustrative example that shows off several features of MkDocs
    and included Markdown extensions.

## Installation

First, create and activate a Python virtual environment:

=== "Linux/macOS"

    ```
    python3 -m venv venv
    source venv/bin/activate
    ```

=== "Windows"

    ```
    py -m venv venv
    venv\Scripts\activate
    ```

Then install the `fact` package:

```
pip install .
```

## Quick Start

To use `fact` within your project, import the `factorial` function and execute it like:

```python
from fact.lib import factorial

assert factorial(3) == 6
```

!!! tip

    Within PyCharm, use ++tab++ to auto-complete suggested imports while typing.
