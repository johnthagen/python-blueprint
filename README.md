# python-blueprint

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![Code style: black][black-badge]](https://github.com/psf/black)
[![Imports: isort][isort-badge]](https://pycqa.github.io/isort/)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[isort-badge]: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336

Example Python project that demonstrates how to create a tested Python package using the latest
Python testing and linting tooling. The project contains a `fact` package that provides a simple
implementation of the [factorial algorithm](https://en.wikipedia.org/wiki/Factorial) (`fact.lib`)
and a command line interface (`fact.cli`).

## Requirements

Python 3.6+.

> **Note**
>
> Because [Python 2.7 support ended January 1, 2020](https://pythonclock.org/), new projects 
> should consider supporting Python 3 only, which is simpler than trying to support both. As a 
> result, support for Python 2.7 in this example project has been dropped.

## Windows Support

Summary: On Windows, use `py` instead of `python3` for many of the examples in this documentation.

This package fully supports Windows, along with Linux and macOS, but Python is
typically [installed differently on Windows](https://docs.python.org/3/using/windows.html). Windows
users typically access Python through the [py](https://www.python.org/dev/peps/pep-0397/) launcher
rather than a `python3` link in their `PATH`. Within a virtual environment, all platforms operate
the same and use a `python` link to access the Python version used in that virtual environment.

## Dependencies

Dependencies are defined in:

- `requirements.in`
- `requirements.txt`
- `dev-requirements.in`
- `dev-requirements.txt`

### Virtual Environments

It is best practice during development to create an
isolated [Python virtual environment](https://docs.python.org/3/library/venv.html) using the `venv`
standard library module. This will keep dependant Python packages from interfering with other
Python projects on your system.

On *Nix:

```bash
# On Python 3.9+, add --upgrade-deps
$ python3 -m venv venv
$ source venv/bin/activate
```

On Windows `cmd`:

```bash
> py -m venv venv
> venv\Scripts\activate.bat
```

Once activated, it is good practice to update core packaging tools (`pip`, `setuptools`,
and `wheel`) to the latest versions.

```bash
(venv) $ python -m pip install --upgrade pip setuptools wheel
```

### (Applications Only) Locking Dependencies

This project uses [pip-tools](https://github.com/jazzband/pip-tools) to lock project dependencies
and create reproducible virtual environments.

**Note:** *Library* projects should not lock their `requirements.txt`. Since `python-blueprint`
also has a CLI application, this end-user application example is used to demonstrate how to lock
application dependencies.

To update dependencies:

```bash
(venv) $ python -m pip install pip-tools
(venv) $ python -m piptools compile --upgrade requirements.in
(venv) $ python -m piptools compile --upgrade dev-requirements.in
```

After upgrading dependencies, run the unit tests as described in the [Unit Testing](#unit-testing)
section to ensure that none of the updated packages caused incompatibilities in the current
project.

### Syncing Virtual Environments

To cleanly install your dependencies into your virtual environment:

```bash
(venv) $ python -m piptools sync requirements.txt dev-requirements.txt
```

## Packaging

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by:

- `pyproject.toml`
- `setup.py`
- `MANIFEST.in`

To package the project as both a 
[source distribution](https://docs.python.org/3/distutils/sourcedist.html) and
a [wheel](https://wheel.readthedocs.io/en/stable/):

```bash
(venv) $ python setup.py sdist bdist_wheel
```

This will generate `dist/fact-1.0.0.tar.gz` and `dist/fact-1.0.0-py3-none-any.whl`.

Read more about the [advantages of wheels](https://pythonwheels.com/) to understand why generating
wheel distributions are important.

### Upload Distributions to PyPI

Source and wheel redistributable packages can
be [uploaded to PyPI](https://packaging.python.org/tutorials/packaging-projects/) or installed
directly from the filesystem using `pip`.

To upload to PyPI:

```bash
(venv) $ python -m pip install twine
(venv) $ twine upload dist/*
```

## Testing

Automated testing is performed using [tox](https://tox.readthedocs.io/en/latest/index.html). tox
will automatically create virtual environments based on `tox.ini` for unit testing, PEP8 style
guide checking, and documentation generation.

```bash
# Run all environments.
#   To only run a single environment, specify it like: -e lint
# Note: tox is installed into the virtual environment automatically by ``piptools sync``
# command above.
(venv) $ tox
```

### Unit Testing

Unit testing is performed with [pytest](https://pytest.org/). pytest has become the defacto Python
unit testing framework. Some key advantages over the built
in [unittest](https://docs.python.org/3/library/unittest.html) module are:

1. Significantly less boilerplate needed for tests.
2. PEP8 compliant names (e.g. `pytest.raises()` instead of `self.assertRaises()`).
3. Vibrant ecosystem of plugins.

pytest will automatically discover and run tests by recursively searching for folders and `.py`
files prefixed with `test` for any functions prefixed by `test`.

The `tests` folder is created as a Python package (i.e. there is an `__init__.py` file within it)
because this helps `pytest` uniquely namespace the test files. Without this, two test files cannot
be named the same, even if they are in different sub-directories.

Code coverage is provided by the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin.

When running a unit test tox environment (e.g. `tox -e py39`), an HTML report is generated in
the `htmlcov` folder showing each source file and which lines were executed during unit testing.
Open `htmlcov/index.html` in a web browser to view the report. Code coverage reports help identify
areas of the project that are currently not tested.

Code coverage is configured in `pyproject.toml`.

To pass arguments to `pytest` through `tox`:

```bash
(venv) $ tox -e py39 -- -k invalid_factorial
```

### Code Style Checking

[PEP8](https://www.python.org/dev/peps/pep-0008/) is the universally accepted style guide for
Python code. PEP8 code compliance is verified using [flake8](http://flake8.pycqa.org/). flake8 is
configured in the `[flake8]` section of `tox.ini`. Extra flake8 plugins are also included:

- `pep8-naming`: Ensure functions, classes, and variables are named with correct casing.

### Automated Code Formatting

Code is automatically formatted using [black](https://github.com/psf/black). Imports are
automatically sorted and grouped using [isort](https://github.com/PyCQA/isort/).

These tools are configured by:

- `pyproject.toml`

To automatically format code, run:

```bash
(venv) $ tox -e fmt
```

To verify code has been formatted, such as in a CI job:

```bash
(venv) $ tox -e fmt-check
```

### Generated API Documentation

API Documentation for the `fact` Python project modules is automatically
generated using a [Sphinx](http://sphinx-doc.org/) tox environment. Sphinx is a documentation
generation tool that is the defacto tool for Python API documentation. Sphinx uses
the [RST](https://www.sphinx-doc.org/en/latest/usage/restructuredtext/basics.html) markup language.

This project uses
the [napoleon](http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) plugin for
Sphinx, which renders Google-style docstrings. Google-style docstrings provide a good mix of
easy-to-read docstrings in code as well as nicely-rendered output.

```python
"""Computes the factorial through a recursive algorithm.

Args:
    n: A positive input value.

Raises:
    InvalidFactorialError: If n is less than 0.

Returns:
    Computed factorial.
"""
```

The Sphinx project is configured in `docs/api/conf.py`.

This project uses the [furo](https://pradyunsg.me/furo/) Sphinx theme for its elegant, simple to
use, dark theme.

Build the docs using the `docs-api` tox environment (e.g. `tox` or `tox -e docs-api`). Once built,
open `docs/api/_build/index.html` in a web browser.

To configure Sphinx to automatically rebuild when it detects changes, run `tox -e docs-api-serve`
and open <http://127.0.0.1:8000> in a browser.

#### Generate a New Sphinx Project

To generate the Sphinx project shown in this project:

```bash
# Note: Sphinx is installed into the virtual environment automatically by ``piptools sync``
# command above.
(venv) $ mkdir -p docs/api
(venv) $ cd docs/api
(venv) $ sphinx-quickstart --no-makefile --no-batchfile --extensions sphinx.ext.napoleon
# When prompted, select all defaults.
```

Modify `conf.py` appropriately:

```python
# Add the project's Python package to the path so that autodoc can find it.
import os
import sys
sys.path.insert(0, os.path.abspath("../../src"))
```

### Generating a User Guide

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) is a powerful static site
generator that combines easy-to-write Markdown, with a number of Markdown extensions that increase
the power of Markdown. This makes it a great fit for user guides and other technical documentation.

The example MkDocs project included in this project is configured to allow the built documentation
to be hosted at any URL or viewed offline from the file system.

To build the user guide, run `tox -e docs-user-guide`. Open `docs/user_guide/site/index.html` using
a web browser.

To build and serve the user guide with automatic rebuilding as you change the contents,
run `tox -e docs-user-guide-serve` and open <http://127.0.0.1:8000> in a browser.

Each time the `master` Git branch is updated, the `.github/workflows/pages.yml` GitHub Action will
automatically build the user guide and publish it to [GitHub Pages](https://pages.github.com/).
This is configured in the `docs-user-guide-github-pages` `tox` environment. This hosted user guide
can be viewed at <https://johnthagen.github.io/python-blueprint/>

### Continuous Integration

Continuous integration is provided by [GitHub Actions](https://github.com/features/actions). This
runs all tests and lints for every commit and pull request to the repository.

GitHub Actions is configured in `.github/workflows/python.yml` and `tox.ini` using
the [tox-gh-actions plugin](https://github.com/ymyzk/tox-gh-actions).

Project Structure
-----------------

Traditionally, Python projects place the source for their packages in the root of the project
structure, like:

``` {.sourceCode .}
fact
├── fact
│   ├── __init__.py
│   ├── cli.py
│   └── lib.py
├── tests
│   ├── __init__.py
│   └── test_fact.py
├── tox.ini
└── setup.py
```

However, this structure
is [known](https://docs.pytest.org/en/latest/goodpractices.html#tests-outside-application-code) to
have bad interactions with `pytest` and `tox`, two standard tools maintaining Python projects. The
fundamental issue is that tox creates an isolated virtual environment for testing. By installing
the distribution into the virtual environment, `tox` ensures that the tests pass even after the
distribution has been packaged and installed, thereby catching any errors in packaging and
installation scripts, which are common. Having the Python packages in the project root subverts
this isolation for two reasons:

1. Calling `python` in the project root (for example, `python -m pytest tests/`) 
   [causes Python to add the current working directory](https://docs.pytest.org/en/latest/pythonpath.html#invoking-pytest-versus-python-m-pytest) (
   the project root) to `sys.path`, which Python uses to find modules. Because the source
   package `fact` is in the project root, it shadows the `fact` package installed in the tox
   environment.
2. Calling `pytest` directly anywhere that it can find the tests will also add the project root
   to `sys.path` if the `tests` folder is a a Python package (that is, it contains a `__init__.py`
   file).
   [pytest adds all folders containing packages](https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery)
   to `sys.path` because it imports the tests like regular Python modules.

In order to properly test the project, the source packages must not be on the Python path. To
prevent this, there are three possible solutions:

1. Remove the `__init__.py` file from `tests` and run `pytest` directly as a tox command.
2. Remove the `__init__.py` file from tests and change the working directory of `python -m pytest`
   to `tests`.
3. Move the source packages to a dedicated `src` folder.

The dedicated `src` directory is the 
[recommended solution](https://docs.pytest.org/en/latest/pythonpath.html#test-modules-conftest-py-files-inside-packages)
by `pytest` when using tox and the solution this blueprint promotes because it is the least brittle
even though it deviates from the traditional Python project structure. It results is a directory
structure like:

``` {.sourceCode .}
fact
├── src
│   └── fact
│       ├── __init__.py
│       ├── cli.py
│       └── lib.py
├── tests
│   ├── __init__.py
│   └── test_fact.py
├── tox.ini
└── setup.py
```

Type Hinting
------------

[Type hinting](https://docs.python.org/3/library/typing.html) allows developers to include optional
static typing information to Python source code. This allows static analyzers such
as [PyCharm](https://www.jetbrains.com/pycharm/), [mypy](http://mypy-lang.org/),
or [Pyre](https://pyre-check.org/) to check that functions are used with the correct
types before runtime.

For [PyCharm in particular](https://www.jetbrains.com/help/pycharm/type-hinting-in-product.html),
the IDE is able to provide much richer auto-completion, refactoring, and type checking while the
user types, resulting in increased productivity and correctness.

This project uses the type hinting syntax introduced in Python 3:

```python
def factorial(n: int) -> int:
```

Type checking is performed by mypy via `tox -e type-check`. mypy is configured in `pyproject.toml`.

See also [awesome-python-typing](https://github.com/typeddjango/awesome-python-typing).

### Distributing Type Hints

[PEP 561](https://www.python.org/dev/peps/pep-0561/) defines how a Python package should
communicate the presence of inline type hints to static type
checkers. [mypy's documentation](https://mypy.readthedocs.io/en/stable/installed_packages.html)
provides further examples on how to do this as well.

Mypy looks for the existence of a file named `py.typed` in the root of the installed package to
indicate that inline type hints should be checked.

## Licensing

Licensing for the project is defined in:

- `LICENSE.txt`
- `setup.py`

This project uses a common permissive license, the MIT license.

You may also want to list the licenses of all of the packages that your Python project depends on.
To automatically list the licenses for all dependencies in `requirements.txt` (and their transitive
dependencies) using [pip-licenses](https://github.com/raimon49/pip-licenses):

```bash
(venv) $ tox -e licenses
...
 Name        Version  License
 colorama    0.4.3    BSD License
 exitstatus  1.3.0    MIT License
```

## Docker

[Docker](https://www.docker.com/) is a tool that allows for software to be packaged into isolated
containers. It is not necessary to use Docker in a Python project, but for the purposes of
presenting best practice examples, a Docker configuration is provided in this project. The Docker
configuration in this repository is optimized for small size and increased security, rather than
simplicity.

Docker is configured in:

- `Dockerfile`
- `.dockerignore`

To build the Docker image:

```bash
$ docker build --tag fact .
```

To run the image in a container:

```bash
# Example calculating the factorial of 5.
$ docker run --rm --interactive --tty fact 5
```

## PyCharm Configuration

To configure PyCharm 2018.3 and newer to align to the code style used in this project:

- Settings | Search "Hard wrap at"
  - Editor | Code Style | General | Hard wrap at: 99

- Settings | Search "Optimize Imports"
  - Editor | Code Style | Python | Imports
      - ☑ Sort import statements
        - ☑ Sort imported names in "from" imports
        - ☐ Sort plain and "from" imports separately within a group
        - ☐ Sort case-insensitively
      - Structure of "from" imports
        - ◎ Leave as is
        - ◉ Join imports with the same source
        - ◎ Always split imports

- Settings | Search "Docstrings"
  - Tools | Python Integrated Tools | Docstrings | Docstring Format: Google

- Settings | Search "pytest"
  - Tools | Python Integrated Tools | Testing | Default test runner: pytest

- Settings | Search "Force parentheses"
  - Editor | Code Style | Python | Wrapping and Braces | "From" Import Statements
    - ☑ Force parentheses if multiline

### Integrate Code Formatters

To integrate automatic code formatters into PyCharm, reference the following instructions:

- [black integration](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea)
  - The File Watchers method (step 6) is recommended. This will run `black` on every save.

- [isort integration](https://github.com/timothycrosley/isort/wiki/isort-Plugins)
    - The File Watchers method (option 1) is recommended. This will run `isort` on every save.

> **Tip**
>
> These tools work best if you properly mark directories as excluded from the project that should 
> be, such as `.tox`. See 
> <https://www.jetbrains.com/help/pycharm/project-tool-window.html#content_pane_context_menu> on 
> how to Right Click | Mark Directory as | Excluded.
