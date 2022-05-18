# python-blueprint

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![Code style: black][black-badge]](https://github.com/psf/black)
[![Imports: isort][isort-badge]](https://pycqa.github.io/isort/)
[![Type checked with mypy][mypy-badge]](https://github.com/python/mypy)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/workflows/python/badge.svg
[black-badge]: https://img.shields.io/badge/code%20style-black-000000.svg
[isort-badge]: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
[mypy-badge]: https://img.shields.io/badge/type%20checked-mypy-blue.svg

Example Python project that demonstrates how to create a tested Python package using the latest
Python testing, linting, and type checking tooling. The project contains a `fact` package that
provides a simple implementation of the
[factorial algorithm](https://en.wikipedia.org/wiki/Factorial) (`fact.lib`) and a command line
interface (`fact.cli`).

# Requirements

Python 3.8+.

# Package Management

This package uses [Poetry](https://python-poetry.org/) to manage dependencies and
isolated [Python virtual environments](https://docs.python.org/3/library/venv.html).

To proceed, 
[install Poetry globally onto your system](https://python-poetry.org/docs/#installation).

## Dependencies

Dependencies are defined in [`pyproject.toml`](./pyproject.toml) and specific versions are locked
into [`poetry.lock`](./poetry.lock). This allows for exact reproducible environments across
all machines that use the project, both during development and in production.

To install all dependencies into an isolated virtual environment:

```bash
# Append --remove-untracked to remove any dependencies no longer in use.
$ poetry install
```

To activate the virtual environment that is automatically created by Poetry:

```bash
$ poetry shell
```

To upgrade all dependencies to their latest versions:

```bash
$ poetry update
```

## Packaging

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by:

- [`pyproject.toml`](./pyproject.toml)

To package the project as both a 
[source distribution](https://docs.python.org/3/distutils/sourcedist.html) and
a [wheel](https://wheel.readthedocs.io/en/stable/):

```bash
$ poetry build
```

This will generate `dist/fact-1.0.0.tar.gz` and `dist/fact-1.0.0-py3-none-any.whl`.

Read more about the [advantages of wheels](https://pythonwheels.com/) to understand why generating
wheel distributions are important.

## Publish Distributions to PyPI

Source and wheel redistributable packages can
be [published to PyPI](https://python-poetry.org/docs/cli#publish) or installed
directly from the filesystem using `pip`.

```bash
$ poetry publish
```

# Enforcing Code Quality

Automated code quality checks are performed using 
[Nox](https://nox.thea.codes/en/stable/) and
[`nox-poetry`](https://nox-poetry.readthedocs.io/en/stable/). Nox will automatically create virtual
environments and run commands based on [`noxfile.py`](./noxfile.py) for unit testing, PEP 8 style
guide checking, type checking and documentation generation.

> Note: `nox` is installed into the virtual environment automatically by the `poetry install`
> command above. Run `poetry shell` to activate the virtual environment.

```bash
# Run all default sessions. To only run a single session, append its name: -s lint
(fact) $ nox
```

## Unit Testing

Unit testing is performed with [pytest](https://pytest.org/). pytest has become the de facto Python
unit testing framework. Some key advantages over the built-in
[unittest](https://docs.python.org/3/library/unittest.html) module are:

1. Significantly less boilerplate needed for tests.
2. PEP 8 compliant names (e.g. `pytest.raises()` instead of `self.assertRaises()`).
3. Vibrant ecosystem of plugins.

pytest will automatically discover and run tests by recursively searching for folders and `.py`
files prefixed with `test` for any functions prefixed by `test`.

The `tests` folder is created as a Python package (i.e. there is an `__init__.py` file within it)
because this helps `pytest` uniquely namespace the test files. Without this, two test files cannot
be named the same, even if they are in different subdirectories.

Code coverage is provided by the [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/) plugin.

When running a unit test Nox session (e.g. `nox -s test`), an HTML report is generated in
the `htmlcov` folder showing each source file and which lines were executed during unit testing.
Open `htmlcov/index.html` in a web browser to view the report. Code coverage reports help identify
areas of the project that are currently not tested.

pytest and code coverage are configured in [`pyproject.toml`](./pyproject.toml).

To pass arguments to `pytest` through `nox`:

```bash
(fact) $ nox -s test -- -k invalid_factorial
```

## Code Style Checking

[PEP 8](https://peps.python.org/pep-0008/) is the universally accepted style guide for
Python code. PEP 8 code compliance is verified using [Flake8](http://flake8.pycqa.org/). Flake8 is
configured in the `[tool.flake8]` section of `pyproject.toml`. Extra Flake8 plugins are also
included:

- `flake8-bugbear`: Find likely bugs and design problems in your program.
- `flake8-broken-line`: Forbid using backslashes (`\`) for line breaks.
- `flake8-comprehensions`: Helps write better `list`/`set`/`dict` comprehensions.
- `pep8-naming`: Ensure functions, classes, and variables are named with correct casing.
- `pyproject-flake8`: Allow configuration of `flake8` through `pyproject.toml`.

Some code style settings are included in [`.editorconfig`](./.editorconfig) and will be
configured automatically in editors such as PyCharm.

## Automated Code Formatting

Code is automatically formatted using [black](https://github.com/psf/black). Imports are
automatically sorted and grouped using [isort](https://github.com/PyCQA/isort/).

These tools are configured by:

- [`pyproject.toml`](./pyproject.toml)

To automatically format code, run:

```bash
(fact) $ nox -s fmt
```

To verify code has been formatted, such as in a CI job:

```bash
(fact) $ nox -s fmt_check
```

## Type Checking

[Type annotations](https://docs.python.org/3/library/typing.html) allows developers to include
optional static typing information to Python source code. This allows static analyzers such
as [mypy](http://mypy-lang.org/), [PyCharm](https://www.jetbrains.com/pycharm/),
or [Pyright](https://github.com/microsoft/pyright) to check that functions are used with the
correct types before runtime.

For [PyCharm in particular](https://www.jetbrains.com/help/pycharm/type-hinting-in-product.html),
the IDE is able to provide much richer auto-completion, refactoring, and type checking while the
user types, resulting in increased productivity and correctness.

```python
def factorial(n: int) -> int:
    ...
```

Type checking is performed by mypy via `nox -s type_check`. mypy is configured in 
[`pyproject.toml`](./pyproject.toml).

See also [awesome-python-typing](https://github.com/typeddjango/awesome-python-typing).

### Distributing Type Annotations

[PEP 561](https://www.python.org/dev/peps/pep-0561/) defines how a Python package should
communicate the presence of inline type annotations to static type
checkers. [mypy's documentation](https://mypy.readthedocs.io/en/stable/installed_packages.html)
provides further examples on how to do this.

Mypy looks for the existence of a file named [`py.typed`](./src/fact/py.typed) in the root of the
installed package to indicate that inline type annotations should be checked.

## Continuous Integration

Continuous integration is provided by [GitHub Actions](https://github.com/features/actions). This
runs all tests, lints, and type checking for every commit and pull request to the repository.

GitHub Actions is configured in [`.github/workflows/python.yml`](./.github/workflows/python.yml).

# Documentation

## Generating a User Guide

[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) is a powerful static site
generator that combines easy-to-write Markdown, with a number of Markdown extensions that increase
the power of Markdown. This makes it a great fit for user guides and other technical documentation.

The example MkDocs project included in this project is configured to allow the built documentation
to be hosted at any URL or viewed offline from the file system.

To build the user guide, run `nox -s docs`. Open `docs/user_guide/site/index.html` using
a web browser.

To build and serve the user guide with automatic rebuilding as you change the contents,
run `nox -s docs_serve` and open <http://127.0.0.1:8000> in a browser.

Each time the `master` Git branch is updated, the 
[`.github/workflows/pages.yml`](.github/workflows/pages.yml) GitHub Action will
automatically build the user guide and publish it to [GitHub Pages](https://pages.github.com/).
This is configured in the `docs_github_pages` Nox session. This hosted user guide
can be viewed at <https://johnthagen.github.io/python-blueprint/>.

## Generating API Documentation

This project uses [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings) plugin for
MkDocs, which renders
[Google-style docstrings](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
into an MkDocs project. Google-style docstrings provide a good mix of easy-to-read docstrings in
code as well as nicely-rendered output.

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

# Project Structure

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
├── noxfile.py
└── pyproject.toml
```

However, this structure
is [known](https://docs.pytest.org/en/latest/goodpractices.html#tests-outside-application-code) to
have bad interactions with `pytest` and `nox`, two standard tools maintaining Python projects. The
fundamental issue is that Nox creates an isolated virtual environment for testing. By installing
the distribution into the virtual environment, `nox` ensures that the tests pass even after the
distribution has been packaged and installed, thereby catching any errors in packaging and
installation scripts, which are common. Having the Python packages in the project root subverts
this isolation for two reasons:

1. Calling `python` in the project root (for example, `python -m pytest tests/`) 
   [causes Python to add the current working directory](https://docs.pytest.org/en/latest/pythonpath.html#invoking-pytest-versus-python-m-pytest)
   (the project root) to `sys.path`, which Python uses to find modules. Because the source
   package `fact` is in the project root, it shadows the `fact` package installed in the Nox
   session.
2. Calling `pytest` directly anywhere that it can find the tests will also add the project root
   to `sys.path` if the `tests` folder is a Python package (that is, it contains a `__init__.py`
   file).
   [pytest adds all folders containing packages](https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery)
   to `sys.path` because it imports the tests like regular Python modules.

In order to properly test the project, the source packages must not be on the Python path. To
prevent this, there are three possible solutions:

1. Remove the `__init__.py` file from `tests` and run `pytest` directly as a Nox session.
2. Remove the `__init__.py` file from tests and change the working directory of `python -m pytest`
   to `tests`.
3. Move the source packages to a dedicated `src` folder.

The dedicated `src` directory is the 
[recommended solution](https://docs.pytest.org/en/latest/pythonpath.html#test-modules-conftest-py-files-inside-packages)
by `pytest` when using Nox and the solution this blueprint promotes because it is the least brittle
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
├── noxfile.py
└── pyproject.toml
```

# Licensing

Licensing for the project is defined in:

- [`LICENSE.txt`](./LICENSE.txt)
- [`pyproject.toml`](./pyproject.toml)

This project uses a common permissive license, the MIT license.

You may also want to list the licenses of all the packages that your Python project depends on.
To automatically list the licenses for all dependencies in (and their transitive dependencies)
using [pip-licenses](https://github.com/raimon49/pip-licenses):

```bash
(fact) $ nox -s licenses
...
 Name        Version  License
 colorama    0.4.3    BSD License
 exitstatus  1.3.0    MIT License
```

# Docker

[Docker](https://www.docker.com/) is a tool that allows for software to be packaged into isolated
containers. It is not necessary to use Docker in a Python project, but for the purposes of
presenting best practice examples, a Docker configuration is provided in this project. The Docker
configuration in this repository is optimized for small size and increased security, rather than
simplicity.

Docker is configured in:

- [`Dockerfile`](./Dockerfile)
- [`.dockerignore`](./.dockerignore)

To build the Docker image:

```bash
$ docker build --tag fact .
```

To run the image in a container:

```bash
# Example calculating the factorial of 5.
$ docker run --rm --interactive --tty fact 5
```

# PyCharm Configuration

> Looking for a vivid dark color scheme for PyCharm?
> Try [One Dark theme](https://plugins.jetbrains.com/plugin/11938-one-dark-theme).

To configure [PyCharm](https://www.jetbrains.com/pycharm/) and newer to align to the code style
used in this project:

- Settings | Search "Hard wrap at" (Note, this will be automatically set by
  [`.editorconfig`](./.editorconfig))
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

## Integrate Code Formatters

To integrate automatic code formatters into PyCharm, reference the following instructions:

- [black integration](https://black.readthedocs.io/en/stable/integrations/editors.html#pycharm-intellij-idea)
  - The File Watchers method (step 6) is recommended. This will run `black` on every save.

- [isort integration](https://github.com/timothycrosley/isort/wiki/isort-Plugins)
    - The File Watchers method (option 1) is recommended. This will run `isort` on every save.

> **Tip**
>
> These tools work best if you properly mark directories as excluded from the project that should 
> be, such as `.nox`. See 
> <https://www.jetbrains.com/help/pycharm/project-tool-window.html#content_pane_context_menu> on 
> how to Right-Click | Mark Directory as | Excluded.
