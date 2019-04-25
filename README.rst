python-blueprint
================

.. image:: https://travis-ci.org/johnthagen/python-blueprint.svg?branch=master
    :target: https://travis-ci.org/johnthagen/python-blueprint

Example Python project that demonstrates how to create a tested Python package using the latest
Python testing and linting tooling. The project contains a ``fact`` package that provides a
simple implementation of the `factorial algorithm <https://en.wikipedia.org/wiki/Factorial>`_
(``fact.lib``) and a command line interface (``fact.cli``).

Requirements
------------

Python 2.7 or 3.5+.

.. note::

    Because `Python 2.7 supports ends January 1, 2020 <https://pythonclock.org/>`_, new projects
    may want to consider supporting Python 3 only, which is simpler than trying to support both.
    Support for Python 2.7 in this example project is provided only for completeness and will
    be removed at a later date.

Windows Support
---------------

Summary: On Windows, use ``py`` instead of ``python3`` for many of the examples in this
documentation.

This package fully supports Windows, along with Linux and macOS, but Python is typically
`installed differently on Windows <https://docs.python.org/3/using/windows.html>`_.
Windows users typically access Python through the
`py <https://www.python.org/dev/peps/pep-0397/>`_ launcher rather than a ``python3``
link in their ``PATH``. Within a virtual environment, all platforms operate the same and use a
``python`` link to access the Python version used in that virtual environment.

Packaging
---------

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by:

- ``setup.py``

- ``MANIFEST.in``

To package the project:

.. code-block:: bash

    $ python3 setup.py sdist

This will generate ``dist/fact-1.0.0.tar.gz``. This redistributable package can be
`uploaded to PyPI <https://packaging.python.org/tutorials/packaging-projects/>`_ or installed
directly from the filesystem using ``pip``.

Testing
-------

Automated testing is performed using `tox <https://tox.readthedocs.io/en/latest/index.html>`_.
tox will automatically create virtual environments based on ``tox.ini`` for unit testing,
PEP8 style guide checking, and documentation generation.

.. code-block:: bash

    # Install tox (only needed once).
    $ python3 -m pip install tox

    # Run all environments.
    #   To only run a single environment, specify it like: -e pep8
    $ tox

Unit Testing
^^^^^^^^^^^^

Unit testing is performed with `pytest <https://pytest.org/>`_. pytest has become the defacto
Python unit testing framework. Some key advantages over the built in
`unittest <https://docs.python.org/3/library/unittest.html>`_ module are:

#. Significantly less boilerplate needed for tests.

#. PEP8 compliant names (e.g. ``pytest.raises()`` instead of ``self.assertRaises()``).

#. Vibrant ecosystem of plugins.

pytest will automatically discover and run tests by recursively searching for folders and ``.py``
files prefixed with ``test`` for any functions prefixed by ``test``.

The ``tests`` folder is created as a Python package (i.e. there is an ``__init__.py`` file
within it) because this helps ``pytest`` uniquely namespace the test files. Without this,
two test files cannot be named the same, even if they are in different sub-directories.

Code coverage is provided by the `pytest-cov <https://pytest-cov.readthedocs.io/en/latest/>`_
plugin.

When running a unit test tox environment (e.g. ``tox``, ``tox -e py37``, etc.), a data file
(e.g. ``.coverage.py37``) containing the coverage data is generated. This file is not readable on
its own, but when the ``coverage`` tox environment is run (e.g. ``tox`` or ``tox -e -coverage``),
coverage from all unit test environments is combined into a single data file and an HTML report is
generated in the ``htmlcov`` folder showing each source file and which lines were executed during
unit testing is generated in Open ``htmlcov/index.html`` in a web browser to view the report. Code
coverage reports help identify areas of the project that are currently not tested.

Code coverage is configured in the ``.coveragerc`` file.

Code Style Checking
^^^^^^^^^^^^^^^^^^^

`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ is the universally accepted style
guide for Python code. PEP8 code compliance is verified using `flake8 <http://flake8.pycqa.org/>`_.
flake8 is configured in the ``[flake8]`` section of ``tox.ini``. Three extra flake8 plugins
are also included:

- pep8-naming: Ensure functions, classes, and variables are named with correct casing.
- flake8-quotes: Ensure that ``' '`` style string quoting is used consistently.
- flake8-import-order: Ensure consistency in the way imports are grouped and sorted.

Generated Documentation
^^^^^^^^^^^^^^^^^^^^^^^

Documentation that includes the ``README.rst`` and the Python project modules is automatically
generated using a `Sphinx <http://sphinx-doc.org/>`_ tox environment. Sphinx is a documentation
generation tool that is the defacto tool for Python documentation. Sphinx uses the
`RST <https://www.sphinx-doc.org/en/latest/usage/restructuredtext/basics.html>`_ markup language.

This project uses the
`napoleon <http://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html>`_ plugin for
Sphinx, which renders Google-style docstrings. Google-style docstrings provide a good mix
of easy-to-read docstrings in code as well as nicely-rendered output.

.. code-block:: python

    """Computes the factorial through a recursive algorithm.

    Args:
        n: A positive input value.

    Raises:
        InvalidFactorialError: If n is less than 0.

    Returns:
        Computed factorial.
    """

The Sphinx project is configured in ``docs/conf.py``.

Build the docs using the ``docs`` tox environment (e.g. ``tox`` or ``tox -e docs``). Once built,
open ``docs/_build/index.html`` in a web browser.

Generate a New Sphinx Project
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To generate the Sphinx project shown in this project:

.. code-block:: bash

    $ mkdir docs
    $ cd docs
    $ sphinx-quickstart --no-makefile --no-batchfile --extensions sphinx.ext.napoleon
    # When prompted, select all defaults.

Modify ``conf.py`` appropriately:

.. code-block:: python

    # Add the project's Python package to the path so that autodoc can find it.
    import os
    import sys
    sys.path.insert(0, os.path.abspath('../src'))

    ...

    html_theme_options = {
        # Override the default alabaster line wrap, which wraps tightly at 940px.
        'page_width': 'auto',
    }

Modify ``index.rst`` appropriately:

::

    .. include:: ../../README.rst

    apidoc/modules.rst

Project Structure
-----------------

Traditionally, Python projects place the source for their packages in the root of the project
structure, like:

.. code-block::

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

However, this structure is `known
<https://docs.pytest.org/en/latest/goodpractices.html#tests-outside-application-code>`_ to have bad
interactions with ``pytest`` and ``tox``, two standard tools maintaining Python projects. The
fundamental issue is that tox creates an isolated virtual environment for testing. By installing
the distribution into the virtual environment, ``tox`` ensures that the tests pass even after the
distribution has been packaged and installed, thereby catching any errors in packaging and
installation scripts, which are common. Having the Python packages in the project root subverts
this isolation for two reasons:

#. Calling ``python`` in the project root (for example, ``python -m pytest tests/``) `causes Python
   to add the current working directory
   <https://docs.pytest.org/en/latest/pythonpath.html#invoking-pytest-versus-python-m-pytest>`_
   (the project root) to ``sys.path``, which Python uses to find modules. Because the source
   package ``fact`` is in the project root, it shadows the ``fact`` package installed in the tox
   environment.

#. Calling ``pytest`` directly anywhere that it can find the tests will also add the project root
   to ``sys.path`` if the ``tests`` folder is a a Python package (that is, it contains a
   ``__init__.py`` file). `pytest adds all folders containing packages
   <https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery>`_
   to ``sys.path`` because it imports the tests like regular Python modules.

In order to properly test the project, the source packages must not be on the Python path. To
prevent this, there are three possible solutions:

#. Remove the ``__init__.py`` file from ``tests`` and run ``pytest`` directly as a tox command.

#. Remove the ``__init__.py`` file from tests and change the working directory of
   ``python -m pytest`` to ``tests``.

#. Move the source packages to a dedicated ``src`` folder.

The dedicated ``src`` directory is the `recommended solution
<https://docs.pytest.org/en/latest/pythonpath.html#test-modules-conftest-py-files-inside-packages>`_
by ``pytest`` when using tox and the solution this blueprint promotes because it is the least
brittle even though it deviates from the traditional Python project structure. It results is a
directory structure like:

.. code-block::

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

Type Hinting
------------

`Type hinting <https://docs.python.org/3/library/typing.html>`_ allows developers to include
optional static typing information to Python source code. This allows static analyzers such
as `PyCharm <https://www.jetbrains.com/pycharm/>`_, `mypy <http://mypy-lang.org/>`_, or
`pytype <https://github.com/google/pytype>`_ to check that functions are used with the correct
types before runtime.

For
`PyCharm in particular <https://www.jetbrains.com/help/pycharm/type-hinting-in-product.html>`_,
the IDE is able to provide much richer auto-completion, refactoring, and type checking while
the user types, resulting in increased productivity and correctness.

This project uses the Python 2.7-compatible type hinting syntax:

.. code-block:: python

    def factorial(n):
    # type: (int) -> int


But Python 3-only projects should prefer the cleaner Python 3-only syntax:

.. code-block:: python

    def factorial(n: int) -> int:

Run CLI with ``pipenv``
-----------------------

.. note::

    There is some discussion in the Python community as to whether ``pipenv`` is a good
    long term solution. For one, it is only designed for application dependency management, not
    for libraries, so it can't replace ``setup.py`` for building an egg or ``requirements.txt``
    that is needed for ``tox``. Nevertheless, it is a useful tool and as such is mentioned here.

`pipenv <https://pipenv.readthedocs.io/en/latest/>`_ is a tool that combines virtual
environment creation and dependency installation into a single, easy-to-use interface.

To run the CLI application included in this project, first install pipenv.

Next, create a pipenv environment and launch a pipenv shell.

.. code-block:: bash

    $ pipenv install --dev
    $ pipenv shell
    (python-blueprint) $ fact -n 10

Regenerate Pipfile from requirements.txt
----------------------------------------

Since some information is duplicated in ``Pipfile`` and ``*requirements.txt``, the following
commands can be used to regenerate the ``Pipfile`` if new dependencies are added to
``requirements.txt``.

.. code-block:: bash

    $ pipenv install -e .
    $ pipenv install -r requirements.txt
    $ pipenv install -r dev-requirements.txt --dev

PyCharm Configuration
---------------------

To configure PyCharm 2018.3 and newer to align to the code style used in this project:

#. Settings | Search "Hard wrap at"

    #. Editor | Code Style | General | Hard wrap at: 99

#. Settings | Search "Optimize Imports"

    #. Editor | Code Style | Python | Imports

        #. **Check:** Sort import statements

        #. **Check:** Sort imported names in "from" imports

        #. **Uncheck:** Sort plain and "from" imports separately within a group

        #. **Check:** Sort case-insensitively

#. Settings | Search "Docstrings"

    #. Tools | Python Integrated Tools | Docstrings | Docstring Format: Google

#. (Optional) Settings | Search "Force parentheses"

    #. Editor | Code Style | Python | Wrapping and Braces | "From Import Statements:
       Check Force parentheses if multiline