python-blueprint
================

.. image:: https://travis-ci.com/johnthagen/python-blueprint.svg?branch=master
    :target: https://travis-ci.com/johnthagen/python-blueprint

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://black.readthedocs.io/en/stable/

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://timothycrosley.github.io/isort/

Example Python project that demonstrates how to create a tested Python package using the latest
Python testing and linting tooling. The project contains a ``fact`` package that provides a
simple implementation of the `factorial algorithm <https://en.wikipedia.org/wiki/Factorial>`_
(``fact.lib``) and a command line interface (``fact.cli``).

Requirements
------------

Python 3.6+.

.. note::

    Because `Python 2.7 supports ended January 1, 2020 <https://pythonclock.org/>`_, new projects
    should consider supporting Python 3 only, which is simpler than trying to support both.
    As a result, support for Python 2.7 in this example project has been dropped.

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

Dependencies
------------

Dependencies are defined in:

- ``requirements.in``

- ``requirements.txt``

- ``dev-requirements.in``

- ``dev-requirements.txt``

Virtual Environments
^^^^^^^^^^^^^^^^^^^^

It is best practice during development to create an isolated
`Python virtual environment <https://docs.python.org/3/library/venv.html>`_ using the
``venv`` standard library module. This will keep dependant Python packages from interfering
with other Python projects on your system.

On \*Nix:

.. code-block:: bash

    $ python3 -m venv venv
    $ source venv/bin/activate

On Windows ``cmd``:

.. code-block:: bash

    > py -m venv venv
    > venv\Scripts\activate.bat

Once activated, it is good practice to update core packaging tools (``pip``, ``setuptools``, and
``wheel``) to the latest versions.

.. code-block:: bash

    (venv) $ python -m pip install --upgrade pip setuptools wheel

(Applications Only) Locking Dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This project uses `pip-tools <https://github.com/jazzband/pip-tools>`_ to lock project
dependencies and create reproducible virtual environments.

**Note:** *Library* projects should not lock their ``requirements.txt``. Since ``python-blueprint``
also has a CLI application, this end-user application example is used to demonstrate how to
lock application dependencies.

To update dependencies:

.. code-block:: bash

    (venv) $ python -m pip install pip-tools
    (venv) $ pip-compile --upgrade
    (venv) $ pip-compile --upgrade dev-requirements.in

After upgrading dependencies, run the unit tests as described in the `Unit Testing`_ section
to ensure that none of the updated packages caused incompatibilities in the current project.

Syncing Virtual Environments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To cleanly install your dependencies into your virtual environment:

.. code-block:: bash

    (venv) $ pip-sync requirements.txt dev-requirements.txt

Packaging
---------

This project is designed as a Python package, meaning that it can be bundled up and redistributed
as a single compressed file.

Packaging is configured by:

- ``pyproject.toml``

- ``setup.py``

- ``MANIFEST.in``

To package the project as both a
`source distribution <https://docs.python.org/3/distutils/sourcedist.html>`_ and a
`wheel <https://wheel.readthedocs.io/en/stable/>`_:

.. code-block:: bash

    (venv) $ python setup.py sdist bdist_wheel

This will generate ``dist/fact-1.0.0.tar.gz`` and ``dist/fact-1.0.0-py3-none-any.whl``.

Read more about the `advantages of wheels <https://pythonwheels.com/>`_ to understand why
generating wheel distributions are important.

Upload Distributions to PyPI
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Source and wheel redistributable packages can be
`uploaded to PyPI <https://packaging.python.org/tutorials/packaging-projects/>`_ or installed
directly from the filesystem using ``pip``.

To upload to PyPI:

.. code-block:: bash

    (venv) $ python -m pip install twine
    (venv) $ twine upload dist/*

Testing
-------

Automated testing is performed using `tox <https://tox.readthedocs.io/en/latest/index.html>`_.
tox will automatically create virtual environments based on ``tox.ini`` for unit testing,
PEP8 style guide checking, and documentation generation.

.. code-block:: bash

    # Run all environments.
    #   To only run a single environment, specify it like: -e lint
    # Note: tox is installed into the virtual environment automatically by pip-sync command above.
    (venv) $ tox

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
unit testing. Open ``htmlcov/index.html`` in a web browser to view the report. Code coverage 
reports help identify areas of the project that are currently not tested.

Code coverage is configured in ``pyproject.toml``.

To pass arguments to ``pytest`` through ``tox``:

.. code-block:: bash

    (venv) $ tox -e py37 -- -k invalid_factorial

Code Style Checking
^^^^^^^^^^^^^^^^^^^

`PEP8 <https://www.python.org/dev/peps/pep-0008/>`_ is the universally accepted style
guide for Python code. PEP8 code compliance is verified using `flake8 <http://flake8.pycqa.org/>`_.
flake8 is configured in the ``[flake8]`` section of ``tox.ini``. Extra flake8 plugins
are also included:

- ``pep8-naming``: Ensure functions, classes, and variables are named with correct casing.

Automated Code Formatting
^^^^^^^^^^^^^^^^^^^^^^^^^

Code is automatically formatted using `black <https://github.com/psf/black>`_. Imports are
automatically sorted and grouped using `isort <https://github.com/PyCQA/isort/>`_.

These tools are configured by:

- ``pyproject.toml``

To automatically format code, run:

.. code-block:: bash

    (venv) $ tox -e fmt

To verify code has been formatted, such as in a CI job:

.. code-block:: bash

    (venv) $ tox -e fmt-check

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

    # Note: Sphinx is installed into the virtual environment automatically by pip-sync command
    # above.
    (venv) $ mkdir docs
    (venv) $ cd docs
    (venv) $ sphinx-quickstart --no-makefile --no-batchfile --extensions sphinx.ext.napoleon
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

    .. include:: ../README.rst

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

This project uses the type hinting syntax introduced in Python 3:

.. code-block:: python

    def factorial(n: int) -> int:

Type checking is performed by mypy via ``tox -e type-check``. mypy is configured in ``setup.cfg``.

Distributing Type Hints
^^^^^^^^^^^^^^^^^^^^^^^

`PEP 561 <https://www.python.org/dev/peps/pep-0561/>`_ defines how a Python package should
communicate the presence of inline type hints to static type checkers.
`mypy's documentation <https://mypy.readthedocs.io/en/stable/installed_packages.html>`_ provides
further examples on how to do this as well.

``mypy`` looks for the existence of a file named ``py.typed`` in the root of the installed
package to indicate that inline type hints should be checked.

Licensing
---------

Licensing for the project is defined in:

- ``LICENSE.txt``

- ``setup.py``

This project uses a common permissive license, the MIT license.

You may also want to list the licenses of all of the packages that your Python project depends on.
To automatically list the licenses for all dependencies in ``requirements.txt`` (and their
transitive dependencies) using
`pip-licenses <https://github.com/raimon49/pip-licenses>`_:

.. code-block:: bash

    (venv) $ tox -e licenses
    ...
     Name        Version  License
     colorama    0.4.3    BSD License
     exitstatus  1.3.0    MIT License

PyCharm Configuration
---------------------

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

- Settings | Search "Force parentheses"

    - Editor | Code Style | Python | Wrapping and Braces | "From" Import Statements

        - ☑ Force parentheses if multiline

Integrate Code Formatters
^^^^^^^^^^^^^^^^^^^^^^^^^

To integrate automatic code formatters into PyCharm, reference the following instructions:

- `black integration <https://black.readthedocs.io/en/stable/editor_integration.html#pycharm-intellij-idea>`_

    - The File Watchers method (step 3) is recommended. This will run ``black`` on every save.

- `isort integration <https://github.com/timothycrosley/isort/wiki/isort-Plugins>`_

    - The File Watchers method (option 1) is recommended. This will run ``isort`` on every save.

.. tip::

    These tools work best if you properly mark directories as excluded from the project that should
    be, such as ``.tox``. See
    https://www.jetbrains.com/help/pycharm/project-tool-window.html#content_pane_context_menu
    on how to Right Click | Mark Directory as | Excluded.
