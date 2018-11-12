python-blueprint
================

.. image:: https://travis-ci.org/johnthagen/python-blueprint.svg?branch=master
    :target: https://travis-ci.org/johnthagen/python-blueprint

Example Python project that demonstrates how to create a Python project using the latest
Python testing and linting tooling.

Requirements
------------

Python 2.7 or 3.4+.

.. note::

    Because `Python 2.7 supports ends January 1, 2020 <https://pythonclock.org/>`_, new projects
    may want to consider supporting Python 3 only, which is simpler than trying to support both.
    Support for Python 2.7 in this example project is provided only for completeness.

Run CLI with ``pipenv``
-----------------------

Install `pipenv <https://pipenv.readthedocs.io/en/latest/>`_.

.. code-block:: bash

    $ pipenv install --dev
    $ pipenv shell
    (python-blueprint) $ fact -n 10

Regenerate Pipfile from requirements.txt
----------------------------------------

.. code-block:: bash

    $ pipenv install -e .
    $ pipenv install -r requirements.txt
    $ pipenv install -r dev-requirements.txt --dev
