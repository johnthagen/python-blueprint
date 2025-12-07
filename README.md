# PINN Library

[![GitHub Actions][github-actions-badge]](https://github.com/johnthagen/python-blueprint/actions)
[![uv][uv-badge]](https://github.com/astral-sh/uv)
[![Nox][nox-badge]](https://github.com/wntrblm/nox)
[![Ruff][ruff-badge]](https://github.com/astral-sh/ruff)
[![Type checked with mypy][mypy-badge]](https://mypy-lang.org/)

[github-actions-badge]: https://github.com/johnthagen/python-blueprint/actions/workflows/ci.yml/badge.svg
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[nox-badge]: https://img.shields.io/badge/%F0%9F%A6%8A-Nox-D85E00.svg
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg

A modular and flexible solution to build Physics-Informed Neural Networks (PINNs) for any mathematical problem.

## Overview

The PINN library is designed to provide a robust framework for solving differential equations using neural networks. It separates the problem definition from the solution architecture, allowing users to easily implement custom problems or leverage existing ones.

### Key Concepts

- **Modular**: Components like datasets, models, and training loops are decoupled.
- **Flexible**: Supports various types of problems. Currently, it includes an integration layer for Ordinary Differential Equations (ODEs), but it is easily expandable to Partial Differential Equations (PDEs) and other mathematical problems.
- **Expandable**: New problems can be implemented by defining the constraints and fields. A future goal is to provide a bootstrap script to help users set up new projects quickly.

### Examples

The library currently includes an implementation for the **SIR Inverse** problem as an example of how to use the ODE integration layer. We aim to build a catalog of problems that can be easily expanded by the community.

## Documentation

Comprehensive documentation for all components is available. You can generate the user guide and API documentation locally.

To build the documentation:

```shell
uv run nox -s docs
```

Open `docs/user_guide/site/index.html` to view it.

## Development

This project uses modern Python tooling:

- **uv** for dependency management.
- **Nox** for automation.
- **Ruff** for linting and formatting.
- **pytest** for testing.

### Installation

Install dependencies:

```shell
uv sync
```

### Running Tests

```shell
uv run nox -s test
```
