# Cursor Rules: Python (project-specific)

These rules guide Python contributions in this repository.

## Environment and tooling
- Use `uv` for dependency management and virtualenvs. Supported Python versions are 3.9–3.13.
- Use Nox sessions defined in `noxfile.py` to run tasks:
  - `uv run nox -s fmt` to sort imports and format code (ruff is the single formatter)
  - `uv run nox -s lint` to lint and check formatting (no autofix)
  - `uv run nox -s type_check` to run mypy (strict)
  - `uv run nox -s test` to run tests with coverage (100% threshold enforced)
  - `uv run nox -s docs` to build docs
  - `uv run nox -s licenses -- --format=json` for license reports when needed
- Pre-commit hooks must pass before pushing (`ruff`, `ruff-format`, `mypy`, and repo checks).
- Prefer running tools via `uv run` so executables resolve from the project environment.

## Project layout and imports
- Place source under `src/` with package root `src/fact/`.
- Place tests under `tests/` with pytest.
- Do not introduce relative imports; always use absolute imports.
- Follow import grouping and order enforced by ruff-isort:
  - standard library, third-party, first-party; type-only imports last when possible.
- Avoid wildcard imports; export public APIs explicitly via `__all__` when applicable.

## Code style
- Line length is 99 characters.
- Ruff is the single source of truth for formatting and linting (see `pyproject.toml`).
- Run `uv run nox -s fmt` before committing. Do not use additional formatters.
- Prefer `pathlib.Path` over `os.path`, and f-strings over string formatting.
- Keep functions small and focused; avoid deep nesting; favor early returns.
- Eliminate trailing whitespace and ensure a newline at end of file.

## Typing and APIs
- Add precise type hints for all new functions, methods, and public APIs.
- Follow strict typing; mypy is configured with `strict = true`.
- Prefer clear names over abbreviations; avoid 1–2 character identifiers.
- Use early-return patterns and handle error cases first.
- Prefer PEP 604 unions (`str | None`), `typing.Literal`, `Final`, `TypedDict`, `Protocol`, `NewType`.
- Use `dataclasses.dataclass` for simple data containers (`frozen=True, slots=True` when sensible).
- Avoid `Any`. If unavoidable, minimize scope and justify with a focused `# type: ignore[code]`.
- Public APIs should have stable, well-typed signatures and clear error semantics.

## Tests
- Add or update tests for new behavior in `tests/`.
- Keep coverage at or above the configured threshold (100%).
- Use pytest naming: files `test_*.py`, tests `test_*`.
- Treat warnings as errors (configured); fix root causes or mark with precise filters.
- Use `pytest.mark.parametrize` for input matrices; prefer fixtures over ad-hoc setup.
- Avoid broad exception catching in tests; assert specific exceptions and messages.
- Use `tmp_path`/`monkeypatch` judiciously; avoid network or external side effects.

## Docs and docstrings
- Use Google-style docstrings for public functions/classes.
- Keep docs up to date when modifying public behavior; update MkDocs content if needed.
- Type hints are the source of truth for types; docstrings should describe behavior and errors.
- Build docs with `uv run nox -s docs`. Validate external links via `docs_check_urls` when needed.

## CLI specifics
- The CLI is implemented with Typer in `src/fact/cli.py`. Preserve the current CLI UX unless explicitly changing it.

- Do not add TODO comments; implement the behavior or open an issue.

## Error handling and security
- Raise specific exceptions (`ValueError`, `TypeError`, etc.); avoid bare `except`.
- Prefer `subprocess.run([...], check=True)` without `shell=True`. If shell is required, document why.
- Never use `eval`/`exec` on untrusted input. Keep secrets in environment variables, not in code.

## Commit messages
- Follow Conventional Commits as documented in `.cursor/rules/commit-messages.md`.

## Useful commands
- `uv run nox -s fmt` — sort imports and format
- `uv run nox -s lint` — lint and format check
- `uv run nox -s type_check` — mypy strict type checking
- `uv run nox -s test` — run tests with coverage enforcement
- `uv run nox -s docs` — build documentation

References: `pyproject.toml`, `noxfile.py`,
`https://github.com/PatrickJS/awesome-cursorrules/blob/main/rules-new/python.mdc`.
