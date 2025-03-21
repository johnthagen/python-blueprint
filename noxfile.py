from pathlib import Path
from tempfile import NamedTemporaryFile

import nox
from nox import param, parametrize
from nox_poetry import Session, session

nox.options.error_on_external_run = True
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "type_check", "test", "docs"]


@session(python=["3.9", "3.10", "3.11", "3.12", "3.13"])
def test(s: Session) -> None:
    s.install(".", "pytest", "pytest-cov", "pytest-randomly")
    s.run(
        "python",
        "-m",
        "pytest",
        "--cov=fact",
        "--cov-report=html",
        "--cov-report=term",
        "tests",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing Poetry
# environment. This requires that nox is run within the `poetry env activate` virtual environment
# or using `poetry run nox ...`.
@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(
            [
                "ruff",
                "check",
                ".",
                "--select",
                "I",
                # Also remove unused imports.
                "--select",
                "F401",
                "--extend-fixable",
                "F401",
                "--fix",
            ],
            id="sort_imports",
        ),
        param(["ruff", "format", "."], id="format"),
    ],
)
def fmt(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
@parametrize(
    "command",
    [
        param(["ruff", "check", "."], id="lint_check"),
        param(["ruff", "format", "--check", "."], id="format_check"),
    ],
)
def lint(s: Session, command: list[str]) -> None:
    s.run(*command)


@session(venv_backend="none")
def lint_fix(s: Session) -> None:
    s.run("ruff", "check", ".", "--extend-fixable", "F401", "--fix")


@session(venv_backend="none")
def type_check(s: Session) -> None:
    s.run("mypy", "src", "tests", "noxfile.py")


# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "src"}


@session(venv_backend="none")
def docs(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env)


@session(venv_backend="none")
def docs_check_urls(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env | {"HTMLPROOFER_VALIDATE_EXTERNAL_URLS": str(True)})


@session(venv_backend="none")
def docs_offline(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env | {"MKDOCS_MATERIAL_OFFLINE": str(True)})


@session(venv_backend="none")
def docs_serve(s: Session) -> None:
    s.run("mkdocs", "serve", env=doc_env)


@session(venv_backend="none")
def docs_github_pages(s: Session) -> None:
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


@session(reuse_venv=False)
def licenses(s: Session) -> None:
    # Generate a unique temporary file name. Poetry cannot write to the temp file directly on
    # Windows, so only use the name and allow Poetry to re-create it.
    with NamedTemporaryFile() as t:
        requirements_file = Path(t.name)

    # Install dependencies without installing the package itself:
    #   https://github.com/cjolowicz/nox-poetry/issues/680
    s.run_always(
        "poetry",
        "export",
        "--without-hashes",
        f"--output={requirements_file}",
        external=True,
    )
    s.install("pip-licenses", "-r", str(requirements_file))
    s.run("pip-licenses", *s.posargs)
    requirements_file.unlink()
