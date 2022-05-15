import nox
from nox_poetry import Session, session

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["fmt_check", "type_check", "lint", "test", "docs"]


@session(python=["3.8", "3.9", "3.10"])
def test(s: Session) -> None:
    s.install(".", "pytest", "pytest-cov")
    s.run(
        "python",
        "-m",
        "pytest",
        "--cov",
        "fact",
        "--cov-report",
        "html",
        "--cov-report",
        "term",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts using the existing
# environment. This requires that nox is installed and run using `poetry install`.
@session(venv_backend="none")
def type_check(s: Session) -> None:
    # It is important to install the main project and test dependencies, as some packages contain
    # inline type hints (PEP 561) that mypy will use.
    s.run("mypy", "src", "tests")


@session(venv_backend="none")
def lint(s: Session) -> None:
    # Run pyproject-flake8 entrypoint to support reading configuration from pyproject.toml.
    s.run("pflake8")


@session(venv_backend="none")
def fmt(s: Session) -> None:
    s.run("isort", ".")
    s.run("black", ".")


@session(venv_backend="none")
def fmt_check(s: Session) -> None:
    s.run("isort", "--check", ".")
    s.run("black", "--check", ".")


# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "src"}


@session(venv_backend="none")
def docs(s: Session) -> None:
    s.run("mkdocs", "build", env=doc_env)


@session(venv_backend="none")
def docs_serve(s: Session) -> None:
    s.run("mkdocs", "serve", env=doc_env)


@session(venv_backend="none")
def docs_github_pages(s: Session) -> None:
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


@session(reuse_venv=False)
def licenses(s: Session) -> None:
    s.install(".", "pip-licenses")
    s.run("pip-licenses", *s.posargs)
