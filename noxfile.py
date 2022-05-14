import nox
from nox_poetry import Session, session

nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["fmt_check", "type_check", "lint", "test", "docs"]

test_dependencies = ["pytest", "pytest-cov"]


@session(python=["3.8", "3.9", "3.10"])
def test(s: Session) -> None:
    s.install(".", *test_dependencies)
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


@session
def type_check(s: Session) -> None:
    # It is important to install the main project and test dependencies, as some packages contain
    # inline type hints (PEP 561) that mypy will use.
    s.install(".", *test_dependencies, "mypy")
    s.run("mypy", "src", "tests")


@session
def lint(s: Session) -> None:
    s.install(
        "flake8",
        "flake8-bugbear",
        "flake8-broken-line",
        "flake8-comprehensions",
        "pep8-naming",
        "pyproject-flake8",
    )
    s.run("pflake8")


fmt_dependencies = ["isort", "black"]


@session
def fmt(s: Session) -> None:
    s.install(*fmt_dependencies)
    s.run("isort", ".")
    s.run("black", ".")


@session
def fmt_check(s: Session) -> None:
    s.install(*fmt_dependencies)
    s.run("isort", "--check", ".")
    s.run("black", "--check", ".")


doc_dependencies = [
    "mkdocs-material",
    "mkdocs-htmlproofer-plugin",
    "mkdocstrings[python]",
    "mkdocs-gen-files",
    "mkdocs-literate-nav",
]

# Environment variable needed for mkdocstrings-python to locate source files.
doc_env = {"PYTHONPATH": "src"}


@session
def docs(s: Session) -> None:
    s.install(*doc_dependencies)
    s.run("mkdocs", "build", env=doc_env)


@session
def docs_serve(s: Session) -> None:
    s.install(*doc_dependencies)
    s.run("mkdocs", "serve", env=doc_env)


@session
def docs_github_pages(s: Session) -> None:
    s.install(*doc_dependencies)
    s.run("mkdocs", "gh-deploy", "--force", env=doc_env)


@session(reuse_venv=False)
def licenses(s: Session) -> None:
    s.install(".", "pip-licenses")
    s.run("pip-licenses", *s.posargs)
