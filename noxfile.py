from nox import Session, options, param, parametrize
from nox_uv import session

options.error_on_external_run = True
options.default_venv_backend = "uv"
options.sessions = ["lint", "type_check", "test", "docs"]


@session(
    python=["3.10", "3.11", "3.12", "3.13", "3.14", "3.14t"],
    uv_groups=["test"],
)
def test(s: Session) -> None:
    s.run(
        "pytest",
        "--cov=fact",
        "--cov-report=html",
        "--cov-report=term",
        "--cov-fail-under=100",
        "tests",
        *s.posargs,
    )


# For some sessions, set venv_backend="none" to simply execute scripts within the existing outer
# uv-generated virtual environment, rather than have nox create a new one for each session. This
# makes commonly repeated sessions execute faster.
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


# Install only main dependencies for the license report.
@session(uv_groups=["licenses"], uv_no_install_project=True)
def licenses(s: Session) -> None:
    s.run(
        "pip-licenses",
        "--format=markdown",
        "--output-file=./docs/licenses/summary.txt",
        *s.posargs,
    )
    s.run(
        "pip-licenses",
        "--format=plain-vertical",
        "--with-license-files",
        "--no-license-path",
        "--output-file=./docs/licenses/license_files.txt",
    )
    s.run("pip-licenses", *s.posargs)
    s.run("pip-licenses", "--summary")
