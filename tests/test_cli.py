from typer.testing import CliRunner

from fact.cli import app


def test_main() -> None:
    """Test the main function of the CLI."""

    runner = CliRunner()
    result = runner.invoke(app, ["5"])
    assert result.exit_code == 0
    assert "fact(5) = 120" in result.output
