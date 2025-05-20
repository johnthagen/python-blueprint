import subprocess


def test_main() -> None:
    """Test the main function of the CLI."""

    # Run the CLI command
    result = subprocess.run(
        ["python", "-m", "fact.cli", "5"],
        capture_output=True,
        text=True,
    )

    # Check the output
    assert result.returncode == 0
    assert "fact(5) = 120" in result.stdout
