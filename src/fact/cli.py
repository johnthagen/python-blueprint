#!/usr/bin/env python3

from typing import Annotated

from rich.console import Console
from typer import Argument, Typer

from fact.lib import factorial

app = Typer(add_completion=False)


@app.command()
def main(n: Annotated[int, Argument(min=0, help="The input n of fact(n)")]) -> None:
    """Compute factorial of a given input."""

    Console().print(f"fact({n}) = {factorial(n)}")


# Allow the script to be run standalone (useful during development).
if __name__ == "__main__":
    app()
