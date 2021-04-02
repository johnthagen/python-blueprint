#!/usr/bin/env python3

import colorama
import typer
from typer import Argument

from fact.lib import factorial


def main(n: int = Argument(..., min=0, help="The input n of fact(n)")) -> None:
    """Compute factorial of a given input."""
    colorama.init(autoreset=True, strip=False)

    print(
        f"fact({colorama.Fore.CYAN}{n}{colorama.Fore.RESET}) = "
        f"{colorama.Fore.GREEN}{factorial(n)}{colorama.Fore.RESET}"
    )


# Allow the script to be run standalone (useful during development).
if __name__ == "__main__":
    typer.run(main)
