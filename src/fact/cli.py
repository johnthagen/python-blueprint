#!/usr/bin/env python3

import argparse
import sys

import colorama
from exitstatus import ExitStatus

from fact.lib import factorial


def parse_args() -> argparse.Namespace:
    """Parse user command line arguments."""
    parser = argparse.ArgumentParser(description='Compute factorial of a given input.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-n',
                        type=int,
                        required=True,
                        help='The input n of fact(n).')
    return parser.parse_args()


def main() -> ExitStatus:
    """Accept arguments from the user, compute the factorial, and display the results."""
    colorama.init(autoreset=True, strip=False)
    args = parse_args()

    print(f'fact({colorama.Fore.CYAN}{args.n}{colorama.Fore.RESET}) = '
          f'{colorama.Fore.GREEN}{factorial(args.n)}{colorama.Fore.RESET}')
    return ExitStatus.success


# Allow the script to be run standalone (useful during development in PyCharm).
if __name__ == '__main__':
    sys.exit(main())
