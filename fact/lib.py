def factorial(n: int) -> int:
    """Computes the factorial of `n` through a recursive algorithm."""
    if n == 0:
        return 1

    return n * factorial(n - 1)
