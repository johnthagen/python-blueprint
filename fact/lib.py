def factorial(n):
    # type: (int) -> int
    """Computes the factorial through a recursive algorithm.

    Args:
        n: Input value.

    Returns:
        Computed factorial.
    """
    if n == 0:
        return 1

    return n * factorial(n - 1)
