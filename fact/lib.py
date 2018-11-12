class InvalidFactorialError(RuntimeError):
    """Error generated if an invalid factorial input is given."""


def factorial(n):
    # type: (int) -> int
    """Computes the factorial through a recursive algorithm.

    Args:
        n: Input value.

    Raises:
        InvalidFactorialError: If n is less than 0.

    Returns:
        Computed factorial.
    """
    if n < 0:
        raise InvalidFactorialError('n is less than zero: {}'.format(n))
    elif n == 0:
        return 1

    return n * factorial(n - 1)
