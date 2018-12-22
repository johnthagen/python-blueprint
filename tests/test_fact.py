import pytest

from fact.lib import factorial, InvalidFactorialError


@pytest.mark.parametrize('n,expected', [
    (1, 1),
    (2, 2),
    (3, 6),
    (10, 3628800)
])
def test_factorial(n, expected):
    # type: (int, int) -> None
    assert factorial(n) == expected


@pytest.mark.parametrize('n', [
    (-1),
    (-100),
])
def test_invalid_factorial(n):
    # type: (int) -> None
    with pytest.raises(InvalidFactorialError):
        factorial(n)
