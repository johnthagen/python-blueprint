import pytest

from fact.lib import factorial, InvalidFactorialError


# fmt: off
@pytest.mark.parametrize('n,expected', [
    (1, 1),
    (2, 2),
    (3, 6),
    (10, 3628800),
])
# fmt: on
def test_factorial(n: int, expected: int) -> None:
    assert factorial(n) == expected


# fmt: off
@pytest.mark.parametrize('n', [
    (-1),
    (-100),
])
# fmt: on
def test_invalid_factorial(n: int) -> None:
    with pytest.raises(InvalidFactorialError):
        factorial(n)
