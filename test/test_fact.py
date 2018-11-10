import pytest

from fact.lib import factorial


@pytest.mark.parametrize('n,expected', [
    (1, 1),
    (2, 2),
    (3, 6),
    (10, 3628800)
])
def test_eval(n: int, expected: int) -> None:
    assert factorial(n) == expected
