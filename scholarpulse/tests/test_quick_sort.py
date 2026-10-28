import pytest

from scholarpulse.core.c import quick_sort


@pytest.mark.parametrize(
    "numbers, expected",
    [
        ([3, 1, 2], [1, 2, 3]),
        ([5, 4, 3, 2, 1], [1, 2, 3, 4, 5]),
        ([10, 10, 10, 10], [10, 10, 10, 10]),
        ([], []),
        ([42], [42]),
        ([9, -3, 0, 5, -1], [-3, -1, 0, 5, 9]),
        ([8, 1, 7, 2, 6, 3, 5, 4], [1, 2, 3, 4, 5, 6, 7, 8]),
        ([2, 1, 2, 1, 2], [1, 1, 2, 2, 2]),
        ([3.5, 1.2, 2.7, 1.2, 9.9], [1.2, 1.2, 2.7, 3.5, 9.9]),
        ([7, 0, -5, 2, 9, 1, -2], [-5, -2, 0, 1, 2, 7, 9]),
    ],
)
def test_quick_sort_cases(numbers, expected):
    assert quick_sort(numbers) == expected
