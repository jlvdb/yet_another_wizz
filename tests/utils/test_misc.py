from math import inf, nan

import numpy as np
from numpy.testing import assert_array_equal
from pytest import mark

from yaw.utils import misc


def test_groupby():
    n_items = 4
    items = np.arange(n_items)
    n_tile = 5
    array = np.tile(items, n_tile)

    for item, (key, data) in zip(items, misc.groupby(array, array)):
        assert item == key
        assert_array_equal(data, np.full(n_tile, item))


@mark.parametrize(
    "value,expect",
    [
        (-123.0, "-123"),
        (1234.0, "1.23K"),
        (1235.0, "1.24K"),
        (12345, "12.3K"),
        (123456, "123K"),
        (1234567, "1.23M"),
    ],
)
def test_format_long_num(value, expect):
    assert misc.format_long_num(value) == expect


@mark.parametrize(
    "value,expect",
    [
        (1.9, "0m01.90s"),
        (59.99, "0m59.99s"),
        (61.9, "1m01.90s"),
        (3600, "60m00.00s"),
    ],
)
def test_format_time(value, expect):
    assert misc.format_time(value) == expect


@mark.parametrize(
    "value,width,expect",
    [
        (0.1, 3, " 0."),
        (-12.3, 5, "-12.3"),
        (0.001, 9, " 0.001000"),
        (123, 9, " 123.0000"),
        (123, 3, " 123"),
        (inf, 2, " inf"),
        (nan, 6, "   nan"),
    ],
)
def test_format_float_fixed_width(value, width, expect):
    assert misc.format_float_fixed_width(value, width) == expect
