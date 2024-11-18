"""
Implements generic utility functions, e.g. for formatting data as string.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Sized
    from typing import Any, Generator

    from h5py import Group
    from numpy.typing import NDArray

__all__ = [
    "HDF_COMPRESSION",
    "groupby",
    "common_len_assert",
    "write_version_tag",
    "load_version_tag",
    "is_legacy_dataset",
    "format_float_fixed_width",
    "format_long_num",
    "format_time",
]


HDF_COMPRESSION = dict(fletcher32=True, compression="gzip", shuffle=True)
"""Default HDF5 compression options."""


def groupby(key_array: NDArray, value_array: NDArray) -> Generator[tuple[Any, NDArray]]:
    """
    Implements a groupby-operation on numpy array.

    Args:
        key_array:
            Array with keys from which unique groups are formed.
        value_array:
            Array with arbitrary type that will split into groups along its
            first dimension.

    Yields:
        Tuples of key and array of values corresponding to this key.
    """
    idx_sort = np.argsort(key_array)
    keys_sorted = key_array[idx_sort]
    values_sorted = value_array[idx_sort]

    uniques, idx_split = np.unique(keys_sorted, return_index=True)
    yield from zip(uniques, np.split(values_sorted, idx_split[1:]))


def common_len_assert(sized: Iterable[Sized]) -> int:
    """Verify that a set of containers has the same length and return this
    common length."""
    length = None
    for item in sized:
        if length is None:
            length = len(item)
        else:
            if len(item) != length:
                raise ValueError("length of inputs does not match")
    return length


def write_version_tag(dest: Group) -> None:
    """Write a ``version`` tag with the current code version to a HDF5 file
    group."""
    from yaw._version import __version__

    dest.create_dataset("version", data=__version__)


def load_version_tag(source: Group) -> str:
    """Load the code version that created a HDF5 file from a ``version`` tag in
    the current group."""
    try:
        tag = source["version"][()]
        return tag.decode("utf-8")

    except KeyError:
        return "2.x.x"


def is_legacy_dataset(source: Group) -> bool:
    """Determine, if the current file has been created by an old version of
    `yet_another_wizz` (version < 3.0)."""
    return "version" not in source


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"
    if "nan" in string or "inf" in string:
        string = f"{string.rstrip():>{width}s}"

    num_digits = len(string.split(".")[0])
    return string[: max(width, num_digits)]


def format_long_num(x: float | int) -> str:
    """
    Format a floating point number as string with a numerical suffix.

    E.g.: 1234.0 is converted to ``1.23K``.
    """
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1000:
        exp += 1
        x /= 1000.0
    prefix = str(x).rstrip("0").rstrip(".")
    suffix = ["", "K", "M", "B", "T"][exp]
    return prefix + suffix


def format_time(elapsed: float) -> str:
    """Format time in seconds as minutes and seconds: ``[MM]MmSS.SSs``"""
    minutes, seconds = divmod(elapsed, 60)
    return f"{minutes:.0f}m{seconds:05.2f}s"
