"""
Implements generic utility functions, e.g. for formatting data as string.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import yaml

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sized
    from io import TextIOBase
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
    "transform_matches",
    "write_yaml",
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


def transform_matches(string: str, regex: str, transform: Callable[[str], str]) -> str:
    """Transforms matches in a regex search to replaces the original."""
    for match_info in re.finditer(regex, string):
        offset = len(string) - len(match_info.string)
        start = match_info.start() + offset
        end = match_info.end() + offset

        matched = match_info[0]
        string = string[:start] + transform(matched) + string[end:]

    return string


def write_yaml(
    obj: Any,
    file: TextIOBase,
    *,
    header_lines: Iterable[str] | None = None,
    indent: int = 2,
    sort_keys: bool = False,
    section: bool = True,
    **kwargs,
) -> None:
    """
    Serialise an object to YAML in a custom format, compatible with PyYAML.

    In particular, indent all list items (- ...) once for better readability.

    Args:
        obj:
            Object to serialise, must contain only python native types supported
            by PyYAML.
        file:
            Writable file object.

    Keyword Args:
        header_lines:
            Iterable of header lines that will in inserted as comments at the
            top of the file.
        indent:
            Number of spaces used as indentation.
        sort_keys:
            Whether to sort keys alphabetically.
        section:
            Whether to insert a new line between top level (i.e. unindented)
            items.
    """
    if header_lines is not None:
        header_lines = ("# " + line.rstrip("\n") for line in header_lines)
        header = "\n".join(header_lines) + "\n"
    else:
        header = ""

    string = yaml.safe_dump_all([obj], indent=indent, sort_keys=sort_keys, **kwargs)
    string = header + string

    # replace items (- ...) with indented items (  - ...)
    indent_str = " " * indent
    string = transform_matches(string, r"[\t ]*- ", lambda match: indent_str + match)

    # insert empty line before a line without indentation
    if section:
        string = transform_matches(string, r"\n\w", lambda match: "\n" + match)

    file.write(string)
