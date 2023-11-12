"""This module implements various small utility functions, mostly for string
formatting numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, Union

import tqdm

__all__ = [
    "job_progress_bar",
    "long_num_format",
    "bytes_format",
    "format_float_fixed_width",
]


TypePathStr = Union[Path, str]
Tjob = TypeVar("Tjob")


def job_progress_bar(
    iterable: Iterable[Tjob], total: int | None = None
) -> Iterable[Tjob]:
    """Configure and return a tqdm progress bar with custom format."""
    config = dict(delay=0.5, leave=False, smoothing=0.1, unit="jobs")
    return tqdm.tqdm(iterable, total=total, **config)


def long_num_format(x: float) -> str:
    """Format a floating point number as string with a numerical suffix.

    E.g.: 1234.0 is converted to ``1.24K``.
    """
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1000:
        exp += 1
        x /= 1000.0
    prefix = str(x).rstrip("0").rstrip(".")
    suffix = ["", "K", "M", "B", "T"][exp]
    return prefix + suffix


def bytes_format(x: float) -> str:
    """Format a floating point number as string in bytes notation.

    E.g.: 123.0 is converted to ``123B``, 1234.0 is converted to ``1.23KB``.
    """
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1024:
        exp += 1
        x /= 1024.0
    prefix = f"{x:.3f}"[:4].rstrip(".")
    suffix = ["B ", "KB", "MB", "GB", "TB"][exp]
    return prefix + suffix


def format_float_fixed_width(value: float, width: int) -> str:
    """Format a floating point number as string with fixed width."""
    string = f"{value: .{width}f}"[:width]
    if "nan" in string or "inf" in string:
        string = f"{string.strip():>{width}s}"
    return string
