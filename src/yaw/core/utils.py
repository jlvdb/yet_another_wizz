"""This module implements various small utility functions, mostly for string
formatting numbers.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TypeVar, Union

import numpy as np
import tqdm
from numpy.typing import NDArray

__all__ = [
    "job_progress_bar",
    "LimitTracker",
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


class LimitTracker:
    """Tracks the global minimum and maximum of batches of arrays."""

    def __init__(self):
        self.min = +np.inf
        self.max = -np.inf

    def update(self, data: NDArray | None):
        """Process a new array with data and update the limits if necessary."""
        if data is not None:
            self.min = np.minimum(self.min, np.min(data))
            self.max = np.maximum(self.max, np.max(data))

    def get(self):
        """Return the minimum and maximum values processed so far."""
        vmin = None if np.isinf(self.min) else self.min
        vmax = None if np.isinf(self.max) else self.max
        return vmin, vmax


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
