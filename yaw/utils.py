from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from timeit import default_timer
from typing import TYPE_CHECKING, Any, Callable, NamedTuple

import h5py
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import IntervalIndex


TypePathStr = Path | str


def outer_triu_sum(a, b , *, k: int = 0, axis: int | None = None) -> NDArray:
    """
    Equivalent to
        np.triu(np.outer(a, b), k).sum(axis)
    but supports extra dimensions in a and b and does not construct the full
    outer product matrix.
    """
    a = np.atleast_1d(a)
    b = np.atleast_1d(b)
    if a.shape != b.shape:
        raise IndexError("shape of 'a' and 'b' does not match")
    # allocate output array
    dtype = (a[0] * b[0]).dtype  # correct dtype for product
    N = len(a)
    # sum all elements
    if axis is None:
        result = np.zeros_like(a[0], dtype=dtype)
        for i in range(min(N, N-k)):
            result += (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum row-wise
    elif axis == 1:
        result = np.zeros_like(b, dtype=dtype)
        for i in range(min(N, N-k)):
            result[i] = (a[i] * b[max(0, i+k):]).sum(axis=0)
    # sum column-wise
    elif axis == 0:
        result = np.zeros_like(a, dtype=dtype)
        for i in range(max(0, k), N):
            result[i] = (a[:min(N, max(0, i-k+1))] * b[i]).sum(axis=0)
    return result[()]


class LimitTracker:

    def __init__(self):
        self.min = +np.inf
        self.max = -np.inf

    def update(self, data: NDArray | None):
        if data is not None:
            self.min = np.minimum(self.min, np.min(data))
            self.max = np.maximum(self.max, np.max(data))

    def get(self):
        vmin = None if np.isinf(self.min) else self.min
        vmax = None if np.isinf(self.max) else self.max
        return vmin, vmax


def scales_to_keys(scales: NDArray[np.float_]) -> list[str]:
    return [f"kpc{scale[0]:.0f}t{scale[1]:.0f}" for scale in scales]


def long_num_format(x: float) -> str:
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1000:
        exp += 1
        x /= 1000.0
    prefix = str(x).rstrip("0").rstrip(".")
    suffix = ["", "K", "M", "B", "T"][exp]
    return prefix + suffix


def bytes_format(x: float) -> str:
    x = float(f"{x:.3g}")
    exp = 0
    while abs(x) >= 1024:
        exp += 1
        x /= 1024.0
    prefix = f"{x:.3f}"[:4].rstrip(".")
    suffix = ["B ", "KB", "MB", "GB", "TB"][exp]
    return prefix + suffix


def format_float_fixed_width(value, width):
    string = f"{value: .{width}f}"[:width]
    if "nan" in string or "inf" in string:
        string = f"{string.strip():>{width}s}"
    return string


class PatchIDs(NamedTuple):
    id1: int
    id2: int

    
class PatchedQuantity(ABC):

    @abstractproperty
    def n_patches(self) -> int: pass


class BinnedQuantity(ABC):

    def get_binning(self) -> IntervalIndex: raise NotImplementedError

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = self.n_bins
        binning = self.get_binning()
        z = f"{binning[0].left:.3f}...{binning[-1].right:.3f}"
        return f"{name}({n_bins=}, {z=})"

    @property
    def n_bins(self) -> int:
        return len(self.get_binning())

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.get_binning()])

    @property
    def edges(self) -> NDArray[np.float_]:
        binning = self.get_binning()
        return np.append(binning.left, binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.diff(self.edges)

    def is_compatible(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if np.any(self.get_binning() != other.get_binning()):
            return False
        return True


class HDFSerializable(ABC):

    @abstractclassmethod
    def from_hdf(
        cls,
        source: h5py.Group
    ) -> HDFSerializable: raise NotImplementedError

    @abstractmethod
    def to_hdf(self, dest: h5py.Group) -> None: raise NotImplementedError

    @classmethod
    def from_file(cls, path: TypePathStr) -> HDFSerializable:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


class DictRepresentation(ABC):

    @abstractclassmethod
    def from_dict(
        cls,
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any]  # passing additinal constructor data
    ) -> DictRepresentation:
        return cls(**the_dict)

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LogCustomWarning:

    def __init__(
        self,
        logger: logging.Logger,
        alt_message: str | None = None,
        ignore: bool = True
    ):
        self._logger = logger
        self._message = alt_message
        self._ignore = ignore

    def _process_warning(self, message, category, filename, lineno, *args):
        if not self._ignore:
            self._old_showwarning(message, category, filename, lineno, *args)
        if self._message is not None:
            message = self._message
        else:
            message = f"{category.__name__}: {message}"
        self._logger.warn(message)

    def __enter__(self) -> TimedLog:
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._process_warning
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        warnings.showwarning = self._old_showwarning


class TimedLog:

    def __init__(
        self,
        logging_callback: Callable,
        msg: str | None = None
    ) -> None:
        self.callback = logging_callback
        self.msg = msg

    def __enter__(self) -> TimedLog:
        self.t = default_timer()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        delta = default_timer() - self.t
        time = str(timedelta(seconds=round(delta)))
        self.callback(f"{self.msg} - done {time}")
