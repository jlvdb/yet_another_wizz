from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, NamedTuple, Type, TypeVar, Protocol, runtime_checkable)

import h5py
import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import IntervalIndex


TypePathStr = Path | str


def outer_triu_sum(a, b , *, k: int = 0, axis: int | None = None) -> NDArray:
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

    
class PatchedQuantity(Protocol):

    n_patches: int


@runtime_checkable
class BinnedQuantity(Protocol):

    binning: IntervalIndex

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = self.n_bins
        z = f"{self.binning[0].left:.3f}...{self.binning[-1].right:.3f}"
        return f"{name}({n_bins=}, {z=})"

    @property
    def n_bins(self) -> int:
        return len(self.binning)

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.binning])

    @property
    def edges(self) -> NDArray[np.float_]:
        return np.append(self.binning.left, self.binning.right[-1])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.diff(self.edges)

    def is_compatible(self, other) -> bool:
        if not isinstance(other, self.__class__):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if np.any(self.binning != other.binning):
            return False
        return True


THF = TypeVar("THF", bound="HDFSerializable")


class HDFSerializable(Protocol):

    @classmethod
    def from_hdf(cls: Type[THF], source: h5py.Group) -> THF:
        raise NotImplementedError

    def to_hdf(self, dest: h5py.Group) -> None:
        raise NotImplementedError

    @classmethod
    def from_file(cls: Type[THF], path: TypePathStr) -> THF:
        with h5py.File(str(path)) as f:
            return cls.from_hdf(f)

    def to_file(self, path: TypePathStr) -> None:
        with h5py.File(str(path), mode="w") as f:
            self.to_hdf(f)


TDR = TypeVar("TDR", bound="DictRepresentation")


class DictRepresentation(Protocol):

    @classmethod
    def from_dict(
        cls: Type[TDR],
        the_dict: dict[str, Any],
        **kwargs: dict[str, Any]  # passing additinal constructor data
    ) -> TDR:
        return cls(**the_dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
