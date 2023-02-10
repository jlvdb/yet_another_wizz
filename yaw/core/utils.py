from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Type, TypeVar, Protocol, runtime_checkable)

import h5py
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import IntervalIndex


TypePatchKey = tuple[int, int]
TypeScaleKey = str

TypePathStr = Path | str


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


def scales_to_keys(scales: NDArray[np.float_]) -> list[TypeScaleKey]:
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


class PatchedQuantity(Protocol):

    n_patches: int


@runtime_checkable
class BinnedQuantity(Protocol):

    binning: IntervalIndex

    def __len__(self) -> int:
        return len(self.binning)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        n_bins = len(self)
        z = f"{self.binning[0].left:.3f}...{self.binning[-1].right:.3f}"
        return f"{name}({n_bins=}, z={z})"

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.binning])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.array([z.right - z.left for z in self.binning])

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
    def from_hdf(cls: Type[THF], source: h5py.Group) -> THF: ...

    def to_hdf(self, dest: h5py.Group) -> None: ...

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
