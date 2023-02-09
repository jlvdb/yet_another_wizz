from __future__ import annotations

from abc import ABC, abstractproperty
from collections.abc import Sized
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import IntervalIndex


TypePatchKey = tuple[int, int]
TypeScaleKey = str


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


class BinnedQuantity(ABC, Sized):

    def __len__(self) -> int:
        return len(self.binning)

    @abstractproperty
    def binning(self) -> IntervalIndex:
        raise NotImplementedError

    @property
    def mids(self) -> NDArray[np.float_]:
        return np.array([z.mid for z in self.binning])

    @property
    def dz(self) -> NDArray[np.float_]:
        return np.array([z.right - z.left for z in self.binning])

    def is_compatible(self, other: BinnedQuantity) -> bool:
        if not isinstance(other, BinnedQuantity):
            raise TypeError(
                f"object of type {type(other)} is not compatible with "
                f"{self.__class__}")
        if np.any(self.binning != other.binning):
            return False
        return True
