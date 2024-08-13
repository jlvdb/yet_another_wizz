"""
################################################################################

CURRENTLY NOT USED

################################################################################
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from yaw.meta import BinwiseData, Serialisable

Tclosed = Literal["left", "right"]


def parse_binning(binning: NDArray | None) -> NDArray | None:
    if binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")


class Binning(Sized, Iterable, Serialisable, BinwiseData):
    __slots__ = ("edges", "closed")

    def __init__(self, edges: Iterable, *, closed: Tclosed = "right") -> None:
        self.edges = np.asarray(edges)
        self.closed = closed

    def __getstate__(self) -> dict:
        return self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return dict(edges=self.edges, closed=self.closed)

    def get_bin_index(self, values: NDArray) -> NDArray[np.int_]:
        return np.digitize(values, self.edges, right=(self.closed == "right")) - 1
