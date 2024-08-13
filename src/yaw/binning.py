from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from typing import Literal

import numpy as np
from numpy.typing import NDArray


Tclosed = Literal["left", "right"]


def parse_binning(binning: NDArray | None) -> NDArray | None:
    if binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")


class Binning(Sized, Iterable):
    __slots__ = ("edges", "closed")

    def __init__(self, edges: Iterable, *, closed: Tclosed = "right") -> None:
        self.edges = np.asarray(edges)
        self.closed = closed

    def __getstate__(self) -> dict:
        return dict(edges=self.edges, closed=self.closed)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __iter__(self) -> Iterator[Binning]:
        for i in range(len(self)):
            yield Binning(self.edges[i:i + 2], self.closed)

    @property
    def num_bins(self) -> int:
        return len(self)

    @property
    def left(self) -> NDArray:
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        return self.edges[1:]

    @property
    def mids(self) -> NDArray:
        return (self.left + self.right) / 2.0

    def get_bin_index(self, values: NDArray) -> NDArray[np.int_]:
        return np.digitize(values, self.edges, right=(self.closed == "right")) - 1
