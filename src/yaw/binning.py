from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterator, Sequence
from typing import Any, Literal, TypeVar

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "Binning",
    "parse_binning",
]

Tbinning = TypeVar("Tbinning", bound="Binning")
Tclosed = Literal["left", "right"]
default_closed = "right"


def parse_binning(binning: NDArray | None, *, optional: bool = False) -> NDArray | None:
    if optional and binning is None:
        return None

    binning = np.asarray(binning, dtype=np.float64)
    if np.all(np.diff(binning) > 0.0):
        return binning

    raise ValueError("bin edges must increase monotonically")


@dataclass(frozen=True, eq=False, repr=False, slots=True)
class Binning:
    edges: NDArray
    closed: Tclosed = field(default=default_closed, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "edges", parse_binning(self.edges))

    def __getstate__(self) -> dict:
        return dict(self.edges, self.closed)

    def __len__(self) -> int:
        return len(self.edges) - 1

    def __getitem__(self, item: int | slice) -> Binning:
        left = np.atleast_1d(self.left[item])
        right = np.atleast_1d(self.right[item])
        return np.append(left, right[-1])

    def __iter__(self) -> Iterator[Binning]:
        for i in range(len(self)):
            yield type(self)(self.edges[i:i+2], closed=self.closed)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            raise NotImplemented
        return np.array_equal(self.edges, other.edges) and self.closed == other.closed

    @property
    def mids(self) -> NDArray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    @property
    def left(self) -> NDArray:
        return self.edges[:-1]

    @property
    def right(self) -> NDArray:
        return self.edges[1:]

    @property
    def dz(self) -> NDArray:
        return np.diff(self.edges)

    def copy(self: Tbinning) -> Tbinning:
        return Binning(self.edges.copy(), closed=str(self.closed))
