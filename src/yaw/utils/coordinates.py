from __future__ import annotations

from collections.abc import Iterable, Sized
from functools import total_ordering
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, TypeVar

    from numpy.typing import ArrayLike, NDArray

    Tarray = TypeVar("Tarray", bound="CustomNumpyArray")

__all__ = [
    "AngularCoordinates",
    "AngularDistances",
]


def sgn(val: ArrayLike) -> ArrayLike:
    """Compute the sign of a (array of) numbers, with positive numbers and 0
    returning 1, negative number returning -1."""
    return np.where(val == 0, 1.0, np.sign(val))


class CustomNumpyArray(Iterable, Sized):
    __slots__ = ("data",)

    data: NDArray

    @property
    def __array_interface__(self) -> dict:
        return self.data.__array_interface__

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)}]"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self: Tarray, idx: ArrayLike) -> Tarray:
        return type(self)(self.data[idx])

    def __iter__(self: Tarray) -> Iterator[Tarray]:
        for i in range(len(self)):
            yield self[i]

    def copy(self: Tarray) -> Tarray:
        return type(self)(self.data.copy())

    def tolist(self) -> list:
        return self.data.tolist()


class AngularCoordinates(CustomNumpyArray):
    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_2d(data).astype(np.float64, copy=False)
        if self.data.shape[1] != 2:
            raise ValueError("invalid coordinate dimensions, expected 2")

    @classmethod
    def from_coords(cls, coords: Iterable[AngularCoordinates]) -> AngularCoordinates:
        return cls(np.concatenate(list(coords)))

    @classmethod
    def from_3d(cls, xyz: ArrayLike) -> AngularCoordinates:
        x, y, z = np.transpose(np.atleast_2d(xyz))

        r_d2 = np.sqrt(x * x + y * y)
        r_d3 = np.sqrt(x * x + y * y + z * z)
        x_normed = np.ones_like(x)  # fallback for zero-division, arccos(1)=0.0
        np.divide(x, r_d2, where=r_d2 > 0.0, out=x_normed)

        ra = np.arccos(x_normed) * sgn(y) % (2.0 * np.pi)
        dec = np.arcsin(z / r_d3)
        ra_dec = np.column_stack([ra, dec])
        return cls(ra_dec)

    def to_3d(self) -> NDArray:
        cos_dec = np.cos(self.dec)
        x = np.cos(self.ra) * cos_dec
        y = np.sin(self.ra) * cos_dec
        z = np.sin(self.dec)
        return np.column_stack([x, y, z])

    @property
    def ra(self) -> NDArray[np.float64]:
        return self.data[:, 0]

    @property
    def dec(self) -> NDArray[np.float64]:
        return self.data[:, 1]

    def __eq__(self, other: Any) -> NDArray[np.bool_]:
        if type(self) is not type(other):
            return NotImplemented

        return self.data == other.data

    def mean(self, weights: ArrayLike | None = None) -> AngularCoordinates:
        mean_xyz = np.average(self.to_3d(), weights=weights, axis=0)
        return type(self).from_3d(mean_xyz)

    def distance(self, other: AngularCoordinates) -> AngularDistances:
        if not isinstance(other, type(self)):
            raise TypeError(f"cannot compute distance with type {type(other)}")

        self_xyz = self.to_3d()
        other_xyz = other.to_3d()
        coord_diff_sq = (self_xyz - other_xyz) ** 2
        dists = np.sqrt(coord_diff_sq.sum(axis=1))
        return AngularDistances.from_3d(dists)


@total_ordering
class AngularDistances(CustomNumpyArray):
    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_1d(data).astype(np.float64, copy=False)

    @classmethod
    def from_dists(cls, dists: Iterable[AngularDistances]) -> AngularDistances:
        return cls(np.concatenate(list(dists)))

    @classmethod
    def from_3d(cls, dists: ArrayLike) -> AngularDistances:
        if np.any(dists > 2.0):
            raise ValueError("distance exceeds size of unit sphere")

        angles = 2.0 * np.arcsin(dists / 2.0)
        return cls(angles)

    def to_3d(self) -> NDArray:
        return 2.0 * np.sin(self.data / 2.0)

    def __eq__(self, other: Any) -> NDArray[np.bool_]:
        if type(self) is not type(other):
            return NotImplemented

        return self.data == other.data

    def __lt__(self, other: Any) -> NDArray[np.bool_]:
        if type(self) is not type(other):
            return NotImplemented

        return self.data < other.data

    def __add__(self, other: Any) -> AngularDistances:
        if type(self) is not type(other):
            return NotImplemented

        return type(self)(self.data + other.data)

    def __sub__(self, other: Any) -> AngularDistances:
        if type(self) is not type(other):
            return NotImplemented

        return type(self)(self.data - other.data)

    def min(self) -> AngularDistances:
        return type(self)(self.data.min())

    def max(self) -> AngularDistances:
        return type(self)(self.data.max())
