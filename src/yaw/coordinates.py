from __future__ import annotations

from collections.abc import Iterable, Iterator, Sized
from functools import total_ordering
from typing import Any, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CoordsSky",
    "Dists3D",
    "DistsSky",
]

Tarray = TypeVar("Tarray", bound="CustomNumpyArray")
Tcoord = TypeVar("Tcoord", bound="AngularCoordinates")
Tdist = TypeVar("Tdist", bound="AngularDistances")


def sgn(val: ArrayLike) -> ArrayLike:
    """Compute the sign of a (array of) numbers, with positive numbers and 0
    returning 1, negative number returning -1."""
    return np.where(val == 0, 1.0, np.sign(val))


def coord_xyz_to_sky(xyz: ArrayLike) -> NDArray:
    x, y, z = np.transpose(xyz)

    r_d2 = np.sqrt(x * x + y * y)
    r_d3 = np.sqrt(x * x + y * y + z * z)
    x_normed = np.ones_like(x)  # fallback for zero-division, arccos(1)=0.0
    np.divide(x, r_d2, where=r_d2 > 0.0, out=x_normed)

    ra = np.arccos(x_normed) * sgn(y) % (2.0 * np.pi)
    dec = np.arcsin(z / r_d3)
    return np.column_stack([ra, dec])


def dist_xyz_to_sky(dist: ArrayLike) -> NDArray:
    if np.any(dist > 2.0):
        raise ValueError("distance exceeds size of unit sphere")
    return 2.0 * np.arcsin(dist / 2.0)


class CustomNumpyArray(Iterable, Sized):
    data: NDArray

    @property
    def __array_interface__(self) -> dict:
        return self.data.__array_interface__

    def __array__(self, dtype=None, copy=None):
        return self.data.__array__()

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)}]"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self: Tarray, idx: ArrayLike) -> Tarray:
        return type(self)(self.data[idx])

    def __iter__(self: Tarray) -> Iterator[Tarray]:
        for i in range(len(self)):
            yield self[i]

    def tolist(self) -> list:
        return self.data.tolist()


class AngularCoordinates(CustomNumpyArray):
    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_2d(data).astype(np.float64, copy=False)
        if not self.data.shape[1] == 2:
            raise ValueError(f"invalid coordinate dimensions, expected 2")

    @classmethod
    def from_coords(cls: type[Tcoord], coords: Iterable[AngularCoordinates]) -> Tcoord:
        return cls([coord.to_sky().data for coord in coords])

    @classmethod
    def from_3d(cls: type[Tcoord], xyz: ArrayLike) -> Tcoord:
        return cls(coord_xyz_to_sky(xyz))

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

    def mean(self: Tcoord) -> Tcoord:
        mean_xyz = self.to_3d().mean(axis=0)
        mean = coord_xyz_to_sky(mean_xyz)
        return type(self)(mean)

    def distance(self, other: AngularCoordinates) -> AngularDistances:
        if not isinstance(other, type(self)):
            raise TypeError(f"cannot compute distance with type {type(other)}")

        self_xyz = self.to_3d()
        other_xyz = other.to_3d()
        coord_diff_sq = (self_xyz - other_xyz)**2
        dists = np.sqrt(coord_diff_sq.sum(axis=1))
        return AngularDistances.from_3d(dists)


@total_ordering
class AngularDistances(CustomNumpyArray):
    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_1d(data).astype(np.float64, copy=False)

    @classmethod
    def from_dists(cls: type[Tdist], dists: Iterable[AngularDistances]) -> Tdist:
        return cls(np.concatenate([dist.to_sky() for dist in dists]))

    @classmethod
    def from_3d(cls: type[Tcoord], dists: ArrayLike) -> Tcoord:
        return cls(dist_xyz_to_sky(dists))

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

    def __add__(self: Tdist, other: Any) -> Tdist:
        if type(self) is not type(other):
            return NotImplemented

        return dist_xyz_to_sky(self.to_3d() + other.to_3d())

    def __sub__(self: Tdist, other: Any) -> Tdist:
        if type(self) is not type(other):
            return NotImplemented
    
        return dist_xyz_to_sky(self.to_3d() - other.to_3d())

    def min(self: Tdist) -> Tdist:
        return type(self)(self.data.min())

    def max(self: Tdist) -> Tdist:
        return type(self)(self.data.max())
