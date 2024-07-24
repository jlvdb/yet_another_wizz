from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable, Iterator, Sized
from functools import total_ordering
from typing import TYPE_CHECKING, TypeVar

try:
    from typing import Type
except ImportError:
    from typing_extensions import Type

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = ["Coords3D", "CoordsSky", "Dists3D", "DistsSky"]


Twrap = TypeVar("Twrap", bound="NDArrayWrapper")
Tcoord = TypeVar("Tcoord", bound="Coordinates")
Tdist = TypeVar("Tdist", bound="Distances")


def sgn(val: ArrayLike) -> ArrayLike:
    """Compute the sign of a (array of) numbers, with positive numbers and 0
    returning 1, negative number returning -1."""
    return np.where(val == 0, 1.0, np.sign(val))


class NDArrayWrapper(Iterable, Sized):
    values: NDArray

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)}]"

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self: Twrap, idx: ArrayLike) -> Twrap:
        return type(self)(self.values[idx])

    def __iter__(self: Twrap) -> Iterator[Twrap]:
        for i in range(len(self)):
            yield self[i]


class Coordinates(NDArrayWrapper):
    def __init__(self, values: ArrayLike) -> None:
        self.values = np.atleast_2d(values).astype(np.float64, copy=False)
        if not self.values.shape[1] == self.ndim:
            raise ValueError(
                f"invalid number of coordinate dimensions, expected {self.ndim}"
            )

    @classmethod
    @abstractmethod
    def from_coords(cls: Type[Tcoord], coords: Iterable[Coordinates]) -> Tcoord:
        pass

    def __eq__(self, other: object) -> NDArray[np.bool_]:
        if type(self) is type(other):
            return self.values == other.values
        return NotImplemented

    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @abstractmethod
    def mean(self: Tcoord) -> Tcoord:
        pass

    @abstractmethod
    def to_3d(self) -> Coords3D:
        pass

    @abstractmethod
    def to_sky(self) -> CoordsSky:
        pass

    @abstractmethod
    def distance(self, other: Coordinates) -> Dists3D | DistsSky:
        pass


class Coords3D(Coordinates):
    @classmethod
    def from_coords(cls, coords: Iterable[Coordinates]) -> Coords3D:
        values = [coord.to_3d().values for coord in coords]
        return cls(np.concatenate(values))

    @property
    def x(self) -> NDArray[np.float64]:
        return self.values[:, 0]

    @property
    def y(self) -> NDArray[np.float64]:
        return self.values[:, 1]

    @property
    def z(self) -> NDArray[np.float64]:
        return self.values[:, 2]

    @property
    def ndim(self) -> int:
        return 3

    def mean(self) -> Coords3D:
        return Coords3D(self.values.mean(axis=0))

    def to_3d(self) -> Coords3D:
        return self

    def to_sky(self) -> CoordsSky:
        radec = np.empty((len(self), 2))
        x = self.x
        y = self.y
        z = self.z

        r_d2 = np.sqrt(x * x + y * y)
        r_d3 = np.sqrt(x * x + y * y + z * z)
        x_normed = np.ones_like(x)  # fallback for zero-division, arccos(1)=0.0
        np.divide(x, r_d2, where=r_d2 > 0.0, out=x_normed)

        radec[:, 0] = np.arccos(x_normed) * sgn(y) % (2.0 * np.pi)
        radec[:, 1] = np.arcsin(self.z / r_d3)
        return CoordsSky(radec)

    def distance(self, other: Coordinates) -> Dists3D:
        diff = self.values - other.to_3d().values
        diff_sq = diff**2
        dist = np.sqrt(diff_sq.sum(axis=1))
        return Dists3D(dist)


class CoordsSky(Coordinates):
    @classmethod
    def from_coords(cls, coords: Iterable[Coordinates]) -> CoordsSky:
        values = [coord.to_sky().values for coord in coords]
        return cls(np.concatenate(values))

    @property
    def ra(self) -> NDArray[np.float64]:
        return self.values[:, 0]

    @property
    def dec(self) -> NDArray[np.float64]:
        return self.values[:, 1]

    @property
    def ndim(self) -> int:
        return 2

    def mean(self) -> CoordsSky:
        return self.to_3d().mean().to_sky()

    def to_3d(self) -> Coords3D:
        xyz = np.empty((len(self), 3))

        cos_dec = np.cos(self.dec)

        xyz[:, 0] = np.cos(self.ra) * cos_dec
        xyz[:, 1] = np.sin(self.ra) * cos_dec
        xyz[:, 2] = np.sin(self.dec)
        return Coords3D(xyz)

    def to_sky(self) -> CoordsSky:
        return self

    def distance(self, other: Coordinates) -> DistsSky:
        return self.to_3d().distance(other).to_sky()


@total_ordering
class Distances(NDArrayWrapper):
    def __init__(self, values: ArrayLike) -> None:
        self.values = np.atleast_1d(values).astype(np.float64, copy=False)

    @classmethod
    @abstractmethod
    def from_dists(cls: Type[Tcoord], dists: Iterable[Distances]) -> Tcoord:
        pass

    def __eq__(self, other: object) -> NDArray[np.bool_]:
        if type(self) is type(other):
            return self.values == other.values
        return NotImplemented

    def __lt__(self, other: object) -> NDArray[np.bool_]:
        if type(self) is type(other):
            return self.values < other.values
        return NotImplemented

    @abstractmethod
    def __add__(self: Tdist, other: object) -> Tdist:
        pass

    @abstractmethod
    def __sub__(self: Tdist, other: object) -> Tdist:
        pass

    def min(self: Tdist) -> Tdist:
        return type(self)(self.values.min())

    def max(self: Tdist) -> Tdist:
        return type(self)(self.values.max())

    @abstractmethod
    def to_3d(self) -> Dists3D:
        pass

    @abstractmethod
    def to_sky(self) -> DistsSky:
        pass


class Dists3D(Distances):
    @classmethod
    def from_dists(cls, dists: Iterable[Distances]) -> Dists3D:
        values = [dist.to_3d().values for dist in dists]
        return cls(np.concatenate(values))

    def __add__(self, other: object) -> Dists3D:
        if type(self) is type(other):
            return type(self)(self.values + other.values)
        return NotImplemented

    def __sub__(self, other: object) -> Dists3D:
        if type(self) is type(other):
            return type(self)(self.values - other.values)
        return NotImplemented

    def to_3d(self) -> Dists3D:
        return self

    def to_sky(self) -> DistsSky:
        if np.any(self.values > 2.0):
            raise ValueError("distance exceeds size of unit sphere")
        return DistsSky(2.0 * np.arcsin(self.values / 2.0))


class DistsSky(Distances):
    @classmethod
    def from_dists(cls, dists: Iterable[Distances]) -> DistsSky:
        values = [dist.to_sky().values for dist in dists]
        return cls(np.concatenate(values))

    def __add__(self, other: object) -> Dists3D:
        if type(self) is type(other):
            return (self.to_3d() + other.to_3d()).to_sky()
        return NotImplemented

    def __sub__(self, other: object) -> Dists3D:
        if type(self) is type(other):
            return (self.to_3d() - other.to_3d()).to_sky()
        return NotImplemented

    def to_3d(self) -> Dists3D:
        return Dists3D(2.0 * np.sin(self.values / 2.0))

    def to_sky(self) -> DistsSky:
        return self
