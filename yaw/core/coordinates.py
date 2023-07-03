from __future__ import annotations

from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence
from functools import total_ordering
from typing import TYPE_CHECKING

import numpy as np

from yaw.core.math import sgn

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray


class Coordinate(Sequence, ABC):

    @abstractmethod
    def __init__(self, coords: dict[str, ArrayLike]) -> None: pass

    @abstractclassmethod
    def from_array(cls, array) -> Coordinate: pass

    @abstractclassmethod
    def from_coords(cls, coords: Sequence[Coordinate]) -> Coordinate: pass

    def __repr__(self) -> str: pass

    @abstractproperty
    def dim(self) -> tuple[str]: pass

    @property
    def ndim(self) -> int:
        return len(self.dim)

    @abstractproperty
    def values(self) -> NDArray[np.float_]: pass

    @abstractmethod
    def mean(self) -> Coordinate: pass

    @abstractmethod
    def to_3d(self) -> Coord3D: pass

    @abstractmethod
    def to_sky(self) -> CoordSky: pass

    @abstractmethod
    def distance(self) -> Distance: pass


class Coord3D(Coordinate):

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> None:
        self.x = np.atleast_1d(x).astype(np.float_)
        self.y = np.atleast_1d(y).astype(np.float_)
        self.z = np.atleast_1d(z).astype(np.float_)

    @classmethod
    def from_array(cls, array) -> Coord3D:
        x, y, z = np.transpose(array)
        return cls(x, y, z)

    @classmethod
    def from_coords(cls, coords: Sequence[Coord3D]) -> Coord3D:
        x = np.concatenate([coord.x for coord in coords], axis=0)
        y = np.concatenate([coord.y for coord in coords], axis=0)
        z = np.concatenate([coord.z for coord in coords], axis=0)
        return cls(x, y, z)

    def __repr__(self) -> str:
        x, y, z = self.x, self.y, self.z
        return f"{self.__class__.__name__}({x=}, {y=}, {z=})"

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: ArrayLike) -> Coord3D:
        return self.__class__(x=self.x[idx], y=self.y[idx], z=self.z[idx])

    def __iter__(self) -> Iterator[Coord3D]:
        for i in range(len(self)):
            yield self[i]

    @property
    def dim(self) -> tuple[str]:
        return ("x", "y", "z")

    @property
    def values(self) -> NDArray[np.float_]:
        return np.squeeze(np.transpose([self.x, self.y, self.z]))

    def mean(self) -> Coord3D:
        return Coord3D(x=self.x.mean(), y=self.y.mean(), z=self.z.mean())

    def to_3d(self) -> Coord3D:
        return self

    def to_sky(self) -> CoordSky:
        x = self.x
        y = self.y
        z = self.z
        r_d2 = np.sqrt(x*x + y*y)
        r_d3 = np.sqrt(x*x + y*y + z*z)
        # transform
        ra = np.arccos(x / r_d2) * sgn(y) % (2.0*np.pi)
        ra[np.isnan(ra)] = 0.0
        dec = np.arcsin(self.z / r_d3)
        return CoordSky(ra, dec)

    def distance(self, other: Coord3D) -> Dist3D:
        c1 = self.to_3d()
        c2 = other.to_3d()
        return Dist3D(np.sqrt(
            (c1.x - c2.x)**2 +
            (c1.y - c2.y)**2 +
            (c1.z - c2.z)**2))


class CoordSky(Coordinate):

    def __init__(self, ra: ArrayLike, dec: ArrayLike) -> None:
        self.ra = np.atleast_1d(ra).astype(np.float_)
        self.dec = np.atleast_1d(dec).astype(np.float_)

    @classmethod
    def from_array(cls, array):
        ra, dec = np.transpose(array)
        return cls(ra, dec)

    @classmethod
    def from_coords(cls, coords: Sequence[CoordSky]) -> CoordSky:
        ra = np.concatenate([coord.ra for coord in coords], axis=0)
        dec = np.concatenate([coord.dec for coord in coords], axis=0)
        return cls(ra, dec)

    def __repr__(self) -> str:
        ra, dec = self.ra, self.dec
        return f"{self.__class__.__name__}({ra=}, {dec=})"

    def __len__(self) -> int:
        return len(self.ra)

    def __getitem__(self, idx: ArrayLike) -> CoordSky:
        return self.__class__(ra=self.ra[idx], dec=self.dec[idx])

    def __iter__(self) -> Iterator[CoordSky]:
        for i in range(len(self)):
            yield self[i]

    @property
    def dim(self) -> tuple[str]:
        return ("ra", "dec")

    @property
    def values(self) -> NDArray[np.float_]:
        return np.squeeze(np.transpose([self.ra, self.dec]))

    def mean(self) -> Coord3D:
        return self.to_3d().mean().to_sky()

    def to_3d(self) -> Coord3D:
        cos_dec = np.cos(self.dec)
        return Coord3D(
            x=np.cos(self.ra) * cos_dec,
            y=np.sin(self.ra) * cos_dec,
            z=np.sin(self.dec))

    def to_sky(self) -> CoordSky:
        return self

    def distance(self, other: CoordSky) -> DistSky:
        # lazy shortcut
        self_3D = self.to_3d()
        other_3D = other.to_3d()
        dist = self_3D.distance(other_3D)
        return dist.to_sky()


@total_ordering
class Distance(ABC, Sequence):

    def __init__(self, distance: ArrayLike) -> None:
        self._values = np.atleast_1d(distance).astype(np.float_)

    @classmethod
    def from_dists(cls, dists: Sequence[Distance]) -> Distance:
        return cls(np.concatenate([dist._values for dist in dists], axis=0))

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: ArrayLike) -> CoordSky:
        return self.__class__(self._values[idx])

    def __iter__(self) -> Iterator[CoordSky]:
        for i in range(len(self)):
            yield self[i]

    @property
    def values(self) -> NDArray[np.float_]:
        return np.squeeze(self._values)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.values})"

    def __format__(self, __format_spec: str) -> str:
        return self.values.__format__(__format_spec)

    def __eq__(self, other: Distance) -> ArrayLike[np.bool_]:
        return self.values == other.values

    def __lt__(self, other: Distance) -> ArrayLike[np.bool_]:
        return self.values < other.values

    @abstractmethod
    def __add__(self, other: Distance) -> Distance: pass

    @abstractmethod
    def __sub__(self, other: Distance) -> Distance: pass

    def min(self) -> Distance:
        return self.__class__(self.values.min())

    def max(self) -> Distance:
        return self.__class__(self.values.max())

    @abstractmethod
    def to_3d(self) -> Dist3D: pass

    @abstractmethod
    def to_sky(self) -> DistSky: pass


class Dist3D(Distance):

    def __add__(self, other: Dist3D) -> DistSky:
        dist_sky = self.to_sky() + other.to_sky()
        return dist_sky.to_3d()

    def __sub__(self, other: Dist3D) -> DistSky:
        dist_sky = self.to_sky() - other.to_sky()
        return dist_sky.to_3d()

    def to_3d(self) -> Dist3D:
        return self

    def to_sky(self) -> DistSky:
        if np.any(self._values > 2.0):
            raise ValueError("distance exceeds size of unit sphere")
        return DistSky(2.0 * np.arcsin(self.values / 2.0))


class DistSky(Distance):

    def __add__(self, other: DistSky) -> DistSky:
        return DistSky(self.values + other.values)

    def __sub__(self, other: DistSky) -> DistSky:
        return DistSky(self.values - other.values)

    def to_3d(self) -> Dist3D:
        return Dist3D(2.0 * np.sin(self.values / 2.0))

    def to_sky(self) -> DistSky:
        return self
