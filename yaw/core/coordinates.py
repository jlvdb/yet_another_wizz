from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from functools import total_ordering
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray


def sgn(val: ArrayLike) -> ArrayLike:
    return np.where(val == 0, 1.0, np.sign(val))


class Coordinate(ABC):

    @abstractmethod
    def __getitem__(self, idx: ArrayLike) -> Coordinate: pass

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

    __slots__ = ("x", "y", "z")

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> None:
        self.x = np.float_(x)
        self.y = np.float_(y)
        self.z = np.float_(z)

    def __repr__(self) -> str:
        digits = 6
        x = np.round(self.x, digits)
        y = np.round(self.y, digits)
        z = np.round(self.z, digits)
        return f"{self.__class__.__name__}({x=}, {y=}, {z=})"

    def __getitem__(self, idx: ArrayLike) -> Coordinate:
        return Coord3D(x=self.x[idx], y=self.y[idx], z=self.z[idx])

    @property
    def values(self) -> NDArray[np.float_]:
        return np.transpose([self.x, self.y, self.z])

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
        isnan = np.isnan(ra)
        if isnan.any():
            if np.isscalar(isnan):
                ra = 0.0
            else:
                ra[np.isnan(ra)] = 0.0
        dec = np.arcsin(self.z / r_d3)
        return CoordSky(ra, dec)

    def distance(self, other: Coord3D) -> Dist3D:
        return Dist3D(np.sqrt(
            (self.x - other.x)**2 +
            (self.y - other.y)**2 +
            (self.z - other.z)**2))


class CoordSky(Coordinate):

    __slots__ = ("ra", "dec")

    def __init__(self, ra: ArrayLike, dec: ArrayLike) -> None:
        self.ra = np.float_(ra)
        self.dec = np.float_(dec)

    def __repr__(self) -> str:
        digits = 6
        ra = np.round(self.ra, digits)
        dec = np.round(self.dec, digits)
        return f"{self.__class__.__name__}({ra=}, {dec=})"

    def __getitem__(self, idx: ArrayLike) -> CoordSky:
        return CoordSky(ra=self.ra[idx], dec=self.dec[idx])

    @property
    def values(self) -> NDArray[np.float_]:
        return np.transpose([self.ra, self.dec])

    def mean(self) -> CoordSky:
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
class Distance(ABC):

    __slots__ = ("values",)

    def __init__(self, distance: ArrayLike) -> None:
        self.values = np.float_(distance)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.values})"

    def __format__(self, __format_spec: str) -> str:
        return self.values.__format__(__format_spec)

    @abstractmethod
    def __add__(self, other: Distance) -> Distance: pass

    @abstractmethod
    def __sub__(self, other: Distance) -> Distance: pass

    def __eq__(self, other: Distance) -> ArrayLike[np.bool_]:
        return self.values == other.values

    def __lt__(self, other: Distance) -> ArrayLike[np.bool_]:
        return self.values < other.values


class Dist3D(Distance):

    def __add__(self, other: Dist3D) -> DistSky:
        dist_sky = self.to_sky() + other.to_sky()
        return dist_sky.to_3d()

    def __sub__(self, other: Dist3D) -> DistSky:
        dist_sky = self.to_sky() - other.to_sky()
        return dist_sky.to_3d()

    def to_sky(self) -> DistSky:
        if np.any(self.values > 2.0):
            raise ValueError("distance exceeds size of unit sphere")
        return DistSky(2.0 * np.arcsin(self.values / 2.0))


class DistSky(Distance):

    def __add__(self, other: DistSky) -> DistSky:
        return DistSky(self.values + other.values)

    def __sub__(self, other: DistSky) -> DistSky:
        return DistSky(self.values - other.values)

    def to_3d(self) -> Dist3D:
        return Dist3D(2.0 * np.sin(self.values / 2.0))
