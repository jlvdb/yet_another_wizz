"""This module defines simple containers for coordiantes and distances. There
are currently two flavours, 3-dim Euclidean coordinates and distances, as well
as angular coordinates and distances in radian.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from functools import total_ordering
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from yaw.core.math import sgn

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = ["Coord3D", "CoordSky", "Dist3D", "DistSky"]


class Coordinate(ABC):
    """Base class for a vector of coordinates."""

    @abstractmethod
    def __init__(self, coords: dict[str, ArrayLike]) -> None:
        pass

    @classmethod
    @abstractmethod
    def from_array(cls, array) -> Coordinate:
        pass

    @classmethod
    @abstractmethod
    def from_coords(cls, coords: Sequence[Coordinate]) -> Coordinate:
        """Concatenate a sequence of coordinates into a new vector of
        coordates."""
        pass

    def __repr__(self) -> str:
        pass

    @property
    @abstractmethod
    def dim(self) -> tuple[str]:
        """A list of names of coordinates in the coordinate system."""
        pass

    @property
    def ndim(self) -> int:
        """The number of coordinates in the coordinate system."""
        return len(self.dim)

    @property
    @abstractmethod
    def values(self) -> NDArray[np.float_]:
        """The coordinate values cast into a numpy array with shape
        `(N, ndim)` or `(ndim,)` if there is only a single entry."""
        pass

    @abstractmethod
    def mean(self) -> Coordinate:
        """The mean coordinate (mean over all dimensions)."""
        pass

    @abstractmethod
    def to_3d(self) -> Coord3D:
        """Get the coordinates as 3-dim Euclidean coordiante :obj:`Coord3D`."""
        pass

    @abstractmethod
    def to_sky(self) -> CoordSky:
        """Get the coordinates as angular coordiante :obj:`CoordSky` in radian.

        .. Warning::
            During conversion, points loose their radial information. After
            back-transformation to :obj:`Coord3D` they lie on the unit sphere.
        """
        pass

    @abstractmethod
    def distance(self, other: Coordinate) -> Distance:
        pass


class Coord3D(Coordinate):
    """A representation of a vector of 3-dim Euclidean coordiantes (x, y, z)."""

    x: NDArray
    """The `x`-coordiante(s)."""
    y: NDArray
    """The `y`-coordiante(s)."""
    z: NDArray
    """The `z`-coordiante(s)."""

    def __init__(self, x: ArrayLike, y: ArrayLike, z: ArrayLike) -> None:
        """Create a new coordinate vector.

        Args:
            x (:obj:`float`, :obj:`NDArray`):
                The `x`-coordiante(s).
            y (:obj:`float`, :obj:`NDArray`):
                The `y`-coordiante(s).
            z (:obj:`float`, :obj:`NDArray`):
                The `z`-coordiante(s).

        .. Note::
            The coordinate vector has a length, supports numpy-style indexing
            and iteration over the elements (always returning new coordinate
            vector instances).
        """
        self.x = np.atleast_1d(x).astype(np.float_)
        self.y = np.atleast_1d(y).astype(np.float_)
        self.z = np.atleast_1d(z).astype(np.float_)

    @classmethod
    def from_array(cls, array) -> Coord3D:
        """Create a new coordinate vector from an array of tuples of
        (`x`, `y`, `z`)."""
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
        r_d2 = np.sqrt(x * x + y * y)
        r_d3 = np.sqrt(x * x + y * y + z * z)
        # transform
        x_normed = np.ones_like(x)  # fallback for zero-division, arccos(1)=0.0
        np.divide(x, r_d2, where=r_d2 > 0.0, out=x_normed)
        ra = np.arccos(x_normed) * sgn(y) % (2.0 * np.pi)
        dec = np.arcsin(self.z / r_d3)
        return CoordSky(ra, dec)

    def distance(self, other: Coordinate) -> Dist3D:
        """Compute the Euclidean distance between two coordinate vectors.

        Coordinates are automatically converted before distance calculation.

        Args:
            other (:obj:`Coordinate`):
                Second coordinate vector.

        Returns:
            :obj:`Dist3D`:
                Euclidean distance between points in this and the other vector.
        """
        c1 = self.to_3d()
        c2 = other.to_3d()
        return Dist3D(
            np.sqrt((c1.x - c2.x) ** 2 + (c1.y - c2.y) ** 2 + (c1.z - c2.z) ** 2)
        )


class CoordSky(Coordinate):
    """A representation of a vector of angular coordinates in radian.

    Angles follow the astronomical convention of R.A./Dec., i.e. declination
    values are in the range of [:math:`-\\pi`, :math:`\\pi`].
    """

    ra: NDArray
    """The right ascension coordinate(s)."""
    dec: NDArray
    """The declination coordinate(s)."""

    def __init__(self, ra: ArrayLike, dec: ArrayLike) -> None:
        """Create a new coordinate vector.

        Args:
            ra (:obj:`float`, :obj:`NDArray`):
                The right ascension coordinate(s).
            dec (:obj:`float`, :obj:`NDArray`):
                The declination coordinate(s).

        .. Note::
            The coordinate vector has a length, supports numpy-style indexing
            and iteration over the elements (always returning new coordinate
            vector instances).
        """
        self.ra = np.atleast_1d(ra).astype(np.float_)
        self.dec = np.atleast_1d(dec).astype(np.float_)

    @classmethod
    def from_array(cls, array):
        """Create a new coordinate vector from an array of tuples of
        (`ra`, `dec`)."""
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
            x=np.cos(self.ra) * cos_dec, y=np.sin(self.ra) * cos_dec, z=np.sin(self.dec)
        )

    def to_sky(self) -> CoordSky:
        return self

    def distance(self, other: Coordinate) -> DistSky:
        """Compute the angular distance in radian between two coordinate
        vectors.

        Coordinates are automatically converted before distance calculation.

        Args:
            other (:obj:`Coordinate`):
                Second coordinate vector.

        Returns:
            :obj:`DistSky`:
                Angular distance in radian between points in this and the other
                vector.
        """
        # lazy shortcut
        self_3D = self.to_3d()
        other_3D = other.to_3d()
        dist = self_3D.distance(other_3D)
        return dist.to_sky()


_Tdist = TypeVar("_Tdist", bound="Distance")


@total_ordering
class Distance(ABC):
    """Base class for a vector of distances."""

    def __init__(self, distance: ArrayLike) -> None:
        """Constructs a new vector of distances.

        Args:
            distance (:obj:`float`, :obj:`NDArray`):
                Distance values.

        .. Note::
            The coordinate vector has a length, supports numpy-style indexing
            and iteration over the elements (always returning new coordinate
            vector instances).

            Additionally, distances implement element-wise addition and
            subtraction, as well as the comparison operators.
        """
        self._values = np.atleast_1d(distance).astype(np.float_)

    @classmethod
    def from_dists(cls: _Tdist, dists: Sequence[_Tdist]) -> _Tdist:
        """Concatenate a sequence of distances into a new vector of distances."""
        return cls(np.concatenate([dist._values for dist in dists], axis=0))

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, idx: ArrayLike) -> CoordSky:
        return self.__class__(self._values[idx])

    def __iter__(self) -> Iterator[CoordSky]:
        for i in range(len(self)):
            yield self[i]

    @property
    def values(self) -> NDArray[np.float_] | float:
        """The distances cast into a numpy array with shape `(N,)` or a scalar
        if there is only a single distance."""
        return np.squeeze(self._values)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.values})"

    def __format__(self, __format_spec: str) -> str:
        return self.values.__format__(__format_spec)

    def __eq__(self, other: object) -> ArrayLike[np.bool_]:
        if isinstance(other, self.__class__):
            return self.values == other.values
        return NotImplemented

    def __lt__(self, other: Distance) -> ArrayLike[np.bool_]:
        if isinstance(other, self.__class__):
            return self.values < other.values
        return NotImplemented

    @abstractmethod
    def __add__(self, other: object) -> _Tdist:
        pass

    @abstractmethod
    def __sub__(self, other: object) -> _Tdist:
        pass

    def min(self) -> _Tdist:
        """Compute the minimum value and return it as new `Distance`
        instance."""
        return self.__class__(self.values.min())

    def max(self) -> _Tdist:
        """Compute the maximum value and return it as new `Distance`
        instance."""
        return self.__class__(self.values.max())

    @abstractmethod
    def to_3d(self) -> Dist3D:
        """Convert the distance to the Euclidean distance."""
        pass

    @abstractmethod
    def to_sky(self) -> DistSky:
        """Convert the distance to an angular separation in radian."""
        pass


class Dist3D(Distance):
    """Implements a vector of Euclidean distances."""

    def __add__(self, other: object) -> DistSky:
        if isinstance(other, Dist3D):
            dist_sky = self.to_sky() + other.to_sky()
            return dist_sky.to_3d()
        return NotImplemented

    def __sub__(self, other: object) -> DistSky:
        if isinstance(other, Dist3D):
            dist_sky = self.to_sky() - other.to_sky()
            return dist_sky.to_3d()
        return NotImplemented

    def to_3d(self) -> Dist3D:
        return self

    def to_sky(self) -> DistSky:
        if np.any(self._values > 2.0):
            raise ValueError("distance exceeds size of unit sphere")
        return DistSky(2.0 * np.arcsin(self.values / 2.0))


class DistSky(Distance):
    """Implements a vector of angular distances in radian."""

    def __add__(self, other: object) -> DistSky:
        if isinstance(other, DistSky):
            return DistSky(self.values + other.values)
        return NotImplemented

    def __sub__(self, other: object) -> DistSky:
        if isinstance(other, DistSky):
            return DistSky(self.values - other.values)
        return NotImplemented

    def to_3d(self) -> Dist3D:
        return Dist3D(2.0 * np.sin(self.values / 2.0))

    def to_sky(self) -> DistSky:
        return self
