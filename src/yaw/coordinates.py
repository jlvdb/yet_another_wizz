"""
Implements to utility classes to represent a set of angular coordinates or
angular separations in radian.

These classes provide conversion methods between points in 3-dim Euclidean
coordinates and angular coordinates, angular separations and chord distances,
as well as simple coordinate arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sized
from functools import total_ordering
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, TypeVar

    from numpy.typing import ArrayLike, NDArray

    TypeArray = TypeVar("TypeArray", bound="CustomNumpyArray")

__all__ = [
    "AngularCoordinates",
    "AngularDistances",
]


def sgn(val: ArrayLike) -> ArrayLike:
    """Compute the sign of a (array of) numbers, with positive numbers and 0
    returning 1, negative number returning -1."""
    return np.where(val == 0, 1.0, np.sign(val))


class CustomNumpyArray(Iterable, Sized):
    """Meta-class that provides a interface for numpy array routines. Internal
    data is stored as a numpy array in an attribute called ``data``."""

    __slots__ = ("data",)

    data: NDArray
    """Corrdinate array with shape `(N, 2)`."""

    @property
    def __array_interface__(self) -> dict:
        return self.data.__array_interface__

    def __repr__(self) -> str:
        return f"{type(self).__name__}[{len(self)}]"

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self: TypeArray, idx: ArrayLike) -> TypeArray:
        return type(self)(self.data[idx])

    def __iter__(self: TypeArray) -> Iterator[TypeArray]:
        for i in range(len(self)):
            yield self[i]

    def copy(self: TypeArray) -> TypeArray:
        """Create a copy of this instance."""
        return type(self)(self.data.copy())

    def tolist(self) -> list:
        """Convert the underlying data array to a python list."""
        return self.data.tolist()


class AngularCoordinates(CustomNumpyArray):
    """
    Container for angular coordinates in radian.

    Provides convenience methods to convert from and to 3-dim Euclidean (`xyz`)
    coordinates and computing distances between coordinates. Additionally
    implements ``len()``, element-wise comparison with ``==`` operator, iteration
    over the contained coordinates, and indexing/slicing.

    Args:
        data:
            Input coordinates in radian that are broadcastable to a 2-dim numpy
            array with shape `(2, N)`.
    """

    data: NDArray
    """Corrdinate array with shape `(N, 2)`."""

    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_2d(data).astype(np.float64, copy=False)
        if self.data.shape[1] != 2:
            raise ValueError("invalid coordinate dimensions, expected 2")

    @classmethod
    def from_coords(cls, coords: Iterable[AngularCoordinates]) -> AngularCoordinates:
        """
        Concatenate a set of angular coordinates with arbitrary length.

        Args:
            coords:
                Any iterable of :obj:`~yaw.AngularCoordinates` to concatenate.

        Returns:
            New instance of :obj:`~yaw.AngularCoordinates`.
        """
        return cls(np.concatenate(list(coords)))

    @classmethod
    def from_3d(cls, xyz: ArrayLike) -> AngularCoordinates:
        """
        Compute angular coordinates from 3-dim Euclidean (`xyz`) coordinates.

        Args:
            xyz:
                Input Euclidean coordinates that are broadcastable to a 2-dim
                numpy array with shape `(3, N)`.

        Returns:
            New instance of :obj:`~yaw.AngularCoordinates`.
        """
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
        """
        Convert angular to Eudlidean (`xyz`) coordinates.

        Coordinates are projected onto the unit-sphere.

        Returns:
            2-dim numpy array with shape `(3, N)`.
        """
        cos_dec = np.cos(self.dec)
        x = np.cos(self.ra) * cos_dec
        y = np.sin(self.ra) * cos_dec
        z = np.sin(self.dec)
        return np.column_stack([x, y, z])

    @property
    def ra(self) -> NDArray[np.float64]:
        """Accessor for array of right ascension in radian."""
        return self.data[:, 0]

    @property
    def dec(self) -> NDArray[np.float64]:
        """Accessor for array of declination in radian."""
        return self.data[:, 1]

    def __eq__(self, other: Any) -> NDArray[np.bool_]:
        if type(self) is not type(other):
            return NotImplemented

        return self.data == other.data

    def mean(self, weights: ArrayLike | None = None) -> AngularCoordinates:
        """
        Compute the mean angular coordinate.

        The mean is computed in Euclidean coordinates which are converted back
        to angular coordinates.

        Args:
            weights:
                Optional weights for the coordinates when computing the mean.

        Returns:
            :obj:`~yaw.AngularCoordinates` with single coordinate reprensenting
            the mean.
        """
        mean_xyz = np.average(self.to_3d(), weights=weights, axis=0)
        return type(self).from_3d(mean_xyz)

    def distance(self, other: AngularCoordinates) -> AngularDistances:
        """
        Compute the angular distance to another set of angular coordinates.

        The coordinates must have either the same length or

        Args:
            weights:
                Optional weights for the coordinates when computing the mean.

        Returns:
            Angular distance between both sets of coordinates, represented as
            :obj:`~yaw.AngularDistances`.
        """
        if not isinstance(other, type(self)):
            raise TypeError(f"cannot compute distance with type {type(other)}")

        self_xyz = self.to_3d()
        other_xyz = other.to_3d()
        coord_diff_sq = (self_xyz - other_xyz) ** 2
        dists = np.sqrt(coord_diff_sq.sum(axis=1))
        return AngularDistances.from_3d(dists)


@total_ordering
class AngularDistances(CustomNumpyArray):
    """
    Container for angular distances in radian.

    Provides convenience methods to convert from and to 3-dim Euclidean (`xyz`)
    distances and finding minima and maxima. Additionally implements ``len()``,
    element-wise comparison with ``==``/``!=``/``<``/``<=``/``>``/``>=``
    operators, addition with ``+``/``-``, iteration over the contained
    distances, and indexing/slicing.

    Args:
        data:
            Input distances in radian that are broadcastable to a 1-dim numpy
            array.
    """

    data: NDArray
    """Distance array with length `N`."""

    def __init__(self, data: ArrayLike) -> None:
        self.data = np.atleast_1d(data).astype(np.float64, copy=False)

    @classmethod
    def from_dists(cls, dists: Iterable[AngularDistances]) -> AngularDistances:
        """
        Concatenate a set of angular distances with arbitrary length.

        Args:
            dists:
                Any iterable of :obj:`~yaw.AngularDistances` to concatenate.

        Returns:
            New instance of :obj:`~yaw.AngularDistances`.
        """
        return cls(np.concatenate(list(dists)))

    @classmethod
    def from_3d(cls, dists: ArrayLike) -> AngularDistances:
        """
        Convert angular distances from 3-dim Euclidean (`xyz`) distances.

        Assumes that distances are chord distances measured between points on
        the surface of the unit-sphere.

        Args:
            dists:
                Input Euclidean distances that are broadcastable to a 1-dim
                numpy array.

        Returns:
            New instance of :obj:`~yaw.AngularDistances`.

        Raises:
            ValueError:
                If any input distance exceeds 2, the diameter of the unit-sphere.
        """
        if np.any(dists > 2.0):
            raise ValueError("distance exceeds size of unit sphere")

        angles = 2.0 * np.arcsin(dists / 2.0)
        return cls(angles)

    def to_3d(self) -> NDArray:
        """
        Convert angular to Eudlidean (`xyz`) chord distance.

        Returns:
            1-dim numpy array.
        """
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
        """
        Get the minimum contained distance.

        Returns:
            :obj:`~yaw.AngularDistances` with single entry.
        """
        return type(self)(self.data.min())

    def max(self) -> AngularDistances:
        """
        Get the maximum contained distance.

        Returns:
            :obj:`~yaw.AngularDistances` with single entry.
        """
        return type(self)(self.data.max())
