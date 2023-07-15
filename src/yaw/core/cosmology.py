"""This module defines an interface for custom cosmological models, as well as
routines that depend on cosmological distance calculations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

try:  # pragma: no cover
    from typing import TypeAlias
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

import numpy as np
from astropy.cosmology import FLRW, Planck15

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, NDArray

__all__ = [
    "get_default_cosmology",
    "CustomCosmology",
    "r_kpc_to_angle",
    "Scale",
    "BinFactory",
]


def get_default_cosmology() -> FLRW:
    """Get the default cosmology (Planck Collaboration et al. 2015)."""
    return Planck15


class CustomCosmology(ABC):
    """Metaclass that defines the API to implement a custom cosmological model.

    The two required methods should behave like the corresponding methods in
    :obj:`astropy.cosmology.FLRW`.
    """

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        """Comoving line-of-sight distance in Mpc at a given redshift.

        The comoving distance along the line-of-sight between two objects
        remains constant with time for objects in the Hubble flow.

        Args:
            z (Quantity-like ``redshift``, :obj:`NDArray`, or scalar number):
                Input redshift.

        Returns:
            :obj:`astropy.units.Quantity`:
                Comoving distance in Mpc to each input redshift.
        """
        pass

    @abstractmethod
    def comoving_transverse_distance(self, z: ArrayLike) -> ArrayLike:
        """Comoving transverse distance in Mpc at a given redshift.

        This value is the transverse comoving distance at redshift :math:`z`
        corresponding to an angular separation of 1 radian. This is the same as
        the comoving distance if :math:`\\Omega_k` is zero (as in the current
        concordance Lambda-CDM model).

        Args:
            z (Quantity-like ``redshift``, :obj:`NDArray`, or scalar number):
                Input redshift.

        Returns:
            :obj:`astropy.units.Quantity`:
                Comoving transverse distance in Mpc at each input redshift.
        """
        pass


TypeCosmology: TypeAlias = Union[FLRW, CustomCosmology]


def r_kpc_to_angle(
    r_kpc: NDArray[np.float_] | Sequence[float], z: float, cosmology: TypeCosmology
) -> NDArray[np.float_]:
    """Convert from a physical separation in kpc to angles in radian at a given
    redshift.

    Args:
        r_kpc (:obj:`NDArray`, number):
            Physical separation in kpc.
        z (:obj:`float`):
            Redshift at which conversion happens.
        cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`CustomCosmology`):
            Cosmological model used for calculations.

    Returns:
        :obj:`NDArray`:
            Angular separation at given redshift.
    """
    f_K = cosmology.comoving_transverse_distance(z)  # for 1 radian in Mpc
    return np.asarray(r_kpc) / 1000.0 * (1.0 + z) / f_K.value


@dataclass(frozen=True)
class Scale:
    """Class that represents a range of physical scales in kpc.

    The range is defined by a lower and an upper scale limit. An instance can be
    converted to a string representation that is used as dictionary key in some
    places of ``yaw``.

    .. rubric:: Examples

    Get the center points:

    >>> scale = Scale(100, 1000)
    >>> scale.mid, scale.mid_log
    (550.0, 316.22776601683796)

    String representation:

    >>> str(scale)
    'kpc100t1000'
    """

    rmin: float
    """Lower scale limit in kpc."""
    rmax: float
    """Upper scale limit in kpc."""

    def __post_init__(self) -> None:
        if self.rmin >= self.rmax:
            raise ValueError("'rmin' must be less than 'rmax'")

    @property
    def mid(self) -> float:
        """The mid point of the scale range."""
        return (self.rmin + self.rmax) / 2.0

    @property
    def mid_log(self) -> float:
        """The logarithmic (base 10) mid point of the scale range."""
        lmin = np.log10(self.rmin)
        lmax = np.log10(self.rmax)
        lmid = (lmin + lmax) / 2.0
        return 10**lmid

    def __str__(self) -> str:
        return f"kpc{self.rmin:.0f}t{self.rmax:.0f}"

    def to_radian(self, z: float, cosmology: TypeCosmology) -> NDArray[np.float_]:
        """Get the separation in radian at a given redshift.

        Args:
            z (:obj:`float`):
                Redshift at which conversion happens.
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`CustomCosmology`):
                Cosmological model used for calculations.

        Returns:
            :obj:`NDArray`:
                Angular separation at given redshift.
        """
        return r_kpc_to_angle([self.rmin, self.rmax], z, cosmology)


class BinFactory:
    """Class used to generate redshift bins."""

    def __init__(
        self,
        zmin: float,
        zmax: float,
        nbins: int,
        cosmology: TypeCosmology | None = None,
    ):
        """Create a new bin generator.

        Args:
            zmin (:obj:`float`):
                Minimum redshift, lowest redshift bin edges.
            zmax (:obj:`float`):
                Maximum redshift, lowest redshift bin edges.
            nbins (:obj:`int`):
                Number of bins to generate.
            cosmology (:obj:`astropy.cosmology.FLRW`, :obj:`CustomCosmology`):
                Cosmological model used for calculations.
        """
        if zmin >= zmax:
            raise ValueError("'zmin' >= 'zmax'")
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins

    def linear(self) -> NDArray[np.float_]:
        """Generate a binning with equal width in redshift."""
        return np.linspace(self.zmin, self.zmax, self.nbins + 1)

    def comoving(self) -> NDArray[np.float_]:
        """Generate a binning with equal width in radial comoving distance."""
        cbinning = np.linspace(
            self.cosmology.comoving_distance(self.zmin).value,
            self.cosmology.comoving_distance(self.zmax).value,
            self.nbins + 1,
        )
        # construct a spline mapping from comoving distance to redshift
        zarray = np.linspace(0, 10.0, 5000)
        carray = self.cosmology.comoving_distance(zarray).value
        return np.interp(cbinning, xp=carray, fp=zarray)  # redshift @ cbinning

    def logspace(self) -> NDArray[np.float_]:
        """Generate a binning with equal width in logarithmic redshift
        :math:`\\log(1+z)`."""
        logbinning = np.linspace(
            np.log(1.0 + self.zmin), np.log(1.0 + self.zmax), self.nbins + 1
        )
        return np.exp(logbinning) - 1.0

    @staticmethod
    def check(zbins: NDArray[np.float_]) -> None:
        """Check if a list of bin edges in monotonicaly increasing.

        Raises a :exc:`ValueError` if the condition is not met.

        Args:
            zbins (:obj:`NDArray`):
                Redshift bin edges to check.
        """
        if np.any(np.diff(zbins) <= 0):
            raise ValueError("redshift bins must be monotonicaly increasing")

    def get(self, method: str) -> NDArray[np.float_]:
        """Call one of the generation methods based on its name.

        Args:
            method (:obj:`str`):
                Name of the generation method.
        """
        try:
            return getattr(self, method)()
        except AttributeError as e:
            raise ValueError(f"invalid binning method '{method}'") from e
