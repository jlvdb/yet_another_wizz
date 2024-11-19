"""
Implements some utilities for cosmological distance computations.

Most notably a function to convert angular diameter distances to angles at a
given redshift and a given cosmological model. By default, uses named
cosmologies from ``astropy``, but also provides a base-class to implement custom
cosmological models. Note that the cosmology typically has a minor impact on the
derive clustering redshifts.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy.cosmology import FLRW, Planck15, cosmology_equal
from astropy.units import Quantity

from yaw.options import Unit

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

TypeCosmology = Union[FLRW, "CustomCosmology"]  # used with get_args

__all__ = [
    "CustomCosmology",
    "cosmology_is_equal",
    "get_default_cosmology",
    "separation_physical_to_angle",
]


class CustomCosmology(ABC):
    """Meta-class that defines interface for custom cosmologies that are
    compatible with `yet_another_wizz` code."""

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        """
        Compute the comoving distance.

        Args:
            z:
                A single or an array of redshifts.

        Returns:
            Float or numpy array with comoving distance for given input
            redshifts.
        """
        pass

    @abstractmethod
    def angular_diameter_distance(self, z: ArrayLike) -> ArrayLike:
        """
        Compute the angular diameter distance.

        Args:
            z:
                A single or an array of redshifts.

        Returns:
            Float or numpy array with angular diameter distance for given input
            redshifts.
        """
        pass


def cosmology_is_equal(cosmo1: TypeCosmology, cosmo2: TypeCosmology) -> bool:
    """Compare if two ``astropy`` cosmologies are equal. Always ``True`` for
    instances of ``CustomCosmology``."""
    if not isinstance(cosmo1, (FLRW, CustomCosmology)):
        raise TypeError("'cosmo1' is not a valid cosmology type")
    if not isinstance(cosmo2, (FLRW, CustomCosmology)):
        raise TypeError("'cosmo2' is not a valid cosmology type")

    is_custom_1 = isinstance(cosmo1, CustomCosmology)
    is_custom_2 = isinstance(cosmo2, CustomCosmology)

    if is_custom_1 and is_custom_2:
        return True

    elif not is_custom_1 and not is_custom_2:
        return cosmology_equal(cosmo1, cosmo2)

    return False


def get_default_cosmology() -> FLRW:
    return Planck15


class Scales:
    def __init__(
        self,
        scale_min: ArrayLike,
        scale_max: ArrayLike,
        *,
        unit: Unit = Unit.kpc,
        cosmology: TypeCosmology | None = None,
    ) -> None:
        self.unit = Unit(unit)
        self.cosmology = cosmology or get_default_cosmology()

        scale_min: NDArray = np.atleast_1d(scale_min)
        scale_max: NDArray = np.atleast_1d(scale_max)

        if scale_min.ndim != scale_max.ndim and scale_min.ndim != 1:
            raise ValueError(  # TODO: ConfigError
                "'scale_min/max' must be scalars or one-dimensional arrays"
            )
        if len(scale_min) != len(scale_max):
            # TODO: ConfigError
            raise ValueError("number of elements in 'scale_min/max' does not match")
        if np.any((scale_max - scale_min) <= 0.0):
            raise ValueError(  # TODO: ConfigError
                "'scale_min' must be smaller than corresponding 'scale_max'"
            )

        self.scale_min = scale_min
        self.scale_max = scale_max

    def get_angle(self, redshift: ArrayLike) -> tuple[NDArray, NDArray]:
        if self.unit in (Unit.deg, Unit.arcmin, Unit.arcsec):
            deg_min = self.scale_min
            deg_max = self.scale_max
            if self.unit == Unit.arcmin:
                deg_min = deg_min / 60.0
                deg_max = deg_min / 60.0
            elif self.unit == Unit.arcsec:
                deg_min = deg_min / 3600.0
                deg_max = deg_min / 3600.0
            return np.deg2rad(deg_min), np.deg2rad(deg_max)

        elif self.unit in (Unit.kpc, Unit.Mpc):
            mpc_min = self.scale_min
            mpc_max = self.scale_max
            if self.unit == Unit.kpc:
                mpc_min = mpc_min / 1000.0
                mpc_max = mpc_max / 1000.0

            rad_per_mpc = 1.0 / self.cosmology.angular_diameter_distance(redshift)
            if isinstance(rad_per_mpc, Quantity):
                rad_per_mpc = rad_per_mpc.value

            return rad_per_mpc * mpc_min, rad_per_mpc * mpc_max

        elif self.unit in (Unit.kpc_h, Unit.Mpc_h):
            mpc_min = self.scale_min
            mpc_max = self.scale_max
            if self.unit == Unit.kpc_h:
                mpc_min = mpc_min / 1000.0
                mpc_max = mpc_max / 1000.0

            comoving_dist = self.cosmology.comoving_distance(redshift)
            return mpc_min / comoving_dist, mpc_max / comoving_dist

        return self.scale_min, self.scale_max  # Unit.rad


def separation_physical_to_angle(
    separation_kpc: ArrayLike,
    redshift: ArrayLike,
    *,
    cosmology: TypeCosmology | None = None,
) -> NDArray:
    """
    Convert physical transverse angular diameter distance to angle at the given
    redshifts.

    Args:
        separation_kpc:
            Single or array of transverse angular diameter distances in kpc.
        redshift:
            Single or array of redshifts.

    Keyword Args:
        cosmology:
            Astropy or ``CustomCosmology`` for distance conversion. Defaults to
            default cosmology.

    Returns:
        An outer product of angular separations for pairs of separation and
        redshift. Along the first axis are fixed values of separation, along the
        second axis fixed values of redshift. The result has therefore a shape
        (``len(separation_kpc)``, ``len(redshift)``).
    """
    cosmology = cosmology or get_default_cosmology()

    rad_per_mpc = 1.0 / cosmology.angular_diameter_distance(redshift)
    if isinstance(rad_per_mpc, Quantity):
        rad_per_mpc = rad_per_mpc.value

    separation_mpc = np.asarray(separation_kpc) / 1000.0

    return np.outer(rad_per_mpc, separation_mpc)
