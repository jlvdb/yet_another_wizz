from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union

import numpy as np
from astropy.cosmology import FLRW, Planck15, cosmology_equal
from astropy.units import Quantity

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
