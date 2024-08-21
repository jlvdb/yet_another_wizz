from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, get_args

import numpy as np
from astropy.cosmology import FLRW, Planck15, cosmology_equal
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CustomCosmology",
    "cosmology_is_equal",
    "get_default_cosmology",
    "separation_physical_to_angle",
]

Tcosmology = Union[FLRW, "CustomCosmology"]


class CustomCosmology(ABC):
    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def angular_diameter_distance(self, z: ArrayLike) -> ArrayLike:
        pass


def cosmology_is_equal(cosmo1: Tcosmology, cosmo2: Tcosmology) -> bool:
    if not isinstance(cosmo1, get_args(Tcosmology)):
        raise TypeError("'cosmo1' is not a valid cosmology type")
    if not isinstance(cosmo2, get_args(Tcosmology)):
        raise TypeError("'cosmo2' is not a valid cosmology type")

    is_custom_1 = isinstance(cosmo1, CustomCosmology)
    is_custom_2 = isinstance(cosmo2, CustomCosmology)

    if is_custom_1 and is_custom_1:
        return cosmo1 == cosmo2

    elif not is_custom_1 and not is_custom_2:
        return cosmology_equal(cosmo1, cosmo2)

    return False


def get_default_cosmology() -> FLRW:
    return Planck15


def separation_physical_to_angle(
    separation_kpc: ArrayLike,
    redshift: ArrayLike,
    *,
    cosmology: Tcosmology | None = None,
) -> NDArray:
    cosmology = cosmology or get_default_cosmology()

    rad_per_mpc = 1.0 / cosmology.angular_diameter_distance(redshift)
    if isinstance(rad_per_mpc, Quantity):
        rad_per_mpc = rad_per_mpc.value

    separation_mpc = np.asarray(separation_kpc) / 1000.0

    return np.outer(rad_per_mpc, separation_mpc)
