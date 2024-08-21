from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import numpy as np
from astropy.cosmology import FLRW, Planck15
from astropy.units import Quantity
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "CustomCosmology",
    "get_default_cosmology",
    "separation_physical_to_angle",
]

Tcosmology = Union[FLRW, "CustomCosmology"]


def get_default_cosmology() -> FLRW:
    return Planck15


class CustomCosmology(ABC):
    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        pass

    @abstractmethod
    def angular_diameter_distance(self, z: ArrayLike) -> ArrayLike:
        pass


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
    return np.outer(separation_mpc, rad_per_mpc)
