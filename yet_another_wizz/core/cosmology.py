from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union
from typing_extensions import TypeAlias

import numpy as np
from astropy.cosmology import FLRW, Planck15
from numpy.typing import ArrayLike, NDArray


def get_default_cosmology() -> FLRW:
    return Planck15


class CustomCosmology(ABC):
    """
    Can be used to implement a custom cosmology outside of astropy.cosmology
    """

    @abstractmethod
    def to_format(self, format: str = "mapping") -> str:
        # TODO: really necessary?
        raise NotImplementedError

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def comoving_transverse_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError


TypeCosmology: TypeAlias = Union[FLRW, CustomCosmology]


def r_kpc_to_angle(
    r_kpc: NDArray[np.float_],
    z: float,
    cosmology: TypeCosmology
) -> tuple[float, float]:
    """from kpc to radian"""
    f_K = cosmology.comoving_transverse_distance(z)  # for 1 radian in Mpc
    return np.asarray(r_kpc) / 1000.0 * (1.0 + z) / f_K.value
