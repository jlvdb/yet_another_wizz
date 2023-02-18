from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Union
try:
    from typing import TypeAlias
except ImportError:
    from typing_extensions import TypeAlias

import numpy as np
from astropy.cosmology import FLRW, Planck15

if TYPE_CHECKING:
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


class BinFactory:

    def __init__(
        self,
        zmin: float,
        zmax: float,
        nbins: int,
        cosmology: TypeCosmology | None = None
    ):
        if zmin >= zmax:
            raise ValueError("'zmin' >= 'zmax'")
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        self.zmin = zmin
        self.zmax = zmax
        self.nbins = nbins

    def linear(self) -> NDArray[np.float_]:
        return np.linspace(self.zmin, self.zmax, self.nbins + 1)

    def comoving(self) -> NDArray[np.float_]:
        cbinning = np.linspace(
            self.cosmology.comoving_distance(self.zmin).value,
            self.cosmology.comoving_distance(self.zmax).value,
            self.nbins + 1)
        # construct a spline mapping from comoving distance to redshift
        zarray = np.linspace(0, 10.0, 5000)
        carray = self.cosmology.comoving_distance(zarray).value
        return np.interp(cbinning, xp=carray, fp=zarray)  # redshift @ cbinning

    def logspace(self) -> NDArray[np.float_]:
        logbinning = np.linspace(
            np.log(1.0 + self.zmin), np.log(1.0 + self.zmax), self.nbins + 1)
        return np.exp(logbinning) - 1.0

    @staticmethod
    def check(zbins: NDArray[np.float_]) -> None:
        if np.any(np.diff(zbins) <= 0):
            raise ValueError("redshift bins must be monotonicaly increasing")

    def get(self, method: str) -> NDArray[np.float_]:
        try:
            return getattr(self, method)()
        except AttributeError as e:
            raise ValueError(f"invalid binning method '{method}'") from e
