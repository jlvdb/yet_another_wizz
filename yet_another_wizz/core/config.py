from __future__ import annotations

import os
from typing import Any, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.core.cosmology import TypeCosmology, get_default_cosmology


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
            raise ValueError("redshift bins are not monotonicaly increasing")

    def get(self, method: str) -> NDArray[np.float_]:
        try:
            return getattr(self, method)()
        except AttributeError:
            raise ValueError(f"invalid binning method '{method}'")


class Configuration:

    def __init__(
        self,
        *,
        # measurement scales
        rmin: ArrayLike,  # kpc
        rmax: ArrayLike,  # kpc
        weight_scale: float | None = None,
        resolution: int = 50,
        rbin_slop: float | None = None,  # treecorr only
        cosmology: TypeCosmology | None = None,
        # redshift binning
        zmin: float | None = None,
        zmax: float | None = None,
        nbins: int | None = None,
        method: str = "linear",
        zbins: NDArray[np.float_] | None = None,
        # others
        num_threads: int | None = None,
        crosspatch: bool = True  # scipy only
    ) -> None:
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.set_cosmology(cosmology)

        self.weight_scale = weight_scale
        self.resolution = resolution
        self.set_scales(rmin, rmax, rbin_slop)

        if zbins is None:
            if zmin is None or zmax is None or nbins is None:
                raise ValueError(
                    "either 'zbins' or 'zmin', 'zmax', 'nbins' are required")
            self.generate_redshift_bins(zmin, zmax, nbins, method)
        else:
            self.set_redshift_bins(zbins)
            self.method = "manual"

        if num_threads is None:
            num_threads = os.cpu_count()
        self.num_threads = num_threads
        self.crosspatch = crosspatch

    @property
    def scales(self) -> NDArray[np.float_]:
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))

    def __repr__(self) -> str:
        name = self.__class__.__name__
        try:
            cosmology = self.cosmology.name
        except AttributeError:
            cosmology = repr(self.cosmology)
        cosm = f"cosmology={cosmology}"
        scales = [f"{int(s[0])}-{int(s[1])}" for s in self.scales]
        scales = f"scales=[{', '.join(scales)}], "
        if self.weight_scale is not None:
            scales += f"weight_scale={self.weight_scale}, "
            scales += f"resolution={self.resolution}, "
        scales += f"rbin_slop={self.rbin_slop}"
        zbins = f"zmin={self.zmin}, zmax={self.zmax}, nbins={self.nbins}, "
        zbins += f"method={self.method}"
        extra = f"num_theads={self.num_threads}, crosspatch={self.crosspatch}"
        return f"{name}:\n    {cosm}\n    {scales}\n    {zbins}\n    {extra}"

    def set_cosmology(self, cosmology: TypeCosmology) -> None:
        if not isinstance(cosmology, get_args(TypeCosmology)):
            which = ", ".join(get_args(TypeCosmology))
            raise TypeError(f"'cosmology' must be instance of: {which}")
        self.cosmology = cosmology

    def set_scales(
        self,
        rmin: ArrayLike,
        rmax: ArrayLike,
        rbin_slop: float | None = None
    ) -> None:
        rmin = np.atleast_1d(rmin)
        rmax = np.atleast_1d(rmax)
        if len(rmin) != len(rmax):
            raise ValueError(
                "number of elements in 'rmin' and 'rmax' do not match")
        for i, (_rmin, _rmax) in enumerate(zip(rmin, rmax)):
            if _rmin >= _rmax:
                raise ValueError(f"scale at index {i} violates 'rmin' < 'rmax'")

        if rbin_slop is None:
            if self.weight_scale is None:
                rbin_slop = 0.01
            else:
                rbin_slop = 0.1

        self.rmin = rmin
        self.rmax = rmax
        self.rbin_slop = rbin_slop

    def generate_redshift_bins(
        self,
        zmin: float,
        zmax: float,
        nbins: int,
        method: str = "linear"
    ) -> None:
        factory = BinFactory(zmin, zmax, nbins, self.cosmology)
        zbins = factory.get(method)

        self.set_redshift_bins(zbins)
        self.method = method

    def set_redshift_bins(self, zbins: NDArray[np.float_]):
        zbins = np.asarray(zbins)
        if len(zbins) < 2:
            raise ValueError("'zbins' must have at least two edges")
        if np.any(np.diff(zbins) <= 0.0):
            raise ValueError("redshift bins are not monotonic")
        self.zbins = zbins

        self.zmin = zbins[0]
        self.zmax = zbins[-1]
        self.nbins = len(zbins) - 1

    def get_version(self):
        raise NotImplementedError

    def as_dict(self) -> dict[str, Any]:
        values = {}
        for attr in dir(self):
            if attr.startswith("_"):
                continue
            attribute = getattr(self, attr)
            if not callable(attribute):
                values[attr] = attribute
        return values
