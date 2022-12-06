from __future__ import annotations

import os
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.catalog import PatchCollection
from yet_another_wizz.correlation import CorrelationFunction
from yet_another_wizz.cosmology import TypeCosmology, get_default_cosmology
from yet_another_wizz.redshifts import BinFactory, NzTrue
from yet_another_wizz.resampling import PairCountResult
from yet_another_wizz.utils import Timed, TypeScaleKey


class YetAnotherWizzBase(ABC):

    def __init__(
        self,
        *,
        # data samples, TODO: decide which arguments are optional
        reference: PatchCollection,
        ref_rand: PatchCollection,
        unknown: PatchCollection,
        unk_rand: PatchCollection,
        # measurement scales
        rmin_kpc: ArrayLike,
        rmax_kpc: ArrayLike,
        dist_weight_scale: float | None = None,  # TODO: test if not None
        dist_weight_res: int = 50,
        # redshift binning
        zmin: float | None = None,
        zmax: float | None = None,
        n_zbins: int | None = None,
        zbin_method: str = "linear",
        cosmology: TypeCosmology | None = None,
        zbins: NDArray | None = None,
        # others
        num_threads: int | None = None
    ) -> None:
        # set data
        self.reference = reference
        self.ref_rand = ref_rand
        self.unknown = unknown
        self.unk_rand = unk_rand
        self._require_redshifts()
        self._check_patches()
        # configure scales
        self.rmin = rmin_kpc
        self.rmax = rmax_kpc
        self.dist_weight_scale = dist_weight_scale
        self.dist_weight_res = dist_weight_res
        if cosmology is None:
            cosmology = get_default_cosmology()
        self.cosmology = cosmology
        # configure redshift binning
        if zbins is not None:
            BinFactory.check(zbins)
            self.binning = zbins
        elif n_zbins is not None:
            if zmin is None:
                zmin = reference.get_min_redshift()
            if zmax is None:
                zmax = reference.get_max_redshift()
            factory = BinFactory(zmin, zmax, n_zbins, cosmology)
            self.binning = factory.get(zbin_method)
        else:
            raise ValueError("either 'zbins' or 'n_zbins' must be provided")
        # others
        if num_threads is None:
            num_threads = os.cpu_count()
        self.num_threads = num_threads

    def _require_redshifts(self) -> None:
        if not self.reference.has_redshift():
            raise ValueError("'reference' has no redshifts")

    def _check_patches(self) -> None:
        n_patches = self.n_patches()
        # check with all other items
        for kind in ("reference", "ref_rand", "unknown"):
            if getattr(self, kind).n_patches() != n_patches:
                raise ValueError(
                    f"number of patches in '{kind}' does not match 'reference'")

    @property
    def scales(self) -> NDArray[np.float_]:
        # TODO: remove single scale limitation (final indexing)
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))#[0]

    def n_patches(self) -> int:
        return self.unk_rand.n_patches()

    def get_config(self) -> dict[str, int | float | bool | str | None]:
        raise NotImplementedError  # TODO

    @abstractmethod
    def _correlate(
        self,
        cat1: PatchCollection,
        bin1: bool,
        cat2: PatchCollection | None = None,
        bin2: bool | None = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        raise NotImplementedError

    def crosscorr(
        self,
        *,
        compute_rr: bool = False,
        progress: bool = False
    ) -> dict[TypeScaleKey, CorrelationFunction]:
        if progress: print("crosscorrelating")
        with Timed("counting data-data pairs", progress):
            DD = self._correlate(self.reference, True, self.unknown, False)
        with Timed("counting data-random pairs", progress):
            DR = self._correlate(self.reference, True, self.unk_rand, False)
        if compute_rr:
            with Timed("counting random-data pairs", progress):
                RD = self._correlate(self.ref_rand, True, self.unknown, False)
            with Timed("counting random-random pairs", progress):
                RR = self._correlate(self.ref_rand, True, self.unk_rand, False)
        else:
            RD = None
            RR = None
        corrfuncs = {}
        for scale_key in DD:
            kwargs = {"dd": DD[scale_key], "dr": DR[scale_key]}
            if compute_rr:
                kwargs.update({"rd": RD[scale_key], "rr": RR[scale_key]})
            corrfuncs[scale_key] = CorrelationFunction(**kwargs)
        return corrfuncs

    def autocorr(
        self,
        which: str,
        *,
        compute_rr: bool = True,
        progress: bool = False
    ) -> CorrelationFunction:
        if which == "reference":
            data = self.reference
            random = self.ref_rand
        elif which == "unknown":
            data = self.unknown
            random = self.unk_rand
        else:
            raise ValueError("'which' must be either 'reference' or 'unknown'")
        # run correlation
        if progress: print(f"autocorrelating {which}")
        with Timed("counting data-data pairs", progress):
            DD = self._correlate(data, True)
        with Timed("counting data-random pairs", progress):
            DR = self._correlate(data, True, random, True)
        if compute_rr:
            with Timed("counting random-random pairs", progress):
                RR = self._correlate(random, True)
        else:
            RR = None
        corrfuncs = {}
        for scale_key in DD:
            kwargs = {"dd": DD[scale_key], "dr": DR[scale_key]}
            if compute_rr:
                kwargs["rr"] = RR[scale_key]
            corrfuncs[scale_key] = CorrelationFunction(**kwargs)
        return corrfuncs

    def true_redshifts(self, progress: bool = False) -> NzTrue:
        if self.unknown.has_redshift() is None:
            raise ValueError("'unknown' has not redshifts provided")
        # compute the reshift histogram in each patch
        with Timed("processing true redshifts", progress):
            hist_counts = []
            for patch in self.unknown.iter_loaded():
                counts, bins = np.histogram(
                    patch.redshift, self.binning, weights=patch.weights)
                hist_counts.append(counts)
        return NzTrue(np.array(hist_counts), bins)
