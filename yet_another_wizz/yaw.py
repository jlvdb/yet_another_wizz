from __future__ import annotations

import itertools
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.catalog import PatchCatalog, PatchCollection
from yet_another_wizz.correlation import CorrelationFunction
from yet_another_wizz.redshifts import BinFactory, NzTrue
from yet_another_wizz.resampling import PairCountData, PairCountResult
from yet_another_wizz.utils import (
    ParallelHelper, Timed, TypeCosmology, TypeScaleKey, TypeThreadResult,
    get_default_cosmology, r_kpc_to_angle)


def scales_to_keys(scales: NDArray[np.float_]) -> list[TypeScaleKey]:
    return [f"kpc{scale[0]:.0f}t{scale[1]:.0f}" for scale in scales]


def count_pairs_binned(
    auto: bool,
    patch1: PatchCatalog,
    patch2: PatchCatalog,
    scales: NDArray[np.float_],
    cosmology: TypeCosmology,
    z_bins: NDArray[np.float_],
    bin1: bool = True,
    bin2: bool = False,
    dist_weight_scale: float | None = None,
    weight_res: int = 50
) -> TypeThreadResult:
    z_intervals = pd.IntervalIndex.from_breaks(z_bins)
    # build trees
    patch1.load(use_threads=False)
    if bin1:
        trees1 = [patch.get_tree() for _, patch in patch1.iter_bins(z_bins)]
    else:
        trees1 = itertools.repeat(patch1.get_tree())
    patch2.load(use_threads=False)
    if bin2:
        trees2 = [patch.get_tree() for _, patch in patch2.iter_bins(z_bins)]
    else:
        trees2 = itertools.repeat(patch2.get_tree())
    # count pairs
    scale_keys = scales_to_keys(scales)
    results = {key: {} for key in scale_keys}
    # iterate through the bins and count pairs between the trees
    totals = []
    counts = []
    for intv, tree1, tree2 in zip(z_intervals, trees1, trees2):
        # if bin1 is False and bin2 is False, these will still give different
        # counts since the angle for scales is chaning
        count, total = tree1.count(
            tree2, scales=r_kpc_to_angle(scales, intv.mid, cosmology),
            dist_weight_scale=dist_weight_scale, weight_res=weight_res)
        if auto and patch1.id == patch2.id:
            # (i, j) pairs are counted twice for i == j, but once for i != j
            total *= 0.5
            count *= 0.5
        totals.append(total)
        counts.append(count)
    totals = np.transpose(totals)
    counts = np.transpose(counts)
    # package result and identify with their patch IDs
    for key, total, count in zip(scale_keys, totals, counts):
        results[key][(patch1.id, patch2.id)] = PairCountData(
            z_intervals, total=total, count=count)
    return results


class YetAnotherWizz:

    def __init__(
        self,
        *,
        # data samples
        reference: PatchCollection,
        ref_rand: PatchCollection,  # TODO: decide which arguments are optional
        unknown: PatchCollection,
        unk_rand: PatchCollection,
        # measurement scales
        rmin_kpc: ArrayLike = 0.1,
        rmax_kpc: ArrayLike = 1.0,
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
        disable_crosspatch: bool = False,
        num_threads: int | None = None
    ) -> None:
        # set data
        if reference.has_z() is None:
            raise ValueError("'reference' has no redshifts")
        self.reference = reference
        self.ref_rand = ref_rand
        self.unknown = unknown
        self.unk_rand = unk_rand
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
                zmin = reference.get_z_min()
            if zmax is None:
                zmax = reference.get_z_max()
            factory = BinFactory(zmin, zmax, n_zbins, cosmology)
            self.binning = factory.get(zbin_method)
        else:
            raise ValueError("either 'zbins' or 'n_zbins' must be provided")
        # generate patch linkage (patches sufficiently close for correlation)
        if disable_crosspatch:
            self._linkage = [(i, i) for i in range(self.n_patches())]
        else:
            # estimate the maximum query radius at low, but non-zero redshift
            z_ref = 0.1  # TODO: resonable? lower redshift => more overlap
            max_ang = r_kpc_to_angle(self.scales, z_ref, self.cosmology).max()
            # generate the patch linkage (sufficiently close patches)
            with Timed("generating patch linkage        "):
                self._linkage = self.ref_rand.generate_linkage(max_ang)
        # others
        if num_threads is None:
            num_threads = os.cpu_count()
        self.num_threads = num_threads

    def _check_patches(self) -> None:
        n_patches = len(self.reference)
        # check with all other items
        for kind in ("ref_rand", "unknown", "unk_rand"):
            if len(getattr(self, kind)) != n_patches:
                raise ValueError(
                    f"number of patches in '{kind}' does not match 'reference'")

    @property
    def scales(self) -> NDArray[np.float_]:
        # TODO: remove single scale limitation (final indexing)
        return np.atleast_2d(np.transpose([self.rmin, self.rmax]))#[0]

    def n_patches(self) -> int:
        return len(self.reference)

    def get_config(self) -> dict[str, int | float | bool | str | None]:
        raise NotImplementedError  # TODO

    def _correlate(
        self,
        collection1: PatchCollection,
        bin1: bool,
        collection2: PatchCollection | None = None,
        bin2: bool | None = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        auto = collection2 is None
        # prepare for autocorrelation
        if auto:
            collection2 = collection1
            # avoid double-counting pairs
            linkage = [(i, j) for i, j in self._linkage if j >= i]
        else:
            if collection2 is None:
                raise ValueError("no 'collection2' provided")
            linkage = self._linkage
        # prepare job scheduling
        pool = ParallelHelper(
            function=count_pairs_binned,
            n_items=len(linkage),
            num_threads=self.num_threads)
        # auto: bool
        pool.add_constant(auto)
        # patch1: PatchCatalog
        pool.add_iterable([collection1[id1] for id1, id2 in linkage])
        # patch2: PatchCatalog
        pool.add_iterable([collection2[id2] for id1, id2 in linkage])
        # scales: NDArray[np.float_]
        pool.add_constant(self.scales)
        # cosmology: TypeCosmology
        pool.add_constant(self.cosmology)
        # z_bins: NDArray[np.float_]
        pool.add_constant(self.binning)
        # bin1: bool
        pool.add_constant(bin1)
        # bin2: bool
        pool.add_constant(bin2 if bin2 is not None else bin1)
        # dist_weight_scale: float | None
        pool.add_constant(self.dist_weight_scale)
        # weight_res: int
        pool.add_constant(self.dist_weight_res)
        # execute
        result_list: list[TypeThreadResult] = pool.run()
        # reorder output data hierarchy
        result_scales: TypeThreadResult = {}
        for scale_dict in result_list:
            for scale_key, patch_dict in scale_dict.items():
                if scale_key not in result_scales:
                    result_scales[scale_key] = {}
                result_scales[scale_key].update(patch_dict)
        # fill up missing totals from skipped cross-patch measurements
        z_intervals = pd.IntervalIndex.from_breaks(self.binning)
        Nbins = len(z_intervals)
        zero_counts = np.zeros(Nbins)
        totals = np.multiply.outer(  # wasts memory, but very fast
            [patch.total for patch in collection1],
            [patch.total for patch in collection2])
        for i in range(0, len(collection1)):
            for j in range(i, len(collection2)):
                if (i, j) not in result_scales[scale_key]:  # use last scale key
                    # entry is missing at all scales
                    for patch_dict in result_scales.values():
                        dummy = PairCountData(
                            z_intervals, zero_counts,
                            np.full(Nbins, totals[i, j]))
                        patch_dict[(i, j)] = dummy
                        if not auto:  # symmetric counting
                            patch_dict[(j, i)] = dummy
        # pack the data
        result = {}
        for scale_key, patch_dict in result_scales.items():
            result[scale_key] = PairCountResult.from_patch_dict(
                self.binning, self.n_patches(), patch_dict)
        return result

    def crosscorr(
        self,
        *,
        compute_rr: bool = False,
        progress: bool = False
    ) -> dict[TypeScaleKey, CorrelationFunction]:
        if progress: print("crosscorrelating")
        with Timed("    counting data-data pairs    ", progress):
            DD = self._correlate(self.reference, True, self.unknown, False)
        with Timed("    counting data-random pairs  ", progress):
            DR = self._correlate(self.reference, True, self.unk_rand, False)
        if compute_rr:
            with Timed("    counting random-data pairs  ", progress):
                RD = self._correlate(self.ref_rand, True, self.unknown, False)
            with Timed("    counting random-random pairs", progress):
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
        with Timed("    counting data-data pairs    ", progress):
            DD = self._correlate(data, True)
        with Timed("    counting data-random pairs  ", progress):
            DR = self._correlate(data, True, random, True)
        if compute_rr:
            with Timed("    counting random-random pairs", progress):
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

    def true_redshifts(self) -> NzTrue:
        if self.unknown.has_z() is None:
            raise ValueError("'unknown' has not redshifts provided")
        # compute the reshift histogram in each patch
        hist_counts = []
        for patch in self.unknown:
            is_loaded = patch.is_loaded()
            patch.load()
            counts, bins = np.histogram(
                patch.z, self.binning, weights=patch.weights)
            hist_counts.append(counts)
            if not is_loaded:
                patch.unload()
        return NzTrue(np.array(hist_counts), bins)
