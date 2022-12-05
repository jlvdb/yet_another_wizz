from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.catalog import PatchCatalog, PatchCollection
from yet_another_wizz.correlation import CorrelationFunction
from yet_another_wizz.redshifts import BinFactory, NzTrue
from yet_another_wizz.resampling import PairCountResult
from yet_another_wizz.utils import (
    ArrayDict, ParallelHelper, Timed, TypeCosmology, TypePatchKey, TypeScaleKey,
    get_default_cosmology, r_kpc_to_angle)


def scales_to_keys(scales: NDArray[np.float_]) -> list[TypeScaleKey]:
    return [f"kpc{scale[0]:.0f}t{scale[1]:.0f}" for scale in scales]


def count_pairs_thread(
    patch1: PatchCatalog,
    patch2: PatchCatalog,
    scales: NDArray[np.float_],
    cosmology: TypeCosmology,
    z_bins: NDArray[np.float_],
    bin1: bool = True,
    bin2: bool = False,
    dist_weight_scale: float | None = None,
    weight_res: int = 50
) -> tuple[TypePatchKey, tuple[NDArray, NDArray], dict[TypeScaleKey, NDArray]]:
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
    # count pairs, iterate through the bins and count pairs between the trees
    counts = np.empty((len(scales), len(z_intervals)))
    totals1 = np.empty(len(z_intervals))
    totals2 = np.empty(len(z_intervals))
    for i, (intv, tree1, tree2) in enumerate(zip(z_intervals, trees1, trees2)):
        # if bin1 is False and bin2 is False, these will still give different
        # counts since the angle for scales is chaning
        counts[:, i] = tree1.count(
            tree2, scales=r_kpc_to_angle(scales, intv.mid, cosmology),
            dist_weight_scale=dist_weight_scale, weight_res=weight_res)
        totals1[i] = tree1.total
        totals2[i] = tree2.total
    counts = {key: count for key, count in zip(scales_to_keys(scales), counts)}
    return (patch1.id, patch2.id), (totals1, totals2), counts


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
        # generate the patch linkage (sufficiently close patches)
        with Timed("generating patch linkage        "):
            if disable_crosspatch:
                self._crosspatch = False
                max_ang = 0.0  # only relevenat for cross-patch
            else:
                self._crosspatch = True
                # estimate maximum query radius at low, but non-zero redshift
                z_ref = 0.1  # TODO: resonable? lower redshift => more overlap
                max_ang = r_kpc_to_angle(
                    self.scales, z_ref, self.cosmology).max()
            self._linkage = self.ref_rand.get_linkage(max_ang)
        # others
        if num_threads is None:
            num_threads = os.cpu_count()
        self.num_threads = num_threads

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

    def _correlate(
        self,
        collection1: PatchCollection,
        bin1: bool,
        collection2: PatchCollection | None = None,
        bin2: bool | None = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        auto = collection2 is None
        # prepare for measurement
        patch1_list, patch2_list = self._linkage.get_patches(
            collection1, collection2, self._crosspatch)
        n_jobs = len(patch1_list)

        # prepare job scheduling
        pool = ParallelHelper(
            function=count_pairs_thread,
            n_items=n_jobs, num_threads=min(n_jobs, self.num_threads))
        # patch1: PatchCatalog
        pool.add_iterable(patch1_list)
        # patch2: PatchCatalog
        pool.add_iterable(patch2_list)
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

        n_bins = len(self.binning) - 1
        n_patches = self.n_patches()
        # execute, unpack the data
        totals1 = np.zeros((n_patches, n_bins))
        totals2 = np.zeros((n_patches, n_bins))
        count_dict = {key: {} for key in scales_to_keys(self.scales)}
        for (id1, id2), (total1, total2), counts in pool.iter_result():
            # record total weight per bin, overwriting OK since identical
            totals1[id1] = total1
            totals2[id2] = total2
            # record counts at each scale
            for scale_key, count in counts.items():
                count_dict[scale_key][(id1, id2)] = count

        # get mask of all used cross-patch combinations, upper triangle if auto
        mask = self._linkage.get_mask(
            collection1, collection2, self._crosspatch)
        keys = [
            tuple(key) for key in np.indices((n_patches, n_patches))[:, mask].T]

        # compute patch-wise product of total of weights
        total_matrix = np.empty((n_patches, n_patches, n_bins))
        for i in range(n_bins):
            # get the patch totals for the current bin
            totals = np.multiply.outer(totals1[:, i], totals2[:, i])
            total_matrix[:, :, i] = totals
        # apply correction for autocorrelation, i. e. no double-counting
        if auto:
            total_matrix[np.diag_indices(n_patches)] *= 0.5
        # flatten to shape (n_patches*n_patches, n_bins), also if auto:
        # (id1, id2) not counted, i.e. dropped, if id1 > id2
        total = total_matrix[mask]
        del total_matrix

        # sort counts into similar data structure, pack result
        result = {}
        for scale_key, counts in count_dict.items():
            count_matrix = np.zeros((n_patches, n_patches, n_bins))
            for patch_key, count in counts.items():
                count_matrix[patch_key] = count
            # apply correction for autocorrelation, i. e. no double-counting
            if auto:
                count_matrix[np.diag_indices(n_patches)] *= 0.5
            count = count_matrix[mask]
            result[scale_key] = PairCountResult(
                n_patches,
                count=ArrayDict(keys, count),
                total=ArrayDict(keys, total),
                mask=mask,
                binning=pd.IntervalIndex.from_breaks(self.binning))
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
