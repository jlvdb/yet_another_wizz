from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.catalog import PatchCatalog, PatchCollection
from yet_another_wizz.core import YetAnotherWizzBase
from yet_another_wizz.cosmology import TypeCosmology, r_kpc_to_angle
from yet_another_wizz.parallel import ParallelHelper
from yet_another_wizz.redshifts import NzTrue
from yet_another_wizz.resampling import ArrayDict, PairCountResult
from yet_another_wizz.utils import Timed, TypePatchKey, TypeScaleKey, scales_to_keys


def count_pairs_thread(
    patch1: PatchCatalog,
    patch2: PatchCatalog,
    scales: NDArray[np.float_],
    cosmology: TypeCosmology,
    z_bins: NDArray[np.float_],
    bin1: bool = True,
    bin2: bool = False,
    dist_weight_scale: float | None = None,
    dist_weight_res: int = 50
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
            dist_weight_scale=dist_weight_scale, weight_res=dist_weight_res)
        totals1[i] = tree1.total
        totals2[i] = tree2.total
    counts = {key: count for key, count in zip(scales_to_keys(scales), counts)}
    return (patch1.id, patch2.id), (totals1, totals2), counts


def histogram_thread(
    patch: PatchCatalog,
    z_bins: NDArray[np.float_]
) -> NDArray[np.float_]:
    is_loaded = patch.is_loaded()
    patch.load()
    counts, _ = np.histogram(patch.redshift, z_bins, weights=patch.weights)
    if not is_loaded:
        patch.unload()
    return counts


class YetAnotherWizz(YetAnotherWizzBase):

    def __init__(
        self,
        *,
        # data samples
        reference: PatchCollection,
        ref_rand: PatchCollection,  # TODO: decide which arguments are optional
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
        num_threads: int | None = None,
        # core method specific
        disable_crosspatch: bool = False
    ) -> None:
        super().__init__(
            reference=reference, ref_rand=ref_rand, unknown=unknown,
            unk_rand=unk_rand, rmin_kpc=rmin_kpc, rmax_kpc=rmax_kpc,
            dist_weight_scale=dist_weight_scale,
            dist_weight_res=dist_weight_res, zmin=zmin, zmax=zmax,
            n_zbins=n_zbins, zbin_method=zbin_method, cosmology=cosmology,
            zbins=zbins, num_threads=num_threads)
        # generate the patch linkage (sufficiently close patches)
        if disable_crosspatch:
            self._crosspatch = False
            max_ang = 0.0  # only relevenat for cross-patch
        else:
            self._crosspatch = True
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = 0.05  # TODO: resonable? lower redshift => more overlap
            max_ang = r_kpc_to_angle(
                self.scales, z_ref, self.cosmology).max()
        self._linkage = self.ref_rand.get_linkage(max_ang)

    def _correlate(
        self,
        cat1: PatchCollection,
        bin1: bool,
        cat2: PatchCollection | None = None,
        bin2: bool | None = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        auto = cat2 is None
        # prepare for measurement
        patch1_list, patch2_list = self._linkage.get_patches(
            cat1, cat2, self._crosspatch)
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
        mask = self._linkage.get_mask(cat1, cat2, self._crosspatch)
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

    def true_redshifts(self, progress: bool = False) -> NzTrue:
        if self.unknown.has_redshift() is None:
            raise ValueError("'unknown' has not redshifts provided")
        # compute the reshift histogram in each patch
        pool = ParallelHelper(
            function=histogram_thread,
            n_items=self.n_patches(),
            num_threads=min(self.n_patches(), self.num_threads))
        # patch: PatchCatalog
        pool.add_iterable(self.unknown)
        # NDArray[np.float_]
        pool.add_constant(self.binning)
        with Timed("processing true redshifts", progress):
            hist_counts = list(pool.iter_result())
        return NzTrue(np.array(hist_counts), self.binning)
