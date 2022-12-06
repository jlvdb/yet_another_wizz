from __future__ import annotations

import itertools

import numpy as np
import treecorr
from numpy.typing import ArrayLike, NDArray

from yet_another_wizz.cosmology import TypeCosmology, r_kpc_to_angle
from yet_another_wizz.redshifts import NzTrue
from yet_another_wizz.treecorr.catalog import BinnedCatalog
from yet_another_wizz.treecorr.resampling import PairCountResultTC
from yet_another_wizz.utils import Timed, TypeScaleKey, scales_to_keys
from yet_another_wizz.yaw import YetAnotherWizzBase


class YetAnotherWizzTC(YetAnotherWizzBase):

    def __init__(
        self,
        *,
        # data samples
        reference: BinnedCatalog,
        ref_rand: BinnedCatalog,
        unknown: BinnedCatalog,
        unk_rand: BinnedCatalog,
        # measurement scales, TODO: implement multiple scales!
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
        rbin_slop: float | None = None
    ) -> None:
        super().__init__(
            reference=reference, ref_rand=ref_rand, unknown=unknown,
            unk_rand=unk_rand, rmin_kpc=rmin_kpc, rmax_kpc=rmax_kpc,
            dist_weight_scale=dist_weight_scale,
            dist_weight_res=dist_weight_res, zmin=zmin, zmax=zmax,
            n_zbins=n_zbins, zbin_method=zbin_method, cosmology=cosmology,
            zbins=zbins, num_threads=num_threads)
        # set treecorr configuration
        if rbin_slop is None:
            if dist_weight_scale is None:
                rbin_slop = 0.01
            else:
                rbin_slop = 0.1
        self.correlator_config = dict(
            sep_units="degrees",
            metric="Arc",
            nbins=(1 if dist_weight_scale is None else dist_weight_res),
            bin_slop=rbin_slop,
            num_threads=num_threads)

    def _require_redshifts(self) -> None:
        if self.reference.redshift is None:
            raise ValueError("'reference' has no redshifts")

    def _correlate(
        self,
        cat1: BinnedCatalog,
        bin1: bool,
        cat2: BinnedCatalog | None = None,
        bin2: bool | None = None
    ) -> dict[TypeScaleKey, PairCountResultTC]:
        # bin the catalogues if necessary
        zbins = self.binning
        cats1 = cat1.bin_iter(zbins) if bin1 else itertools.repeat((None, cat1))
        cats2 = cat2.bin_iter(zbins) if bin2 else itertools.repeat((None, cat2))
        # iterate the bins and compute the correlation
        result = {scale_key: [] for scale_key in scales_to_keys(self.scales)}
        for (intv, bin_cat1), (_, bin_cat2) in zip(cats1, cats2):
            scales = r_kpc_to_angle(self.scales, intv.mid, self.cosmology)
            for scale_key, (ang_min, ang_max) in zip(
                    scales_to_keys(self.scales), scales):
                correlation = treecorr.NNCorrelation(
                    min_sep=ang_min, max_sep=ang_max, **self.correlator_config)
                correlation.process(bin_cat1, bin_cat2)
                result[scale_key].append(
                    PairCountResultTC.from_nncorrelation(intv, correlation))
        for scale_key, binned_result in result.items():
            result[scale_key] = PairCountResultTC.from_bins(binned_result)
        return result

    def true_redshifts(self, progress: bool = False) -> NzTrue:
        if self.unknown.redshift is None:
            raise ValueError("'unknown' has not redshifts provided")
        # compute the reshift histogram in each patch
        with Timed("processing true redshifts", progress):
            hist_counts = []
            for _, patch_cat in self.unknown.patch_iter():
                counts, bins = np.histogram(
                    patch_cat.redshift, self.binning, weights=patch_cat.w)
                hist_counts.append(counts)
        return NzTrue(np.array(hist_counts), bins)
