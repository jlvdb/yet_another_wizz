from __future__ import annotations

import itertools
from collections.abc import Iterable, Iterator

import numpy as np
import treecorr
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval, IntervalIndex

from yet_another_wizz.cosmology import TypeCosmology, get_default_cosmology, r_kpc_to_angle
from yet_another_wizz.redshifts import NzTrue
from yet_another_wizz.resampling import ArrayDict
from yet_another_wizz.resampling import PairCountResult as _PairCountResult
from yet_another_wizz.utils import TypeScaleKey, scales_to_keys
from yet_another_wizz.yaw import YetAnotherWizzBase


def _iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "right"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in zip(range(1, len(bins)), intervals):
        yield interval, bin_ids==i


class BinnedCatalog(treecorr.Catalog):

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        patches: int | BinnedCatalog | str,
        ra: str,
        dec: str,
        redshift: str | None = None,
        w: str | None = None,
        **kwargs
    ) -> BinnedCatalog:
        if isinstance(patches, int):
            kwargs.update(dict(npatch=patches))
        elif isinstance(patches, str):
            kwargs.update(dict(patch=data[patches]))
        else:
            kwargs.update(dict(patch_centers=patches.patch_centers))
        r = None if redshift is None else data[redshift]
        w = None if w is None else data[w]
        new = cls(
            ra=data[ra], ra_units="degrees",
            dec=data[dec], dec_units="degrees",
            r=r, w=w, **kwargs)
        return new

    @classmethod
    def from_catalog(cls, cat: treecorr.Catalog) -> BinnedCatalog:
        new = cls.__new__(cls)
        new.__dict__ = cat.__dict__
        return new

    def __iter__(self) -> Iterator[BinnedCatalog]:
        for patch in self.get_patches(low_mem=True):
            yield patch

    def n_patches(self) -> int:
        return self.npatch

    @property
    def redshift(self) -> NDArray:
        return self.r

    def has_z(self) -> bool:
        return self.r is not None

    def get_z_min(self) -> float:
        try:
            return self.redshift.min()
        except AttributeError:
            return None

    def get_z_max(self) -> float:
        try:
            return self.redshift.max()
        except AttributeError:
            return None

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, treecorr.Catalog]]:
        if self.redshift is None:
            raise ValueError("no redshifts for iteration provided")
        for interval, bin_mask in _iter_bin_masks(self.redshift, z_bins):
            new = self.copy()
            new.select(bin_mask)
            yield interval, BinnedCatalog.from_catalog(new)

    def patch_iter(self) -> Iterator[tuple[int, treecorr.Catalog]]:
        patch_ids = sorted(set(self.patch))
        patches = self.get_patches()
        for patch_id, patch in zip(patch_ids, patches):
            yield patch_id, BinnedCatalog.from_catalog(patch)


class PairCountResult(_PairCountResult):

    @classmethod
    def from_nncorrelation(
        cls,
        interval: Interval,
        correlation: treecorr.NNCorrelation
    ) -> PairCountResult:
        # extract the (cross-patch) pair counts
        npatch = max(correlation.npatch1, correlation.npatch2)
        keys = []
        count = np.empty((len(correlation.results), 1))
        total = np.empty((len(correlation.results), 1))
        for i, (patches, result) in enumerate(correlation.results.items()):
            keys.append(patches)
            count[i] = result.weight
            total[i] = result.tot
        return cls(
            npatch=npatch,
            count=ArrayDict(keys, count),
            total=ArrayDict(keys, total),
            mask=correlation._ok,
            binning=IntervalIndex([interval]))

    @classmethod
    def from_bins(
        cls,
        zbins: Iterable[PairCountResult]
    ) -> PairCountResult:
        # check that the data is compatible
        if len(zbins) == 0:
            raise ValueError("'zbins' is empty")
        npatch = zbins[0].npatch
        mask = zbins[0].mask
        keys = tuple(zbins[0].keys())
        nbins = zbins[0].nbins
        for zbin in zbins[1:]:
            if zbin.npatch != npatch:
                raise ValueError("the patch numbers are inconsistent")
            if not np.array_equal(mask, zbin.mask):
                raise ValueError("pair masks are inconsistent")
            if tuple(zbin.keys()) != keys:
                raise ValueError("patches are inconsistent")
            if zbin.nbins != nbins:
                raise IndexError("number of bins is inconsistent")

        # check the ordering of the bins based on the provided intervals
        binning = IntervalIndex.from_tuples([
            zbin.binning.to_tuples()[0]  # contains just one entry
            for zbin in zbins])
        if not binning.is_non_overlapping_monotonic:
            raise ValueError(
                "the binning interval is overlapping or not monotonic")
        for this, following in zip(binning[:-1], binning[1:]):
            if this.right != following.left:
                raise ValueError(f"the binning interval is not contiguous")

        # merge the ArrayDicts
        count = ArrayDict(
            keys, np.column_stack([zbin.count.as_array() for zbin in zbins]))
        total = ArrayDict(
            keys, np.column_stack([zbin.total.as_array() for zbin in zbins]))
        return cls(
            npatch=npatch,
            count=count,
            total=total,
            mask=mask,
            binning=binning)


class YetAnotherWizz(YetAnotherWizzBase):

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
    ) -> dict[TypeScaleKey, PairCountResult]:
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
                    PairCountResult.from_nncorrelation(intv, correlation))
        for scale_key, binned_result in result.items():
            result[scale_key] = PairCountResult.from_bins(binned_result)
        return result

    def true_redshifts(self) -> NzTrue:
        if self.unknown.redshift is None:
            raise ValueError("'unknown' has not redshifts provided")
        # compute the reshift histogram in each patch
        hist_counts = []
        for _, patch_cat in self.unknown.patch_iter():
            counts, bins = np.histogram(
                patch_cat.redshift, self.binning, weights=patch_cat.w)
            hist_counts.append(counts)
        return NzTrue(np.array(hist_counts), bins)
