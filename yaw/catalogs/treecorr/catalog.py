from __future__ import annotations

import itertools
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING
try:  # pragma: no cover
    from typing import TypeAlias
except ImportError:  # pragma: no cover
    from typing_extensions import TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from treecorr import Catalog, NNCorrelation

from yaw.catalogs import BaseCatalog
from yaw.config import Configuration
from yaw.core.coordinates import Coordinate, Coord3D, CoordSky, DistSky
from yaw.core.logging import TimedLog
from yaw.correlation.paircounts import (
    PairCountResult, PatchedCount, PatchedTotal, pack_results)
from yaw.redshifts import HistogramData

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame, Interval
    from yaw.catalogs import PatchLinkage
    from yaw.config import ResamplingConfig


TypeNNResult: TypeAlias = dict[tuple[int, int], NNCorrelation]


def _iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "left"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = pd.IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in enumerate(intervals, 1):
        yield interval, (bin_ids == i)


def take_subset(
    cat: TreecorrCatalog,
    items: NDArray[np.bool_] | NDArray[np.int_] | slice
) -> TreecorrCatalog:
    kwargs = dict(
        ra=cat.ra[items], ra_units="radian",
        dec=cat.dec[items], dec_units="radian",
        patch=cat.patch[items])
    if cat.has_redshifts():
        kwargs["r"] = cat.redshifts[items]
    if cat.has_weights():
        kwargs["w"] = cat.weights[items]
    return TreecorrCatalog.from_treecorr(Catalog(**kwargs))


class TreecorrCatalog(BaseCatalog):

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: BaseCatalog | Coordinate | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None,
        progress: bool = False
    ) -> None:
        # construct the underlying TreeCorr catalogue
        kwargs = dict()
        if cache_directory is not None:
            kwargs["save_patch_dir"] = cache_directory
            if not os.path.exists(cache_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{cache_directory}'")
            self.logger.info(f"using cache directory '{cache_directory}'")

        if n_patches is not None:
            kwargs["npatch"] = n_patches
            log_msg = f"creating {n_patches} patches"
        elif patch_name is not None:
            kwargs["patch"] = data[patch_name]
            log_msg = f"splitting data into predefined patches"
        elif isinstance(patch_centers, BaseCatalog):
            kwargs["patch_centers"] = patch_centers.centers.to_3d().values
            n_patches = patch_centers.n_patches
            log_msg = f"applying {n_patches} patches from external data"
        elif isinstance(patch_centers, Coordinate):
            centers = patch_centers.to_3d()
            kwargs["patch_centers"] = centers.values
            n_patches = len(centers)
            log_msg = f"applying {n_patches} patches from external data"
        else:
            raise ValueError(
                "either of 'patch_name', 'patch_centers', or 'n_patches' "
                "must be provided")

        with TimedLog(self.logger.info, log_msg):
            self._catalog = Catalog(
                ra=data[ra_name], ra_units="degrees",
                dec=data[dec_name], dec_units="degrees",
                r=None if redshift_name is None else data[redshift_name],
                w=None if weight_name is None else data[weight_name],
                **kwargs)
            self._make_patches()

        if cache_directory is not None:
            self.unload()

    @classmethod
    def from_cache(
        cls,
        cache_directory: str,
        progress: bool = False
    ) -> TreecorrCatalog:
        # super().from_cache(cache_directory)
        #self._make_patches()
        raise NotImplementedError

    @classmethod
    def from_treecorr(cls, cat: Catalog) -> TreecorrCatalog:
        new = cls.__new__(cls)
        new._catalog = cat
        new._make_patches()
        return new

    def _make_patches(self) -> None:
        c = self._catalog
        if c._patches is None:
            low_mem = (not self.is_loaded()) and (c.save_patch_dir is not None)
            c.get_patches(low_mem=low_mem)

    def to_treecorr(self) -> Catalog:
        return self._catalog

    def __len__(self) -> int:
        return self._catalog.ntot

    def __getitem__(self, item: int) -> Catalog:
        return self._catalog._patches[item]

    @property
    def ids(self) -> list[int]:
        return list(range(self.n_patches))

    @property
    def n_patches(self) -> int:
        return self._catalog.npatch

    def __iter__(self) -> Iterator[Catalog]:
        for patch in self._catalog._patches:
            yield patch

    def is_loaded(self) -> bool:
        return self._catalog.loaded

    def load(self) -> None:
        super().load()
        self._catalog.load()

    def unload(self) -> None:
        super().unload()
        self._catalog.unload()

    def has_redshifts(self) -> bool:
        return self.redshifts is not None

    def has_weights(self) -> bool:
        return self.weights is not None

    @property
    def ra(self) -> NDArray[np.float_]:
        return self._catalog.ra

    @property
    def dec(self) -> NDArray[np.float_]:
        return self._catalog.dec

    @property
    def redshifts(self) -> NDArray[np.float_] | None:
        return self._catalog.r

    @property
    def weights(self) -> NDArray[np.float_]:
        return self._catalog.w

    @property
    def patch(self) -> NDArray[np.int_]:
        return self._catalog.patch

    def get_min_redshift(self) -> float:
        if not hasattr(self, "_zmin"):
            if self.has_redshifts():
                self._zmin = self.redshifts.min()
            else:
                self._zmin = None
        return self._zmin

    def get_max_redshift(self) -> float:
        if not hasattr(self, "_zmax"):
            if self.has_redshifts():
                self._zmax = self.redshifts.max()
            else:
                self._zmax = None
        return self._zmax

    @property
    def total(self) -> float:
        return self._catalog.sumw

    def get_totals(self) -> NDArray[np.float_]:
        return np.array([patch.sumw for patch in iter(self)])

    @property
    def centers(self) -> CoordSky:
        centers = Coord3D.from_array(self._catalog.get_patch_centers())
        return centers.to_sky()

    @property
    def radii(self) -> DistSky:
        radii = []
        cls = self.__class__
        for cat in iter(self):
            # build a new TreecorrCatalog without any postprocessing
            patch = cls.__new__(cls)
            patch._catalog = cat
            # compute the angular radius from the maximum separation in 3D
            position = patch.pos.to_3d()
            radius = patch.centers.to_3d().distance(position).max()
            radii.append(radius.to_sky())
        return DistSky.from_dists(radii)

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, TreecorrCatalog]]:
        if not allow_no_redshift and not self.has_redshifts():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in pd.IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for interval, bin_mask in _iter_bin_masks(self.redshifts, z_bins):
                yield interval, take_subset(self, bin_mask)

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: TreecorrCatalog = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False
    ) -> PairCountResult | dict[str, PairCountResult]:
        super().correlate(config, binned, other, linkage)

        auto = other is None
        if not auto and not isinstance(other, TreecorrCatalog):
            raise TypeError
        nncorr_config = dict(
            sep_units="radian",
            metric="Arc",
            nbins=(
                1 if config.scales.rweight is None else config.scales.rbin_num),
            bin_slop=config.backend.rbin_slop,
            num_threads=config.backend.get_threads())

        # bin the catalogues if necessary
        cats1 = self.iter_bins(config.binning.zbins)
        if auto:
            cats2 = itertools.repeat((None, None))
        else:
            if binned:
                cats2 = other.iter_bins(config.binning.zbins)
            else:
                cats2 = itertools.repeat((None, other))

        # allocate output data containers
        binning = pd.IntervalIndex.from_breaks(config.binning.zbins)
        n_bins = len(binning)
        n_patches = self.n_patches
        totals1 = np.zeros((n_patches, n_bins))
        totals2 = np.zeros((n_patches, n_bins))
        count_dict = {
            str(scale): PatchedCount.zeros(binning, n_patches, auto=auto)
            for scale in config.scales}

        # iterate the bins and compute the correlation
        self.logger.debug(
            f"running treecorr on {config.backend.get_threads()} threads")
        for i, ((intv, bincat1), (_, bincat2)) in enumerate(zip(cats1, cats2)):
            angles = [
                scale.to_radian(intv.mid, config.cosmology)
                for scale in config.scales]
            # extract the total number of objects per patch
            totals1[:, i] = bincat1.get_totals()
            if bincat2 is None:
                totals2[:, i] = totals1[:, i]
            else:
                totals2[:, i] = bincat2.get_totals()

            for scale, (ang_min, ang_max) in zip(config.scales, angles):
                # run the correlation measurement
                correlation = NNCorrelation(
                    min_sep=ang_min, max_sep=ang_max, **nncorr_config)
                correlation.process(
                    bincat1.to_treecorr(),
                    None if bincat2 is None else bincat2.to_treecorr())

                # extract the pair counts
                scale_counts = count_dict[str(scale)]
                result: TypeNNResult = correlation.results
                for (pid1, pid2), corr_result in result.items():
                    scale_counts.counts[pid1, pid2, i] = corr_result.weight

        total = PatchedTotal(  # not scale-dependent
            binning=binning,
            totals1=totals1,
            totals2=totals2,
            auto=auto)
        return pack_results(count_dict, total)

    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False
    ) -> HistogramData:
        super().true_redshifts(config)
        if not self.has_redshifts():
            raise ValueError("catalog has no redshifts")
        # compute the reshift histogram in each patch
        hist_counts = []
        for patch in iter(self):
            counts, bins = np.histogram(
                patch.redshifts, config.binning.zbins, weights=patch.weights)
            hist_counts.append(counts)

        # construct the output data samples
        binning = pd.IntervalIndex.from_breaks(config.binning.zbins)
        patch_idx = sampling_config.get_samples(self.n_patches)
        nz_data = hist_counts.sum(axis=0)
        nz_samp = np.sum(hist_counts[patch_idx], axis=1)
        return HistogramData(
            binning=binning,
            data=nz_data,
            samples=nz_samp,
            method=sampling_config.method)
