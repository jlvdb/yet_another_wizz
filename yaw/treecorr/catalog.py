from __future__ import annotations

import itertools
import logging
import os
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
import pandas as pd
from treecorr import Catalog as TreeCorrCatalog, NNCorrelation

from yaw.core.catalog import CatalogBase
from yaw.core.config import Configuration
from yaw.core.coordinates import (
    distance_sphere2sky, position_sky2sphere, position_sphere2sky)
from yaw.core.cosmology import r_kpc_to_angle
from yaw.core.redshifts import NzTrue
from yaw.core.resampling import PairCountResult
from yaw.core.utils import TypeScaleKey

from yaw.logger import TimedLog

if TYPE_CHECKING:
    from pandas import DataFrame, Interval
    from yaw.core.catalog import PatchLinkage


logger = logging.getLogger(__name__)


def _iter_bin_masks(
    data: NDArray,
    bins: NDArray,
    closed: str = "left"
) -> Iterator[tuple[Interval, NDArray[np.bool_]]]:
    if closed not in ("left", "right"):
        raise ValueError("'closed' must be either of 'left', 'right'")
    intervals = pd.IntervalIndex.from_breaks(bins, closed=closed)
    bin_ids = np.digitize(data, bins, right=(closed=="right"))
    for i, interval in zip(range(1, len(bins)), intervals):
        yield interval, bin_ids==i


class Catalog(CatalogBase):

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: Catalog | NDArray[np.float_] | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None
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
        # TODO: different convention of patch centers in TC and SC
        elif isinstance(patch_centers, Catalog):
            kwargs["patch_centers"] = patch_centers._catalog.get_patch_centers()
            n_patches = patch_centers.n_patches()
            log_msg = f"applying {n_patches} patches from external data"
        elif patch_centers is not None:
            self.logger.warn(
                "treecorr and scipy versions of the Catalog class have "
                "different representations of patch centers internally")
            kwargs["patch_centers"] = patch_centers
            n_patches = len(patch_centers)
            log_msg = f"applying {n_patches} patches from external data"
        else:
            raise ValueError(
                "either of 'patch_name', 'patch_centers', or 'n_patches' "
                "must be provided")

        with TimedLog(self.logger.info, log_msg):
            self._catalog = TreeCorrCatalog(
                ra=data[ra_name], ra_units="degrees",
                dec=data[dec_name], dec_units="degrees",
                r=None if redshift_name is None else data[redshift_name],
                w=None if weight_name is None else data[weight_name],
                **kwargs)

        if cache_directory is not None:
            self.unload()

    @classmethod
    def from_treecorr(cls, cat: TreeCorrCatalog) -> Catalog:
        new = cls.__new__(cls)
        new._catalog = cat
        return new

    def to_treecorr(self) -> TreeCorrCatalog:
        return self._catalog

    def __len__(self) -> int:
        return self._catalog.ntot

    def _make_patches(self) -> None:
        c = self._catalog
        low_mem = (not self.is_loaded()) and (c.save_patch_dir is not None)
        c.get_patches(low_mem=low_mem)

    def __getitem__(self, item: int) -> Catalog:
        self._make_patches()
        return self._catalog._patches[item]

    @property
    def ids(self) -> list[int]:
        return list(range(self.n_patches()))

    def n_patches(self) -> int:
        return self._catalog.npatch

    def __iter__(self) -> Iterator[TreeCorrCatalog]:
        self._make_patches()
        for patch in self._catalog._patches:
            yield self.__class__.from_treecorr(patch)

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
    def centers(self) -> NDArray[np.float_]:
        # TODO: figure out why double transform is necessary
        centers = position_sky2sphere(
            position_sphere2sky(self._catalog.get_patch_centers()))
        return centers

    @property
    def radii(self) -> NDArray[np.float_]:
        radii = []
        for patch, center in zip(iter(self), self.centers):
            ra_dec = np.transpose([patch.ra, patch.dec])
            xyz = position_sky2sphere(ra_dec)
            radius_xyz = np.sqrt(np.sum((xyz - center)**2, axis=1)).max()
            radii.append(distance_sphere2sky(radius_xyz))
        return np.array(radii)

    def bin_iter(
        self,
        z_bins: NDArray[np.float_],
    ) -> Iterator[tuple[Interval, Catalog]]:
        if not self.has_redshifts():
            raise ValueError("no redshifts for iteration provided")
        for interval, bin_mask in _iter_bin_masks(self.redshifts, z_bins):
            new = self._catalog.copy()
            new.select(bin_mask)
            yield interval, self.__class__.from_treecorr(new)

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: Catalog = None,
        linkage: PatchLinkage | None = None
    ) -> PairCountResult | dict[TypeScaleKey, PairCountResult]:
        super().correlate(config, binned, other, linkage)

        auto = other is None
        if not auto and not isinstance(other, Catalog):
            raise TypeError
        nncorr_config = dict(
            sep_units="radian",
            metric="Arc",
            nbins=(
                1 if config.scales.rweight is None else config.scales.rbin_num),
            bin_slop=config.backend.rbin_slop,
            num_threads=config.backend.thread_num)

        # bin the catalogues if necessary
        cats1 = self.bin_iter(config.binning.zbins)
        if auto:
            cats2 = itertools.repeat((None, None))
        else:
            if binned:
                cats2 = other.bin_iter(config.binning.zbins)
            else:
                cats2 = itertools.repeat((None, other))

        # iterate the bins and compute the correlation
        self.logger.debug(
            f"running treecorr on {config.backend.thread_num} threads")
        result = {
            scale_key: [] for scale_key in config.scales.dict_keys()}
        for (intv, bin_cat1), (_, bin_cat2) in zip(cats1, cats2):
            scales = r_kpc_to_angle(
                config.scales.scales, intv.mid, config.cosmology)
            for scale_key, (ang_min, ang_max) in zip(
                    config.scales.dict_keys(), scales):
                correlation = NNCorrelation(
                    min_sep=ang_min, max_sep=ang_max, **nncorr_config)
                correlation.process(
                    bin_cat1.to_treecorr(),
                    None if bin_cat2 is None else bin_cat2.to_treecorr())
                result[scale_key].append(
                    PairCountResult.from_nncorrelation(intv, correlation))
        if len(result) == 1:
            result = PairCountResult.from_bins(tuple(result.values())[0])
        else:
            for scale_key, binned_result in result.items():
                result[scale_key] = PairCountResult.from_bins(binned_result)
        return result

    def true_redshifts(self, config: Configuration) -> NzTrue:
        super().true_redshifts(config)

        if not self.has_redshifts():
            raise ValueError("catalog has no redshifts")
        # compute the reshift histogram in each patch
        hist_counts = []
        for patch in iter(self):
            print(patch)
            counts, bins = np.histogram(
                patch.redshifts, config.binning.zbins, weights=patch.weights)
            hist_counts.append(counts)
        return NzTrue(np.array(hist_counts), bins)
