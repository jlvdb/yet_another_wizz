from __future__ import annotations

import multiprocessing
import logging
import os
from collections.abc import Iterator
from itertools import repeat
from typing import TYPE_CHECKING, Any

import astropandas as apd
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from yaw.catalogs import PatchLinkage
from yaw.catalogs import utils
from yaw.catalogs.patches import (
    PatchCatalog,
    assign_patches,
    create_patches,
    patch_id_from_path,
)
from yaw.config import Configuration, ResamplingConfig
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, DistSky
from yaw.core.logging import TimedLog
from yaw.core.utils import LimitTracker, job_progress_bar, long_num_format
from yaw.correlation.paircounts import NormalisedCounts
from yaw.redshifts import HistData

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame

    from yaw.catalogs import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.core.containers import PatchCorrelationData
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = ["Catalog"]


def _worker_correlate(
    args: tuple[PatchCatalog, PatchCatalog, Configuration, bool, bool]
) -> PatchCorrelationData:
    return utils.count_pairs_patches(*args)


def _worker_true_redshifts(
    args: tuple[PatchCatalog, NDArray[np.float_]]
) -> NDArray[np.float_]:
    return utils.count_histogram_patch(*args)


class Catalog:
    """TODO: See factory
    """

    _logger = logging.getLogger("yaw.catalog")

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: Catalog | Coordinate | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> None:
        """TODO: See factory"""
        if len(data) == 0:
            raise ValueError("data catalog is empty")
        # check if the columns exist
        renames = {ra_name: "ra", dec_name: "dec"}
        if redshift_name is not None:
            renames[redshift_name] = "redshift"
        if weight_name is not None:
            renames[weight_name] = "weights"
        for col_name, kind in renames.items():
            if col_name not in data:
                raise KeyError(f"column {kind}='{col_name}' not found in data")

        # check if patches should be written and unloaded from memory
        unload = cache_directory is not None
        if patch_name is not None:
            patch_mode = "dividing"
        else:
            if n_patches is not None:
                patch_mode = "creating"
            elif patch_centers is not None:
                patch_mode = "applying"
            else:
                raise ValueError(
                    "either of 'patch_name', 'patch_centers', or 'n_patches' "
                    "must be provided"
                )
        if unload:
            if not os.path.exists(cache_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{cache_directory}'"
                )
            self._logger.debug("using cache directory '%s'", cache_directory)

        # create new patches
        if patch_mode != "dividing":
            position = CoordSky.from_array(
                np.deg2rad(data[[ra_name, dec_name]].to_numpy())
            )
            if patch_mode == "creating":
                patch_centers, patch_ids = create_patches(
                    position=position, n_patches=n_patches
                )
                log_msg = "creating %i patches"
            else:
                if isinstance(patch_centers, Catalog):
                    patch_centers = patch_centers.centers.to_3d()
                patch_ids = assign_patches(centers=patch_centers, position=position)
                n_patches = len(patch_centers)
                log_msg = "applying %i patches from external data"
            patch_name = "patch"  # the default name
            data[patch_name] = patch_ids
            centers = {pid: pos for pid, pos in enumerate(patch_centers)}
        else:
            n_patches = len(data[patch_name].unique())
            log_msg = "dividing data into %i predefined patches"
            centers = dict()  # this can be empty
        self._logger.debug(log_msg, n_patches)

        # run groupby first to avoid any intermediate copies of full data
        n_obj_str = long_num_format(len(data))
        with TimedLog(self._logger.info, f"processed {n_obj_str} records"):
            limits = LimitTracker()
            patches: dict[int, PatchCatalog] = {}
            patch_iter = data.groupby(patch_name)
            if progress:
                patch_iter = job_progress_bar(patch_iter, total=n_patches)
            for patch_id, patch_data in patch_iter:
                if patch_id < 0:
                    raise ValueError("negative patch IDs are not supported")
                # drop extra columns
                patch_data = patch_data.drop(
                    columns=[col for col in patch_data.columns if col not in renames]
                )
                patch_data.rename(columns=renames, inplace=True)
                patch_data.reset_index(drop=True, inplace=True)
                # look up the center of the patch if given
                kwargs = dict(center=centers.get(patch_id))
                if unload:
                    # data will be written as feather file and loaded on demand
                    kwargs["cachefile"] = os.path.join(
                        cache_directory, f"patch_{patch_id:.0f}.feather"
                    )
                patch = PatchCatalog(int(patch_id), patch_data, **kwargs)
                limits.update(patch.redshifts)
                if unload:
                    patch.unload()
                patches[patch.id] = patch
            if progress:  # clean up if any patch was empty and skipped
                patch_iter.close()
            self._zmin, self._zmax = limits.get()
            self._patches = patches

        # also store the patch properties
        if unload:
            centers = self.centers.to_3d()
            property_df = pd.DataFrame(
                dict(
                    ids=self.ids,
                    x=centers.x,
                    y=centers.y,
                    z=centers.z,
                    r=self.radii.values,
                )
            )
            fpath = os.path.join(cache_directory, "properties.feather")
            property_df.to_feather(fpath)

    @classmethod
    def from_file(
        cls,
        filepath: str,
        patches: str | int | Catalog | Coordinate,
        ra: str,
        dec: str,
        *,
        redshift: str | None = None,
        weight: str | None = None,
        sparse: int | None = None,
        cache_directory: str | None = None,
        file_ext: str | None = None,
        progress: bool = False,
        **kwargs,
    ) -> Catalog:
        """TODO: See factory"""
        columns = [c for c in [ra, dec, redshift, weight] if c is not None]
        if isinstance(patches, str):
            columns.append(patches)
            patch_kwarg = dict(patch_name=patches)
        elif isinstance(patches, int):
            patch_kwarg = dict(n_patches=patches)
        elif isinstance(patches, Coordinate):
            patch_kwarg = dict(patch_centers=patches)
        elif isinstance(patches, Catalog):
            patch_kwarg = dict(patch_centers=patches.centers)
        else:
            raise TypeError(
                "'patches' must be either of type 'str' (col. name), 'int' "
                "(number of patches), or 'Catalog' or 'Coordinate' (specify "
                "centers)"
            )

        cls._logger.info("reading catalog file '%s'", filepath)
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
            cls._logger.debug("sparse sampling data %ix", sparse)
            data = data[::sparse]
        return cls(
            data,
            ra,
            dec,
            **patch_kwarg,
            redshift_name=redshift,
            weight_name=weight,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    def from_cache(cls, cache_directory: str, progress: bool = False) -> Catalog:
        """TODO: See factory"""
        cls._logger.info("restoring from cache directory '%s'", cache_directory)
        new = cls.__new__(cls)
        # load the patch properties
        fpath = os.path.join(cache_directory, "properties.feather")
        property_df = pd.read_feather(fpath)
        # transform data frame to dictionaries
        ids = property_df["ids"]
        centers = Coord3D.from_array(property_df[["x", "y", "z"]].to_numpy())
        radii = DistSky(property_df["r"].to_numpy())
        # transform to dictionary
        centers = {pid: center for pid, center in zip(ids, centers)}
        radii = {pid: radius for pid, radius in zip(ids, radii)}
        # load the patches
        limits = LimitTracker()
        new._patches = {}
        patch_files = list(os.listdir(cache_directory))
        if progress:
            patch_files = job_progress_bar(patch_files)
        for path in patch_files:
            if not path.startswith("patch"):
                continue
            abspath = os.path.join(cache_directory, path)
            if not os.path.isfile(abspath):
                continue
            patch_id = patch_id_from_path(path)
            patch = PatchCatalog.from_cached(
                abspath, center=centers.get(patch_id), radius=radii.get(patch_id)
            )
            limits.update(patch.redshifts)
            patch.unload()
            new._patches[patch.id] = patch
        new._zmin, new._zmax = limits.get()
        return new

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            loaded=self.is_loaded(),
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshifts(),
        )
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    def __len__(self) -> int:
        return sum(len(patch) for patch in self._patches.values())

    def __getitem__(self, item: int) -> Any:
        return self._patches[item]

    @property
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog."""
        return sorted(self._patches.keys())

    @property
    def n_patches(self) -> int:
        """The number of spatial patches of this catalogue."""
        pass

    def __iter__(self) -> Iterator:
        for patch_id in self.ids:
            patch = self._patches[patch_id]
            loaded = patch.is_loaded()
            patch.load()
            yield patch
            if not loaded:
                patch.unload()

    def is_loaded(self) -> bool:
        """Indicates whether the catalog data is loaded.

        Always ``True`` if no cache is used. If the catalog is unloaded, data
        will be read from cache every time data is accessed."""
        return all([patch.is_loaded() for patch in self._patches.values()])

    def load(self) -> None:
        """Permanently load data from cache into memory.

        Raises a :obj:`~yaw.catalogs.scipy.patches.CachingError` if no cache
        is configured.
        """
        self._logger.debug("bulk loading catalog")
        for patch in self._patches.values():
            patch.load()

    def unload(self) -> None:
        """Unload data from memory if a disk cache is provided."""
        self._logger.debug("bulk unloading catalog")
        for patch in self._patches.values():
            patch.unload()

    def has_redshifts(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        return all(patch.has_redshifts() for patch in self._patches.values())

    def has_weights(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        return all(patch.has_weights() for patch in self._patches.values())

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky(self.ra, self.dec)

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians."""
        return np.concatenate([patch.ra for patch in iter(self)])

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians."""
        return np.concatenate([patch.dec for patch in iter(self)])

    @property
    def redshifts(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available."""
        if self.has_redshifts():
            return np.concatenate([patch.redshifts for patch in iter(self)])
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        """Get the object weights as array or ``None`` if not available."""
        weights = []
        for patch in iter(self):
            if patch.has_weights():
                weights.append(patch.weights)
            else:
                weights.append(np.ones(len(patch)))
        return np.concatenate(weights)

    @property
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object as array."""
        return np.concatenate([np.full(len(patch), patch.id) for patch in iter(self)])

    def get_min_redshift(self) -> float:
        """Get the minimum redshift or ``None`` if not available."""
        return self._zmin

    def get_max_redshift(self) -> float:
        """Get the maximum redshift or ``None`` if not available."""
        return self._zmax

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""
        return self.get_totals().sum()

    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects in each
        patch."""
        return np.array([patch.total for patch in self._patches.values()])

    @property
    def centers(self) -> CoordSky:
        """Get a vector of sky coordinates of the patch centers in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky.from_coords([self._patches[pid].center for pid in self.ids])

    @property
    def radii(self) -> DistSky:
        """Get a vector of angular separations in radians that describe the
        patch sizes.

        The radius of the patch is defined as the maximum angular distance of
        any object from the patch center.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return DistSky.from_dists([self._patches[pid].radius for pid in self.ids])

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: Catalog | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[str, NormalisedCounts]:
        """Count pairs between objects at a given separation and in bins of
        redshift.

        If another catalog instance is passed to ``other``, then pairs are
        formed between these catalogues (cross), otherwise pairs are formed with
        the catalog (auto). Pairs are counted in bins of redshift, as defined in
        the configuration object (``config``). Pairs are only considered within
        fixed angular scales that are computed from the physical scales in the
        configuration and the mid of the current redshift bin.

        Args:
            config (:obj:`yaw.Configuration`):
                Configuration object that defines measurement scales, redshift
                binning, cosmological model, and various backend specific
                parameters.
            binned (:obj:`bool`):
                Whether to apply the redshift binning to the second catalogue
                (see ``other``).
            other (Catalog instance, optional):
                Second catalog instance used for cross-catalogue pair counting.
                Catalogue must use the same backend.
            linkage (:obj:`~yaw.catalogs.linkage.PatchLinkage`, optional):
                Linkage object that defines with patches must be correlated for
                a given scales and which patch combinations can be skipped. Can
                be used for the ``scipy`` backend to count pairs consistently
                between multiple catalogue instances.
            progress (:obj:`bool`):
                Show a progress indication, depends on backend.

        There are three different modes of operation that are determined by the
        combination of the ``binned`` and ``other`` parameters:

        1. If no second catalogue is provided, pairs are counted within the
           catalogue while applying the redshift binning.
        2. If a second catalogue is provided and ``binned=True``, pairs are
           counted between the catalogues and the binning is applied to both
           cataluges.
        3. If a second catalogue is provided and ``binned=False``, the redshift
           binning is not applied to the second catalogue, otherwise above.

        The catalogue from the calling instance of :meth:`correlate` has always
        redshift binning applied.
        """
        n1 = long_num_format(len(self))
        n2 = long_num_format(len(self) if other is None else len(other))
        self._logger.debug(
            "correlating with %sbinned catalog (%sx%s) in %d redshift bins",
            "" if binned else "un",
            n1,
            n2,
            config.binning.zbin_num,
        )

        auto = other is None
        patch1_list, patch2_list = utils.get_patch_list(
            self, other, config, linkage, auto
        )

        # process the patch pairs, add an optional progress bar
        n_jobs = len(patch1_list)
        bin1 = self.has_redshifts()
        bin2 = binned if other is not None else True
        iter_args = zip(
            patch1_list, patch2_list, repeat(config), repeat(bin1), repeat(bin2)
        )
        if progress:
            iter_args = job_progress_bar(iter_args, total=n_jobs)
        with multiprocessing.Pool(config.backend.get_threads(n_jobs)) as pool:
            patch_datasets = list(pool.imap_unordered(_worker_correlate, iter_args))

        # merge the pair counts from all patch combinations
        return utils.merge_pairs_patches(patch_datasets, config, self.n_patches, auto)

    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False,
    ) -> HistData:
        """
        Compute a histogram of the object redshifts from the binning defined in
        the provided configuration.

        Args:
            config (:obj:`~yaw.config.Configuration`):
                Defines the bin edges used for the histogram.
            sampling_config (:obj:`~yaw.config.ResamplingConfig`, optional):
                Specifies the spatial resampling for error estimates.
            progress (:obj:`bool`):
                Show a progress bar.

        Returns:
            HistData:
                Object holding the redshift histogram
        """
        self._logger.info("computing true redshift distribution")
        if not self.has_redshifts():
            raise ValueError("catalog has no redshifts")

        # compute the reshift histogram in each patch
        n_jobs = self.n_patches
        iter_args = zip(self._patches.values(), repeat(config.binning.zbins))
        if progress:
            iter_args = job_progress_bar(iter_args, total=n_jobs)
        with multiprocessing.Pool(config.backend.get_threads(n_jobs)) as pool:
            hist_counts = list(pool.imap_unordered(_worker_true_redshifts, iter_args))

        # construct the output data samples
        return utils.merge_histogram_patches(
            np.array(hist_counts), config.binning.zbins, sampling_config
        )
