from __future__ import annotations

import json
import logging
import multiprocessing
import os
from collections.abc import Iterable, Iterator
from itertools import repeat
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
import polars as pl

from yaw.catalogs import streaming, utils
from yaw.catalogs.patches import (
    PatchCatalog,
    assign_patches,
    check_columns,
    create_patches,
)
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, DistSky
from yaw.core.logging import TimedLog
from yaw.core.utils import job_progress_bar, long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame

    from yaw.catalogs import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.core.containers import PatchCorrelationData
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = ["Catalog"]


def get_patch_mode(
    patches: Any, data: DataFrame
) -> Literal["dividing", "creating", "applying"]:
    if "patch" in data:
        patch_mode = "dividing"
    elif patches is None:
        raise ValueError(
            "either 'data' must contain 'patch' column or 'patches' must be provided"
        )
    elif isinstance(patches, int):
        patch_mode = "creating"
    elif isinstance(patches, (Catalog, Coordinate)):
        patch_mode = "applying"
    else:
        raise TypeError(f"invalid type {type(patches)} for 'patches'")
    return patch_mode


def _worker_correlate(
    args: tuple[PatchCatalog, PatchCatalog, Configuration, bool, bool]
) -> PatchCorrelationData:
    return utils.count_pairs_patches(*args)


def _worker_true_redshifts(
    args: tuple[PatchCatalog, NDArray[np.float_]]
) -> NDArray[np.float_]:
    return utils.count_histogram_patch(*args)


class Catalog:
    """TODO: See factory"""

    _logger = logging.getLogger("yaw.catalog")

    def __init__(
        self,
        data: DataFrame,
        patches: Catalog | Coordinate | int | None = None,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> None:
        # run some prechecks
        if len(data) == 0:
            raise ValueError("data catalog is empty")
        check_columns(data, extra=("patch",))

        # check if data should be cached and dropped from memory
        cache = cache_directory is not None
        if cache:
            if not os.path.exists(cache_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{cache_directory}'"
                )
            self._logger.debug("using cache directory '%s'", cache_directory)

        # create the patch column and get the number of patches
        data = self.data
        patch_mode = get_patch_mode(patches, data)
        if patch_mode == "dividing":
            n_patches, centers = self._patch_col_divide(data)
        elif patch_mode == "creating":
            n_patches, centers = self._patch_col_create(patches, data)
        else:  # applying
            n_patches, centers = self._patch_col_apply(patches, data)
        # NOTE: centers is empty when 'dividing'

        # create the patches
        metadata = []
        patches: dict[int, PatchCatalog] = {}
        patch_iter = data.groupby("patch")
        if progress:
            patch_iter = job_progress_bar(patch_iter, total=n_patches)

        n_obj_str = long_num_format(len(data))
        with TimedLog(self._logger.info, f"processed {n_obj_str} records"):
            for patch_id, patch_data in patch_iter:
                if patch_id < 0:
                    raise ValueError("negative patch IDs are not supported")
                # build the patch
                if cache:
                    cachefile = os.path.join(
                        cache_directory, f"patch_{patch_id:d}.feather"
                    )
                else:
                    cachefile = None
                patch = PatchCatalog(
                    patch_id,
                    data=patch_data.drop(columns="patch").reset_index(drop=True),
                    cachefile=cachefile,
                    center=centers.get(patch_id),
                )
                patches[patch_id] = patch
                # get metadata and clean up

                redshift = patch.redshift
                meta = dict(
                    id=patch_id,
                    zmin=redshift.min(),
                    zmax=redshift.max(),
                    x=patch._center.x,
                    y=patch._center.y,
                    z=patch._center.z,
                    r=patch.radius.values,
                )
                meta.update(
                    {
                        attr: getattr(patch, f"_{attr}")
                        for attr in ("length", "total", "has_z", "has_w")
                    }
                )
                metadata.append(meta)
                if cache:
                    patch.unload()
            if progress:  # clean up if any patch was empty and skipped
                patch_iter.close()
        self._patches = patches

        # store the patch metadata
        if cache:
            fpath = os.path.join(cache_directory, "properties.json")
            with open(fpath, "w") as f:
                json.dump(metadata, f)
        self._set_z_limits(metadata)

    def _patch_col_divide(self, data: DataFrame) -> tuple[int, dict[int, Coord3D]]:
        # issue log
        log_msg = "dividing data into %i predefined patches"
        n_patches = data["patch"].nunique()
        self._logger.debug(log_msg, n_patches)
        # 'patch' column already exists
        return n_patches, dict()

    def _patch_col_apply(
        self,
        centers: Catalog | Coordinate,
        data: DataFrame,
    ) -> tuple[int, dict[int, Coord3D]]:
        # get the centers
        if isinstance(centers, Catalog):
            centers = centers.centers.to_3d()
        else:
            centers = centers.to_3d()
        # issue log
        log_msg = "applying %i patches from external data"
        n_patches = len(centers)
        self._logger.debug(log_msg, n_patches)
        # assign patch column
        positions = CoordSky.from_array(data[["ra", "dec"]].to_numpy())
        data["patch"] = assign_patches(centers, positions)
        return n_patches, {pid: pos for pid, pos in enumerate(centers)}

    def _patch_col_create(
        self,
        n_patches: int,
        data: DataFrame,
    ) -> tuple[int, dict[int, Coord3D]]:
        # issue log
        log_msg = "creating %i patches"
        self._logger.debug(log_msg, n_patches)
        # compute centers from a sparse sample
        positions = CoordSky.from_array(data[["ra", "dec"]].to_numpy())
        SUBSET_SIZE = 1000 * n_patches
        if len(data) < SUBSET_SIZE:
            positions_sparse = positions
        else:
            rng = np.random.default_rng(seed=12345)
            take = rng.integers(0, len(data), size=SUBSET_SIZE)
            positions_sparse = positions[take]
        centers, _ = create_patches(positions_sparse, n_patches)
        # assign patch column
        data["patch"] = assign_patches(centers, positions)
        return n_patches, {pid: pos for pid, pos in enumerate(centers)}

    def _set_z_limits(self, metadata: dict[str, dict]) -> None:
        self._zmin = min(meta["zmin"] for meta in metadata.values())
        self._zmax = max(meta["zmin"] for meta in metadata.values())

    @classmethod
    def from_dataframe(
        cls,
        data: DataFrame,
        patches: str | Catalog | Coordinate | int,
        *,
        ra_name: str,
        dec_name: str,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        degrees: bool = True,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> Catalog:
        # build a dataframe with correct column names
        normalised = data.rename(
            columns={
                ra_name: "ra",
                dec_name: "dec",
                redshift_name: "redshift",
                weight_name: "weight",
            }
        )
        if degrees:
            normalised["ra"] = np.deg2rad(normalised["ra"])
            normalised["dec"] = np.deg2rad(normalised["dec"])
        # add the patch index column if provided
        is_patch_col = isinstance(patches, str)
        if is_patch_col:
            normalised["patch"] = data[patches]
        # build the catalog
        return cls(
            data=normalised,
            patches=None if is_patch_col else patches,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    def from_records(
        cls,
        data: DataFrame,
        patches: Iterable | Catalog | Coordinate | int,
        *,
        ra: Iterable,
        dec: Iterable,
        redshift: Iterable | None = None,
        weight: Iterable | None = None,
        degrees: bool = True,
        cache_directory: str | None = None,
        progress: bool = True,
    ) -> Catalog:
        # pack the records into a dataframe with correct column names
        if degrees:
            ra = np.deg2rad(ra)
            dec = np.deg2rad(dec)
        normalised = pd.DataFrame(dict(ra=ra, dec=dec))
        if redshift is not None:
            normalised["redshift"] = redshift
            normalised["weight"] = weight
        # add the patch index column if provided
        is_patch_col = not isinstance(patches, (Catalog, Coordinate, int))
        if is_patch_col:
            normalised["patch"] = patches
        # build the catalog
        return cls(
            data=normalised,
            patches=None if is_patch_col else patches,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    def from_cache(cls, cache_directory: str, progress: bool = False) -> Catalog:
        """TODO: See factory"""
        cls._logger.info("restoring from cache directory '%s'", cache_directory)
        new = cls.__new__(cls)
        # load the patch properties
        fpath = os.path.join(cache_directory, "properties.json")
        with open(fpath) as f:
            metadata = json.load(f)
        meta_iter = iter(metadata)
        if progress:
            meta_iter = job_progress_bar(meta_iter, total=len(metadata))
        # create the patches without loading the data
        new._patches = {}
        for meta in meta_iter:
            patch_id = meta["id"]
            cachefile = os.path.join(cache_directory, f"patch_{patch_id:d}.feather")
            center = Coord3D(meta["x"], meta["y"], meta["z"])
            radius = DistSky(meta["r"])
            new._patches[patch_id] = PatchCatalog.from_cached(
                cachefile,
                length=meta["length"],
                total=meta["total"],
                has_z=meta["has_z"],
                has_w=meta["has_w"],
                center=center,
                radius=radius,
            )
        new._set_z_limits(metadata)
        return new

    @classmethod
    def from_file(
        cls,
        path: str,
        patches: str | int | Catalog | Coordinate,
        *,
        ra_name: str,
        dec_name: str,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        degrees: bool = True,
        reader: type[streaming.Reader] | None = None,
        cache_directory: str | None = None,
        progress: bool = False,
        **kwargs,
    ) -> Catalog:
        """TODO: See factory"""
        cls._logger.info("reading catalog file '%s'", path)
        # get the correct reader for the input table file
        if reader is None:
            reader = streaming.get_reader(path)
        # get the columns to load
        rename = {ra_name: "ra", dec_name: "dec"}
        extra = []
        if redshift_name is not None:
            rename[redshift_name] = "redshift"
            extra.append("redshift")
        if weight_name is not None:
            rename[weight_name] = "weight"
            extra.append("weight")
        if isinstance(patches, str):
            rename[patches] = "patch"
            extra.append("patch")

        # load the data in chunks
        loader: streaming.Reader
        with reader(path, rename.keys()) as loader:
            for chunk in loader.iter():
                chunk = chunk.rename(rename).select(
                    pl.col(["ra", "dec"]).radians(), pl.col(extra)
                )

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
