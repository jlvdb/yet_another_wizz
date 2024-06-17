from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterator
from itertools import repeat
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from yaw.catalogs import BaseCatalog, PatchLinkage
from yaw.catalogs.scipy import utils
from yaw.catalogs.scipy.patches import (
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

    from yaw.core.containers import PatchCorrelationData


__all__ = ["ScipyCatalog"]


def _worker_correlate(
    args: tuple[PatchCatalog, PatchCatalog, Configuration, bool, bool]
) -> PatchCorrelationData:
    return utils.count_pairs_patches(*args)


def _worker_true_redshifts(
    args: tuple[PatchCatalog, NDArray[np.float64]]
) -> NDArray[np.float64]:
    return utils.count_histogram_patch(*args)


class ScipyCatalog(BaseCatalog):
    """An implementation of the :obj:`BaseCatalog` using a wrapper around
    :obj:`scipy.spatial.cKDTree` for the pair counting, which is implemented in
    :obj:`yaw.catalogs.scipy.kdtree`. Fully supports caching.

    .. Note::

        This is currently the default backend and has the best support and
        performance. Currently, trees cannot be shared across the
        multiprocessing interface and must be rebuilt every time a patch is
        used for pair counting again.
    """

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
        progress: bool = True,
    ) -> None:
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
                if isinstance(patch_centers, BaseCatalog):
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
            patch_iter = data.groupby(patch_name, observed=True)
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
    def from_cache(cls, cache_directory: str, progress: bool = False) -> ScipyCatalog:
        super().from_cache(cache_directory)
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

    def __len__(self) -> int:
        return sum(len(patch) for patch in self._patches.values())

    def __getitem__(self, item: int) -> PatchCatalog:
        return self._patches[item]

    @property
    def ids(self) -> list[int]:
        return sorted(self._patches.keys())

    @property
    def n_patches(self) -> int:
        # seems ok to drop the last patch if that is empty and therefore missing
        return max(self._patches.keys()) + 1

    def __iter__(self) -> Iterator[PatchCatalog]:
        for patch_id in self.ids:
            patch = self._patches[patch_id]
            loaded = patch.is_loaded()
            patch.load()
            yield patch
            if not loaded:
                patch.unload()

    def is_loaded(self) -> bool:
        return all([patch.is_loaded() for patch in self._patches.values()])

    def load(self) -> None:
        super().load()
        for patch in self._patches.values():
            patch.load()

    def unload(self) -> None:
        super().unload()
        for patch in self._patches.values():
            patch.unload()

    def has_redshifts(self) -> bool:
        return all(patch.has_redshifts() for patch in self._patches.values())

    def has_weights(self) -> bool:
        return all(patch.has_weights() for patch in self._patches.values())

    @property
    def ra(self) -> NDArray[np.float64]:
        return np.concatenate([patch.ra for patch in iter(self)])

    @property
    def dec(self) -> NDArray[np.float64]:
        return np.concatenate([patch.dec for patch in iter(self)])

    @property
    def redshifts(self) -> NDArray[np.float64] | None:
        if self.has_redshifts():
            return np.concatenate([patch.redshifts for patch in iter(self)])
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float64]:
        weights = []
        for patch in iter(self):
            if patch.has_weights():
                weights.append(patch.weights)
            else:
                weights.append(np.ones(len(patch)))
        return np.concatenate(weights)

    @property
    def patch(self) -> NDArray[np.int64]:
        return np.concatenate([np.full(len(patch), patch.id) for patch in iter(self)])

    def get_min_redshift(self) -> float:
        return self._zmin

    def get_max_redshift(self) -> float:
        return self._zmax

    @property
    def total(self) -> float:
        return self.get_totals().sum()

    def get_totals(self) -> NDArray[np.float64]:
        return np.array([patch.total for patch in self._patches.values()])

    @property
    def centers(self) -> CoordSky:
        return CoordSky.from_coords([self._patches[pid].center for pid in self.ids])

    @property
    def radii(self) -> DistSky:
        return DistSky.from_dists([self._patches[pid].radius for pid in self.ids])

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: ScipyCatalog | None = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[str, NormalisedCounts]:
        super().correlate(config, binned, other, linkage)
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
        super().true_redshifts(config)
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
