from __future__ import annotations

import itertools
import os
from collections.abc import Iterator

import astropandas as apd
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas import DataFrame

from yet_another_wizz.core.catalog import CatalogBase, PatchLinkage
from yet_another_wizz.core.config import Configuration
from yet_another_wizz.core.coordinates import position_sphere2sky
from yet_another_wizz.core.cosmology import TypeCosmology, r_kpc_to_angle
from yet_another_wizz.core.parallel import ParallelHelper
from yet_another_wizz.core.redshifts import NzTrue
from yet_another_wizz.core.resampling import ArrayDict, PairCountResult
from yet_another_wizz.core.utils import (
    LimitTracker, Timed, TypePatchKey, TypeScaleKey, scales_to_keys)

from yet_another_wizz.scipy.patches import (
    PatchCatalog, patch_id_from_path, create_patches, assign_patches)


def _count_pairs_thread(
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


def _histogram_thread(
    patch: PatchCatalog,
    z_bins: NDArray[np.float_]
) -> NDArray[np.float_]:
    is_loaded = patch.is_loaded()
    patch.load()
    counts, _ = np.histogram(patch.redshifts, z_bins, weights=patch.weights)
    if not is_loaded:
        patch.unload()
    return counts


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
                    "must be provided")
        if unload:
            if not os.path.exists(cache_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{cache_directory}'")

        # create new patches
        if patch_mode != "dividing":
            if patch_mode == "creating":
                patch_centers, patch_ids = create_patches(
                    data[ra_name].to_numpy(), data[dec_name].to_numpy(),
                    n_patches)
            elif patch_mode == "applying":
                if isinstance(patch_centers, Catalog):
                    patch_centers = position_sphere2sky(patch_centers.centers)
                patch_ids = assign_patches(
                    patch_centers,
                    data[ra_name].to_numpy(), data[dec_name].to_numpy())
            patch_name = "patch"  # the default name
            data[patch_name] = patch_ids
            centers = {pid: pos for pid, pos in enumerate(patch_centers)}
        else:
            centers = dict()  # this can be empty

        # run groupby first to avoid any intermediate copies of full data
        if unload:
            msg = f"{patch_mode} and caching patches"
        else:
            msg = f"{patch_mode} patches"
        with Timed(msg):
            limits = LimitTracker()
            patches: dict[int, PatchCatalog] = {}
            for patch_id, patch_data in data.groupby(patch_name):
                if patch_id < 0:
                    raise ValueError("negative patch IDs are not supported")
                # drop extra columns
                patch_data = patch_data.drop(columns=[
                    col for col in patch_data.columns if col not in renames])
                patch_data.rename(columns=renames, inplace=True)
                patch_data.reset_index(drop=True, inplace=True)
                # look up the center of the patch if given
                kwargs = dict(center=centers.get(patch_id))
                if unload:
                    # data will be written as feather file and loaded on demand
                    kwargs["cachefile"] = os.path.join(
                        cache_directory, f"patch_{patch_id:.0f}.feather")
                patch = PatchCatalog(int(patch_id), patch_data, **kwargs)
                limits.update(patch.redshifts)
                if unload:
                    patch.unload()
                patches[patch.id] = patch
            self._zmin, self._zmax = limits.get()
            self._patches = patches

        # also store the patch properties
        if unload:
            property_df = pd.DataFrame(dict(ids=self.ids))
            for colname, values in zip("xyz", self.centers.T):
                property_df[colname] = values
            property_df["r"] = self.radii
            fpath = os.path.join(cache_directory, "properties.feather")
            property_df.to_feather(fpath)

    @classmethod
    def from_cache(
        cls,
        cache_directory: str
    ) -> Catalog:
        new = cls.__new__(cls)
        # load the patch properties
        fpath = os.path.join(cache_directory, "properties.feather")
        property_df = pd.read_feather(fpath)
        # transform data frame to dictionaries
        ids = property_df["ids"]
        centers = property_df[["x", "y", "z"]].to_numpy()
        centers = {pid: center for pid, center in zip(ids, centers)}
        radii = property_df["r"].to_numpy()
        radii = {pid: radius for pid, radius in zip(ids, radii)}
        # load the patches
        limits = LimitTracker()
        new._patches = {}
        for path in os.listdir(cache_directory):
            if not path.startswith("patch"):
                continue
            abspath = os.path.join(cache_directory, path)
            if not os.path.isfile(abspath):
                continue
            patch_id = patch_id_from_path(path)
            patch = PatchCatalog.from_cached(
                abspath,
                center=centers.get(patch_id),
                radius=radii.get(patch_id))
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
        for patch in self._patches.values():
            patch.load()

    def unload(self) -> None:
        for patch in self._patches.values():
            patch.unload()

    def has_redshifts(self) -> bool:
        return all(patch.has_redshifts() for patch in self._patches.values())

    @property
    def ra(self) -> NDArray[np.float_]:
        return np.concatenate([patch.ra for patch in iter(self)])

    @property
    def dec(self) -> NDArray[np.float_]:
        return np.concatenate([patch.dec for patch in iter(self)])

    @property
    def redshifts(self) -> NDArray[np.float_] | None:
        if self.has_redshifts():
            return np.concatenate([
                patch.redshift for patch in iter(self)])
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        weights = []
        for patch in iter(self):
            if patch.has_weights():
                weights.append(patch.weights)
            else:
                weights.append(np.ones(len(patch)))
        return np.concatenate(weights)

    @property
    def patch(self) -> NDArray[np.int_]:
        return np.concatenate([
            np.full(len(patch), patch.id) for patch in iter(self)])

    def get_min_redshift(self) -> float:
        return self._zmin

    def get_max_redshift(self) -> float:
        return self._zmax

    @property
    def total(self) -> float:
        return self.get_totals().sum()

    def get_totals(self) -> NDArray[np.float_]:
        return np.array([patch.total for patch in self._patches.values()])

    @property
    def centers(self) -> NDArray[np.float_]:
        return np.array([patch.center for patch in iter(self)])

    @property
    def radii(self) -> NDArray[np.float_]:
        return np.array([patch.radius for patch in iter(self)])

    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: Catalog | None = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        auto = other is None
        if not auto and not isinstance(other, Catalog):
            raise TypeError

        # generate the patch linkage (sufficiently close patches)
        if config.crosspatch:
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = 0.05  # TODO: resonable? lower redshift => more overlap
            max_ang_rad = r_kpc_to_angle(
                config.scales, z_ref, config.cosmology).max()
        else:
            max_ang_rad = 0.0  # only relevenat for cross-patch
        cat_for_linkage = self
        if not auto:
            if len(other) > len(self):
                cat_for_linkage = other
        linkage = PatchLinkage.from_catalog(cat_for_linkage, max_ang_rad)
        patch1_list, patch2_list = linkage.get_patches(
            self, other, config.crosspatch)
        n_jobs = len(patch1_list)

        # prepare job scheduling
        pool = ParallelHelper(
            function=_count_pairs_thread,
            n_items=n_jobs, num_threads=min(n_jobs, config.num_threads))
        # patch1: PatchCatalog
        pool.add_iterable(patch1_list)
        # patch2: PatchCatalog
        pool.add_iterable(patch2_list)
        # scales: NDArray[np.float_]
        pool.add_constant(config.scales)
        # cosmology: TypeCosmology
        pool.add_constant(config.cosmology)
        # z_bins: NDArray[np.float_]
        pool.add_constant(config.zbins)
        # bin1: bool
        pool.add_constant(True)
        # bin2: bool
        pool.add_constant(binned if other is not None else True)
        # dist_weight_scale: float | None
        pool.add_constant(config.weight_scale)
        # weight_res: int
        pool.add_constant(config.resolution)

        n_bins = len(config.zbins) - 1
        n_patches = self.n_patches()
        # execute, unpack the data
        totals1 = np.zeros((n_patches, n_bins))
        totals2 = np.zeros((n_patches, n_bins))
        count_dict = {key: {} for key in scales_to_keys(config.scales)}
        for (id1, id2), (total1, total2), counts in pool.iter_result():
            # record total weight per bin, overwriting OK since identical
            totals1[id1] = total1
            totals2[id2] = total2
            # record counts at each scale
            for scale_key, count in counts.items():
                count_dict[scale_key][(id1, id2)] = count

        # get mask of all used cross-patch combinations, upper triangle if auto
        mask = linkage.get_mask(self, other, config.crosspatch)
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
                binning=pd.IntervalIndex.from_breaks(config.zbins))
        return result

    def true_redshifts(self, config: Configuration) -> NzTrue:
        if not self.has_redshifts():
            raise ValueError("catalog has no redshifts")
        # compute the reshift histogram in each patch
        pool = ParallelHelper(
            function=_histogram_thread,
            n_items=self.n_patches(),
            num_threads=min(self.n_patches(), config.num_threads))
        # patch: PatchCatalog
        pool.add_iterable(list(self))
        # NDArray[np.float_]
        pool.add_constant(config.zbins)
        with Timed("processing true redshifts"):
            hist_counts = list(pool.iter_result())
        return NzTrue(np.array(hist_counts), config.zbins)
