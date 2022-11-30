from __future__ import annotations

import gc
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence

import astropandas as apd
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Interval, IntervalIndex
from scipy.spatial import distance_matrix

from yet_another_wizz.kdtree import EmptyKDTree, KDTree, SphericalKDTree
from yet_another_wizz.utils import Timed, TypePatchKey


class NotAPatchFileError(Exception):
    pass


class Patch(ABC):

    id = 0
    cachefile = None
    _data = DataFrame()
    _len = 0
    _has_z = False
    _has_weights = False

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, loaded={self.is_loaded()})"
        return s

    def __len__(self) -> int:
        return self._len

    def total(self) -> float:
        if self.has_weights():
            return self.weights.sum()
        else:
            return float(len(self))

    def is_loaded(self) -> bool:
        return self._data is not None

    def require_loaded(self) -> None:
        if not self.is_loaded():
            raise AttributeError("data is not loaded")

    @abstractmethod
    def load(self, use_threads: bool = True) -> None:
        NotImplemented

    @abstractmethod
    def unload(self) -> None:
        self._data = None
        gc.collect()

    def has_z(self) -> bool:
        return self._has_z

    def has_weights(self) -> bool:
        return self._has_weights

    @property
    def data(self) -> DataFrame:
        self.require_loaded()
        return self._data

    @property
    def pos(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data[["ra", "dec"]].to_numpy()

    @property
    def ra(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["ra"].to_numpy()

    @property
    def dec(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["dec"].to_numpy()

    @property
    def z(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_z():
            return self._data["z"].to_numpy()
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_weights():
            return self._data["weights"].to_numpy()
        else:
            return None

    @abstractmethod
    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, Patch]]:
        # allow_no_redshift=True may be used for randoms and should yield self
        NotImplemented

    @abstractmethod
    def get_tree(self, **kwargs) -> KDTree:
        NotImplemented


class EmptyPatch(Patch):

    def __init__(
        self,
        id: int,
        has_z: bool,
        has_weights: bool
    ) -> None:
        self.id = id
        self._has_z = has_z
        self._has_weights = has_weights
        # construct dummy dataframe with no data but the correct columns
        dummy_data = {
            "ra": pd.Series(dtype=np.float_),
            "dec": pd.Series(dtype=np.float_)}
        if has_z:
            dummy_data["z"] = pd.Series(dtype=np.float_)
        if has_weights:
            dummy_data["weights"] = pd.Series(dtype=np.float_)
        self._data = DataFrame(dummy_data)

    def load(self, use_threads: bool = True) -> None:
        pass

    def unload(self) -> None:
        pass

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, EmptyPatch]]:
        if not allow_no_redshift and not self.has_z():
            raise ValueError("no redshifts for iteration provdided")
        for intv in IntervalIndex.from_breaks(z_bins):
            yield intv, self

    def get_tree(self, **kwargs) -> EmptyKDTree:
        return EmptyKDTree()


class PatchCatalog(Patch):

    def __init__(
        self,
        id: int,
        data: DataFrame,
        cachefile: str | None = None
    ) -> None:
        self.id = id
        if "ra" not in data:
            raise KeyError("right ascension column ('ra') is required")
        if "dec" not in data:
            raise KeyError("declination column ('dec') is required")
        if not set(data.columns) <= set(["ra", "dec", "z", "weights"]):
            raise KeyError(
                "'data' contains unidentified columns, optional columns are "
                "restricted to 'z' and 'weights'")
        # if there is a file path, store the file
        if cachefile is not None:
            data.to_feather(cachefile)
            self.cachefile = cachefile
        self._data = data
        self._len = len(data)
        self._has_z = "z" in data
        self._has_weights = "weights" in data

    @classmethod
    def from_cached(cls, cachefile: str) -> PatchCatalog:
        ext = ".feather"
        if not cachefile.endswith(ext):
            raise NotAPatchFileError("input must be a .feather file")
        prefix, patch_id = cachefile[:-len(ext)].rsplit("_", 1)
        # create the data instance
        new = cls.__new__(cls)
        new.id = int(patch_id)
        new.cachefile = cachefile
        try:
            new._data = pd.read_feather(cachefile)
            raise KeyError
        except Exception as e:
            args = ()
            if hasattr(e, "args"):
                args = e.args
            raise NotAPatchFileError(*args)
        new._len = len(new._data)
        new._has_z = "z" in new.data
        new._has_weights = "weights" in new.data
        return new

    def load(self, use_threads: bool = True) -> None:
        if not self.is_loaded():
            if self.cachefile is None:
                raise ValueError("no datapath provided to load the data")
            self._data = pd.read_feather(
                self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        self._data = None

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, PatchCatalog]]:
        if not allow_no_redshift and not self.has_z():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in IntervalIndex.from_breaks(z_bins):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(pd.cut(self.z, z_bins)):
                yield intv, PatchCatalog(self.id, bin_data)

    def get_info(
        self,
        n_subset: int | None = 1000
    ) -> dict[str, ArrayLike]:
        if n_subset is not None:
            n_subset = min(len(self), int(n_subset))
            which = np.random.randint(0, len(self), size=n_subset)
            pos = self.pos[which]
        else:
            pos = self.pos
        xyz = SphericalKDTree.position_sky2sphere(pos)
        # compute mean coordinate, which will not be located on the unit sphere
        mean_xyz = np.mean(xyz, axis=0)
        # map back onto unit sphere
        mean_sky = SphericalKDTree.position_sphere2sky(mean_xyz)
        mean_sphere = SphericalKDTree.position_sky2sphere(mean_sky)
        # compute maximum distance to any of the data points
        max_dist = np.sqrt(np.sum((xyz - mean_sphere)**2, axis=1)).max()
        return dict(center=mean_sphere, radius=max_dist, items=len(self))

    def get_tree(self, **kwargs) -> SphericalKDTree:
        return SphericalKDTree(self.ra, self.dec, self.weights, **kwargs)


class PatchCollection(Sequence):

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        patch_name: str,
        *,
        z_name: str | None = None,
        weight_name: str | None = None,
        n_patches: int | None = None,
        cache_directory: str | None = None
    ) -> None:
        if len(data) == 0:
            raise ValueError("data catalog is empty")
        # check if the columns exist
        renames = {ra_name: "ra", dec_name: "dec"}
        if z_name is not None:
            renames[z_name] = "z"
        if weight_name is not None:
            renames[weight_name] = "weights"
        for col_name, kind in renames.items():
            if col_name not in data:
                raise KeyError(f"column {kind}='{col_name}' not found in data")
        # check if patches should be written and unloaded from memory
        unload = cache_directory is not None
        prefix = "constructing "
        if unload:
            if not os.path.exists(cache_directory):
                raise FileNotFoundError(
                    f"patch directory does not exist: '{cache_directory}'")
            prefix += "and caching "
        # run groupby first to avoid any intermediate copies of full data
        with Timed(prefix + "patches"):
            patches: dict[int, PatchCatalog] = {}
            self._zmin = np.inf
            self._zmax = -np.inf
            for patch_id, patch_data in data.groupby(patch_name):
                if patch_id < 0:
                    raise ValueError("negative patch IDs are not supported")
                patch_data = patch_data.drop("patch", axis=1)
                patch_data.rename(columns=renames, inplace=True)
                patch_data.reset_index(drop=True, inplace=True)
                kwargs = {}
                if unload:
                    # data will be written as feather file and loaded on demand
                    kwargs["cachefile"] = os.path.join(
                        cache_directory, f"patch_{patch_id:.0f}.feather")
                patch = PatchCatalog(int(patch_id), patch_data, **kwargs)
                if patch.has_z():
                    self._zmin = np.minimum(self._zmin, patch.z.min())
                    self._zmax = np.maximum(self._zmax, patch.z.max())
                if unload:
                    patch.unload()
                patches[patch.id] = patch
            if np.isinf(self._zmin):
                self._zmin = None
                self._zmax = None
            self._patches = patches
            self._fill_missing(n_patches)

    def _fill_missing(self, n_patches: int | None = None) -> None:
        # assume that regions are labelled by integers, so fill up any empty
        # regions with dummies
        if n_patches is None:
            n_patches = max(self._patches.keys())
        ref_patch = next(iter(self._patches.values()))  # use any non-empty patch
        for patch_id in range(n_patches):
            if patch_id not in self._patches:
                self._patches[patch_id] = EmptyPatch(
                    patch_id, ref_patch.has_z(), ref_patch.has_weights())

    @classmethod
    def from_file(
        cls,
        filepath: str,
        ra_name: str,
        dec_name: str,
        patch_name: str,
        *,
        z_name: str | None = None,
        weight_name: str | None = None,
        n_patches: int | None = None,
        cache_directory: str | None = None
    ) -> PatchCatalog:
        columns = [
            col for col in [ra_name, dec_name, patch_name, z_name, weight_name]
            if col is not None]
        data = apd.read_auto(filepath, columns=columns)
        return cls(
            data, ra_name, dec_name, patch_name,
            z_name=z_name,
            weight_name=weight_name,
            n_patches=n_patches,
            cache_directory=cache_directory)

    @classmethod
    def restore(
        cls,
        patch_directory: str,
        n_patches: int | None = None
    ) -> PatchCollection:
        new = cls.__new__(cls)
        new._zmin = np.inf
        new._zmax = -np.inf
        # load the patches
        new._patches = {}
        for path in os.listdir(patch_directory):
            patch = PatchCatalog.from_cached(
                os.path.join(patch_directory, path))
            if patch.has_z():
                new._zmin = np.minimum(new._zmin, patch.z.min())
                new._zmax = np.maximum(new._zmax, patch.z.max())
            patch.unload()
            new._patches[patch.id] = patch
        if np.isinf(new._zmin):
            new._zmin = None
            new._zmax = None
        new._fill_missing(n_patches)
        return new

    def __repr__(self):
        length = len(self)
        loaded = sum(1 for patch in iter(self) if patch.is_loaded())
        empty = sum(1 for patch in iter(self) if isinstance(patch, EmptyPatch))
        s = self.__class__.__name__
        s += f"(patches={length}, loaded={loaded}/{length}, empty={empty})"
        return s

    def __contains__(self, item: int) -> bool:
        return item in self._patches

    def __getitem__(self, item: int) -> Patch:
        return self._patches[item]

    def __len__(self) -> int:
        return len(self._patches)

    def __iter__(self) -> Iterator[Patch]:
        for patch_id in sorted(self._patches.keys()):
            yield self._patches[patch_id]

    def iter_loaded(self) -> Iterator[Patch]:
        for patch in iter(self):
            loaded = patch.is_loaded()
            patch.load()
            yield patch
            if not loaded:
                patch.unload()

    def is_loaded(self) -> bool:
        return all([patch.is_loaded() for patch in iter(self)])

    def load(self) -> None:
        return [patch.load() for patch in iter(self)]

    def unload(self) -> None:
        return [patch.unload() for patch in iter(self)]

    def has_z(self) -> bool:
        return all(patch.has_z() for patch in iter(self))  # always equal

    def has_weights(self) -> bool:
        return all(patch.has_weights() for patch in iter(self))  # always equal

    def get_z_min(self) -> float:
        return self._zmin

    def get_z_max(self) -> float:
        return self._zmax

    @property
    def ra(self) -> NDArray[np.float_]:
        return np.concatenate([patch.ra for patch in self.iter_loaded()])

    @property
    def dec(self) -> NDArray[np.float_]:
        return np.concatenate([patch.dec for patch in self.iter_loaded()])

    @property
    def z(self) -> NDArray[np.float_]:
        if self.has_z():
            return np.concatenate([patch.z for patch in self.iter_loaded()])
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        if self.has_weights():
            return np.concatenate([
                patch.weights for patch in self.iter_loaded()])
        else:
            return None

    @property
    def patch(self) -> NDArray[np.float_]:
        return np.concatenate([
            np.full(len(patch), patch.id) for patch in iter(self)])

    def total(self) -> float:
        return np.sum([patch.total() for patch in iter(self)])

    def generate_joblist(
        self,
        max_query_radius_deg: float,
        n_subset: int | None = 1000
    ):
        """Remove duplicates for autocorrelation"""
        # calculate the patch centers and sizes
        infos = [patch.get_info(n_subset) for patch in self.iter_loaded()]
        centers = np.array([info["center"] for info in infos])
        sizes = SphericalKDTree.distance_sphere2sky(  # patch radius in degrees
            np.array([info["radius"] for info in infos]))
        # compute distance in degrees between all patch centers
        dist_deg = SphericalKDTree.distance_sphere2sky(
            distance_matrix(centers, centers))
        # compare minimum separation required for patchs to not overlap
        min_sep_deg = np.add.outer(sizes, sizes)
        # check which patches overlap when factoring in query radius
        overlaps = (dist_deg - max_query_radius_deg) < min_sep_deg
        patch_pairs = []
        for id1, overlap in enumerate(overlaps):
            patch_pairs.extend((id1, id2) for id2 in np.where(overlap)[0])
        return patch_pairs
