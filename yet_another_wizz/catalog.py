from __future__ import annotations

import gc
import os
from collections.abc import Iterator, Sequence

import astropandas as apd
import numpy as np
import pandas as pd
from scipy.cluster import vq
from numpy.typing import NDArray
from pandas import DataFrame, Interval, IntervalIndex
from scipy.spatial import distance_matrix

from yet_another_wizz.kdtree import (
    SphericalKDTree, distance_sphere2sky, position_sky2sphere,
    position_sphere2sky)
from yet_another_wizz.utils import LimitTracker, Timed, TypePatchKey


# Determine patch centers with k-means clustering. The implementation in
# treecorr is quite good, but might not be available. Implement a fallback using
# the scipy.cluster module.
try:
    import treecorr

    def treecorr_patches(
        ra: NDArray[np.float_],
        dec: NDArray[np.float_],
        n_patches: int,
        **kwargs
    ) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
        cat = treecorr.Catalog(
            ra=ra, ra_units="degrees",
            dec=dec, dec_units="degrees",
            npatch=n_patches)
        centers = position_sphere2sky(cat.patch_centers)
        patches = np.copy(cat.patch)
        del cat  # might not be necessary
        return centers, patches
    
    create_patches = treecorr_patches

except ImportError:

    def scipy_patches(
        ra: NDArray[np.float_],
        dec: NDArray[np.float_],
        n_patches: int,
        n_max: int = 500_000
    ) -> tuple[NDArray[np.float_], NDArray[np.int_]]:
        if len(ra) != len(dec):
            raise ValueError("length of 'ra' and 'dec' does not match")
        subset = np.random.randint(0, len(ra), size=min(n_max, len(xyz)))
        # place on unit sphere to avoid coordinate distortions
        xyz = position_sky2sphere(np.column_stack([ra[subset], dec[subset]]))
        centers, _ = vq.kmeans2(xyz[subset], n_patches, minit="points")
        centers = position_sphere2sky(centers)
        patches = assign_patches(centers, ra, dec)
        return centers, patches

    create_patches = scipy_patches


def assign_patches(
    centers_ra_dec: NDArray[np.float_],
    ra: NDArray[np.float_],
    dec: NDArray[np.float_]
) -> NDArray[np.int_]:
    # place on unit sphere to avoid coordinate distortions
    centers_xyz = position_sky2sphere(centers_ra_dec)
    xyz = position_sky2sphere(np.column_stack([ra, dec]))
    patches, _ = vq.vq(xyz, centers_xyz)
    return patches


def patch_id_from_path(fpath: str) -> int:
    ext = ".feather"
    if not fpath.endswith(ext):
        raise NotAPatchFileError("input must be a .feather file")
    prefix, patch_id = fpath[:-len(ext)].rsplit("_", 1)
    return int(patch_id)


class NotAPatchFileError(Exception):
    pass


class PatchCatalog:

    id = 0
    cachefile = None
    _data = DataFrame()
    _len = 0
    _total = None
    _has_z = False
    _has_weights = False
    _center = None
    _radius = None

    def __init__(
        self,
        id: int,
        data: DataFrame,
        cachefile: str | None = None,
        center: NDArray[np.float_] | None = None,
        radius: float | None = None
    ) -> None:
        self.id = id
        if "ra" not in data:
            raise KeyError("right ascension column ('ra') is required")
        if "dec" not in data:
            raise KeyError("declination column ('dec') is required")
        if not set(data.columns) <= set(["ra", "dec", "redshift", "weights"]):
            raise KeyError(
                "'data' contains unidentified columns, optional columns are "
                "restricted to 'redshift' and 'weights'")
        # if there is a file path, store the file
        if cachefile is not None:
            self.cachefile = cachefile
            data.to_feather(cachefile)
        self._data = data
        self._init(center, radius)

    def _init(
        self,
        center: NDArray[np.float_] | None = None,
        radius: float | None = None
    ) -> None:
        self._len = len(self._data)
        self._has_z = "redshift" in self._data
        self._has_weights = "weights" in self._data
        if self.has_weights():
            self._total = float(self.weights.sum())
        else:
            self._total = float(len(self))

        # precompute (estimate) the patch center and size since it is quite fast
        # and the data is still loaded
        if center is None or radius is None:
            SUBSET_SIZE = 1000  # seems a reasonable, fast but not too sparse
            if self._len < SUBSET_SIZE:
                pos = self.pos
            else:
                which = np.random.randint(0, self._len, size=SUBSET_SIZE)
                pos = self.pos[which]
            xyz = position_sky2sphere(pos)

        if center is None:
            # compute mean coordinate, which will not be located on unit sphere
            mean_xyz = np.mean(xyz, axis=0)
            # map back onto unit sphere
            mean_sky = position_sphere2sky(mean_xyz)
            self._center = position_sky2sphere(mean_sky)
        else:
            if len(center) == 2:
                self._center = position_sky2sphere(center)
            elif len(center) == 3:
                self._center = center
            else:
                raise ValueError("'center' must be length 2 or 3")

        if center is None or radius is None:  # new center requires recomputing
            # compute maximum distance to any of the data points
            radius_xyz = np.sqrt(np.sum((xyz - self.center)**2, axis=1)).max()
            radius = distance_sphere2sky(radius_xyz)
        self._radius = radius

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, loaded={self.is_loaded()})"
        return s

    def __len__(self) -> int:
        return self._len

    @classmethod
    def from_cached(
        cls,
        cachefile: str,
        center: NDArray[np.float_] | None = None,
        radius: float | None = None
    ) -> PatchCatalog:
        # create the data instance
        new = cls.__new__(cls)
        new.id = patch_id_from_path(cachefile)
        new.cachefile = cachefile
        try:
            new._data = pd.read_feather(cachefile)
        except Exception as e:
            args = ()
            if hasattr(e, "args"):
                args = e.args
            raise NotAPatchFileError(*args)
        new._init(center, radius)
        return new

    def is_loaded(self) -> bool:
        return self._data is not None

    def require_loaded(self) -> None:
        if not self.is_loaded():
            raise AttributeError("data is not loaded")

    def load(self, use_threads: bool = True) -> None:
        if not self.is_loaded():
            if self.cachefile is None:
                raise ValueError("no datapath provided to load the data")
            self._data = pd.read_feather(
                self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        self._data = None
        gc.collect()

    def has_redshift(self) -> bool:
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
    def redshift(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_redshift():
            return self._data["redshift"].to_numpy()
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_weights():
            return self._data["weights"].to_numpy()
        else:
            return None

    @property
    def total(self) -> float:
        return self._total

    @property
    def center(self) -> float:
        return self._center

    @property
    def radius(self) -> float:
        return self._radius

    def iter_bins(
        self,
        z_bins: NDArray[np.float_],
        allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, PatchCatalog]]:
        if not allow_no_redshift and not self.has_redshift():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(
                    pd.cut(self.redshift, z_bins)):
                yield intv, PatchCatalog(self.id, bin_data)

    def get_tree(self, **kwargs) -> SphericalKDTree:
        tree = SphericalKDTree(self.ra, self.dec, self.weights, **kwargs)
        tree._total = self.total  # no need to recompute this
        return tree


class PatchCollection(Sequence):

    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: NDArray[np.float_] | None = None,
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
                patch_data = patch_data.drop("patch", axis=1)
                patch_data.rename(columns=renames, inplace=True)
                patch_data.reset_index(drop=True, inplace=True)
                # look up the center of the patch if given
                kwargs = dict(center=centers.get(patch_id))
                if unload:
                    # data will be written as feather file and loaded on demand
                    kwargs["cachefile"] = os.path.join(
                        cache_directory, f"patch_{patch_id:.0f}.feather")
                patch = PatchCatalog(int(patch_id), patch_data, **kwargs)
                limits.update(patch.redshift)
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
    def from_file(
        cls,
        filepath: str,
        patches: int | PatchCollection | str,
        ra: str,
        dec: str,
        *,
        redshift: str | None = None,
        weight: str | None = None,
        sparse: int | None = None,
        cache_directory: str | None = None,
        file_ext: str | None = None,
        **kwargs
    ) -> PatchCollection:
        columns = [c for c in [ra, dec, redshift, weight] if c is not None]
        if isinstance(patches, str):
            columns.append(patches)
            patch_kwarg = dict(patch_name=patches)
        elif isinstance(patches, PatchCollection):
            patch_kwarg = dict(
                patch_centers=position_sphere2sky(patches.centers))
        elif isinstance(patches, int):
            patch_kwarg = dict(n_patches=patches)
        else:
            raise TypeError(
                "'patches' must be either of type 'int', 'str', or "
                "'PatchCollection'")
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
            data = data[::sparse]
        return cls(
            data, ra, dec, **patch_kwarg,
            redshift_name=redshift,
            weight_name=weight,
            cache_directory=cache_directory)

    @classmethod
    def from_cache(
        cls,
        cache_directory: str
    ) -> PatchCollection:
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
            patch_id = patch_id_from_path(path)
            patch = PatchCatalog.from_cached(
                os.path.join(cache_directory, path),
                center=centers.get(patch_id),
                radius=radii.get(patch_id))
            limits.update(patch.redshift)
            patch.unload()
            new._patches[patch.id] = patch
        new._zmin, new._zmax = limits.get()
        return new

    def __repr__(self):
        length = len(self)
        loaded = sum(1 for patch in iter(self) if patch.is_loaded())
        s = self.__class__.__name__
        s += f"(patches={length}, loaded={loaded}/{length})"
        return s

    def __contains__(self, item: int) -> bool:
        return item in self._patches

    def __getitem__(self, item: int) -> PatchCatalog:
        return self._patches[item]

    def __len__(self) -> int:
        return len(self._patches)

    @property
    def ids(self) -> list[int]:
        return sorted(self._patches.keys())

    def __iter__(self) -> Iterator[PatchCatalog]:
        for patch_id in self.ids:
            yield self._patches[patch_id]

    def iter_loaded(self) -> Iterator[PatchCatalog]:
        for patch in iter(self):
            loaded = patch.is_loaded()
            patch.load()
            yield patch
            if not loaded:
                patch.unload()

    def is_loaded(self) -> bool:
        return all([patch.is_loaded() for patch in iter(self)])

    def load(self) -> None:
        for patch in iter(self):
            patch.load()

    def unload(self) -> None:
        for patch in iter(self):
            patch.unload()

    def n_patches(self) -> int:
        # seems ok to drop the last patch if that is empty and therefore missing
        return max(self._patches.keys()) + 1

    def has_redshift(self) -> bool:
        return all(patch.has_redshift() for patch in iter(self))  # always equal

    def has_weights(self) -> bool:
        return all(patch.has_weights() for patch in iter(self))  # always equal

    def get_min_redshift(self) -> float:
        return self._zmin

    def get_max_redshift(self) -> float:
        return self._zmax

    @property
    def ra(self) -> NDArray[np.float_]:
        return np.concatenate([patch.ra for patch in self.iter_loaded()])

    @property
    def dec(self) -> NDArray[np.float_]:
        return np.concatenate([patch.dec for patch in self.iter_loaded()])

    @property
    def redshift(self) -> NDArray[np.float_]:
        if self.has_redshift():
            return np.concatenate([
                patch.redshift for patch in self.iter_loaded()])
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

    @property
    def totals(self) -> NDArray[np.float_]:
        return np.array([patch.total for patch in iter(self)])

    @property
    def total(self) -> float:
        return self.totals.sum()

    @property
    def centers(self) -> NDArray[np.float_]:
        return np.array([patch.center for patch in iter(self)])

    @property
    def radii(self) -> NDArray[np.float_]:
        return np.array([patch.radius for patch in iter(self)])

    def get_linkage(
        self,
        max_query_radius_deg: float
    ) -> PatchLinkage:
        centers = self.centers  # in RA / Dec
        radii = self.radii  # in degrees, maximum distance measured from center
        # compute distance in degrees between all patch centers
        dist_deg = distance_sphere2sky(distance_matrix(centers, centers))
        # compare minimum separation required for patchs to not overlap
        min_sep_deg = np.add.outer(radii, radii)
        # check which patches overlap when factoring in the query radius
        overlaps = (dist_deg - max_query_radius_deg) < min_sep_deg
        patch_pairs = []
        for id1, overlap in enumerate(overlaps):
            patch_pairs.extend((id1, id2) for id2 in np.where(overlap)[0])
        return PatchLinkage(patch_pairs)


class PatchLinkage:

    def __init__(
        self,
        patch_tuples: list[TypePatchKey]
    ) -> None:
        self.pairs = patch_tuples

    def __len__(self) -> int:
        return len(self.pairs)

    def get_pairs(
        self,
        auto: bool,
        crosspatch: bool = True
    ) -> list[TypePatchKey]:
        if crosspatch:
            if auto:
                pairs = [(i, j) for i, j in self.pairs if j >= i]
            else:
                pairs = self.pairs
        else:
            pairs = [(i, j) for i, j in self.pairs if i == j]
        return pairs

    @staticmethod
    def _parse_collections(
        collection1: PatchCollection,
        collection2: PatchCollection | None = None
    ) -> tuple[bool, PatchCollection, PatchCollection, int]:
        auto = collection2 is None
        if auto:
            collection2 = collection1
        n_patches = max(collection1.n_patches(), collection2.n_patches())
        return auto, collection1, collection2, n_patches

    def get_matrix(
        self,
        collection1: PatchCollection,
        collection2: PatchCollection | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2, n_patches = self._parse_collections(
            collection1, collection2)
        pairs = self.get_pairs(auto, crosspatch)
        # make a boolean matrix indicating the exisiting patch combinations
        matrix = np.zeros((n_patches, n_patches), dtype=np.bool_)
        for pair in pairs:
            matrix[pair] = True
        return matrix

    def get_mask(
        self,
        collection1: PatchCollection,
        collection2: PatchCollection | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2, n_patches = self._parse_collections(
            collection1, collection2)
        # make a boolean mask indicating all patch combinations
        shape = (n_patches, n_patches)
        if crosspatch:
            mask = np.ones(shape, dtype=np.bool_)
            if auto:
                mask = np.triu(mask)
        else:
            mask = np.eye(n_patches, dtype=np.bool_)
        return mask

    def get_weight_matrix(
        self,
        collection1: PatchCollection,
        collection2: PatchCollection | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.float_]:
        auto, collection1, collection2, n_patches = self._parse_collections(
            collection1, collection2)
        # compute the product of the total weight per patch
        totals1 = np.zeros(n_patches)
        for i, total in zip(collection1.ids, collection1.totals):
            totals1[i] = total
        totals2 = np.zeros(n_patches)
        for i, total in zip(collection2.ids, collection2.totals):
            totals2[i] = total
        totals = np.multiply.outer(totals1, totals2)
        if auto:
            totals = np.triu(totals)  # (i, j) with i > j => 0
            totals[np.diag_indices(len(totals))] *= 0.5  # avoid double-counting
        return totals

    def get_patches(
        self,
        collection1: PatchCollection,
        collection2: PatchCollection | None = None,
        crosspatch: bool = True
    ) -> tuple[list[PatchCollection], list[PatchCollection]]:
        auto, collection1, collection2, n_patches = self._parse_collections(
            collection1, collection2)
        pairs = self.get_pairs(auto, crosspatch)
        # generate the patch lists
        patches1 = []
        patches2 = []
        for id1, id2 in pairs:
            patches1.append(collection1[id1])
            patches2.append(collection2[id2])
        return patches1, patches2
