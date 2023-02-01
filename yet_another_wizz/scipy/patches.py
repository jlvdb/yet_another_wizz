from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster import vq

from yet_another_wizz.core.coordinates import (
    distance_sphere2sky, position_sky2sphere, position_sphere2sky)

from yet_another_wizz.scipy.kdtree import SphericalKDTree

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval


class NotAPatchFileError(Exception):
    pass


def patch_id_from_path(fpath: str) -> int:
    ext = ".feather"
    if not fpath.endswith(ext):
        raise NotAPatchFileError("input must be a .feather file")
    prefix, patch_id = fpath[:-len(ext)].rsplit("_", 1)
    return int(patch_id)


class PatchCatalog:

    id = 0
    cachefile = None
    _data = pd.DataFrame()
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
        radius: float | None = None,
        degrees: bool = True
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
        if degrees:
            self._data["ra"] = np.deg2rad(self._data["ra"])
            self._data["dec"] = np.deg2rad(self._data["dec"])
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
            self._total = len(self)

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
            raise NotAPatchFileError(*args) from e
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

    def has_redshifts(self) -> bool:
        return self._has_z

    def has_weights(self) -> bool:
        return self._has_weights

    @property
    def data(self) -> DataFrame:
        self.require_loaded()
        return self._data

    @property
    def ra(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["ra"].to_numpy()

    @property
    def dec(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data["dec"].to_numpy()

    @property
    def pos(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data[["ra", "dec"]].to_numpy()

    @property
    def redshifts(self) -> NDArray[np.float_]:
        self.require_loaded()
        if self.has_redshifts():
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
        if not allow_no_redshift and not self.has_redshifts():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in pd.IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(
                    pd.cut(self.redshifts, z_bins)):
                yield intv, PatchCatalog(
                    self.id, bin_data, degrees=False,
                    center=self._center, radius=self._radius)

    def get_tree(self, **kwargs) -> SphericalKDTree:
        tree = SphericalKDTree(self.ra, self.dec, self.weights, **kwargs)
        tree._total = self.total  # no need to recompute this
        return tree


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
