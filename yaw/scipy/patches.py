from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster import vq

from yaw.core.coordinates import Coordinate, Coord3D, CoordSky, Dist3D, DistSky

from yaw.scipy.kdtree import SphericalKDTree

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval


class NotAPatchFileError(Exception):
    pass


class CachingError(Exception):
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
        center: Coordinate | None = None,
        radius: DistSky | float | None = None,
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
        # next line is crucial, otherwise lines below modify data inplace
        self._data = data.copy()
        if degrees:
            self._data["ra"] = np.deg2rad(data["ra"])
            self._data["dec"] = np.deg2rad(data["dec"])
        # if there is a file path, store the file
        if cachefile is not None:
            self.cachefile = cachefile
            self._data.to_feather(cachefile)
        self._init(center, radius)

    def _init(
        self,
        center: Coordinate | None = None,
        radius: DistSky | float | None = None
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
                positions = self.pos.to_3d()
            else:
                rng = np.random.default_rng(seed=12345)
                which = rng.integers(0, self._len, size=SUBSET_SIZE)
                positions = self.pos[which].to_3d()

        if center is None:
            self._center = positions.mean()
        else:
            self._center = center.to_3d()

        if center is None or radius is None:  # new center requires recomputing
            # compute maximum distance to any of the data points
            radius_3d = positions.distance(self._center).values.max()
            self._radius = Dist3D(radius_3d).to_sky()
        elif isinstance(radius, DistSky):
            self._radius = radius
        else:
            self._radius = DistSky(radius)

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
        center: Coordinate | None = None,
        radius: DistSky | float | None = None
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
            raise CachingError("data is not loaded")

    def load(self, use_threads: bool = True) -> None:
        if not self.is_loaded():
            if self.cachefile is None:
                raise CachingError("no datapath provided to load the data")
            self._data = pd.read_feather(
                self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        if self.cachefile is None:
            raise CachingError("no datapath provided to unload the data")
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
    def pos(self) -> CoordSky:
        self.require_loaded()
        return CoordSky(self.ra, self.dec)

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
    def center(self) -> CoordSky:
        return self._center.to_sky()

    @property
    def radius(self) -> DistSky:
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
        tree = SphericalKDTree(self.pos, self.weights, **kwargs)
        tree._total = self.total  # no need to recompute this
        return tree


# Determine patch centers with k-means clustering. The implementation in
# treecorr is quite good, but might not be available. Implement a fallback using
# the scipy.cluster module.
try:
    import treecorr

    def treecorr_patches(
        position: Coordinate,
        n_patches: int,
        **kwargs
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        position = position.to_sky()
        n_points = len(position.ra)
        cat = treecorr.Catalog(
            ra=position.ra, ra_units="radians",
            dec=position.dec, dec_units="radians",
            npatch=n_patches)
        xyz = np.atleast_2d(cat.patch_centers)
        centers = Coord3D(xyz[:, 0], xyz[:, 1], xyz[:, 2])
        if cat.patch is None:
            patches = np.zeros(len(n_points), dtype=np.int_)
        else:
            patches = np.copy(cat.patch)
        del cat  # might not be necessary
        return centers, patches

    create_patches = treecorr_patches

except ImportError:

    def scipy_patches(
        position: Coordinate,
        n_patches: int,
        n_max: int = 500_000
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        position = position.to_3d()
        n_points = len(position.x)
        subset = np.random.randint(0, n_points, size=min(n_max, n_points))
        # place on unit sphere to avoid coordinate distortions
        centers, _ = vq.kmeans2(
            position[subset].values, n_patches, minit="points")
        centers = Coord3D(*centers.T)
        patches = assign_patches(centers=centers, position=position)
        return centers, patches

    create_patches = scipy_patches


def assign_patches(
    centers: Coordinate,
    position: Coordinate
) -> NDArray[np.int_]:
    patches, _ = vq.vq(position.to_3d().values, centers.to_3d().values)
    return patches
