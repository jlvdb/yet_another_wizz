from __future__ import annotations

import gc
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.cluster import vq

from yaw.catalogs.scipy.kdtree import SphericalKDTree
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, Distance, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval

__all__ = ["PatchCatalog"]


class NotAPatchFileError(Exception):
    pass


class CachingError(Exception):
    pass


def patch_id_from_path(fpath: str) -> int:
    """Extract the patch ID from the file name in the cache directory"""
    ext = ".feather"
    if not fpath.endswith(ext):
        raise NotAPatchFileError("input must be a .feather file")
    prefix, patch_id = fpath[: -len(ext)].rsplit("_", 1)
    return int(patch_id)


class PatchCatalog:
    """Represents a single spatial patch of a :obj:`ScipyCatalog`.

    A patch holds the data from single patch of the catalogue and provides
    method to access this data. Furthermore, it implements the caching to and
    restoring from disk. Data is temporarily saved to a .feather file, making
    it easy to pass patches to new threads and processes, which can, load the
    data back into memory if necessary.
    """

    id = 0
    """Unique index of the patch."""
    cachefile = None
    """The patch to the cached .feather data file if caching is enabled."""
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
        radius: Distance | None = None,
        degrees: bool = True,
    ) -> None:
        """Create a new patch from a data frame.

        Coordiantes are converted to radian. If a cache path is provided, a
        cache file is created and the data is dropped from memory.

        Args:
            id (:obj:`int`):
                Unique index of the patch.
            data (:obj:`pandas.DataFrame`):
                Data frame with columns ``ra``, ``dec`` (by default assumed to
                be in degrees) and optionally ``weights``, ``redshift`` if
                either data is available.
            cachefile (:obj:`str`, optional):
                If provided, the data is cached as .feather file at this path.
            center (:obj:`yaw.core.coordiante.Coordiante`, optional):
                Center coordinates of the patch. Computed automatically if not
                provided.
            radius (:obj:`yaw.core.coordiante.Distance`, optional):
                The angular size of the patch. Computed automatically if not
                provided.
            degrees (:obj:`bool`):
                Whether the input coordinates ``ra``, ``dec`` are in degrees.
        """
        self.id = id
        if "ra" not in data:
            raise KeyError("right ascension column ('ra') is required")
        if "dec" not in data:
            raise KeyError("declination column ('dec') is required")
        if not set(data.columns) <= set(["ra", "dec", "redshift", "weights"]):
            raise KeyError(
                "'data' contains unidentified columns, optional columns are "
                "restricted to 'redshift' and 'weights'"
            )
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
        self, center: Coordinate | None = None, radius: Distance | None = None
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

        # store in xyz coordinates
        if center is None:
            self._center = positions.mean()
        else:
            self._center = center.to_3d()

        if center is None or radius is None:  # new center requires recomputing
            # compute maximum distance to any of the data points
            radius = positions.distance(self._center).max()

        # store radius in radians
        self._radius = radius.to_sky()

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
        radius: Distance | None = None,
    ) -> PatchCatalog:
        """Restore the patch instance from its cache file.

        Optionally, the center and radius of the patch can be provided to avoid
        recomputing these quantities.

        Args:
            cachefile (:obj:`str`):
                Path to the cach file (.feather)
            center (:obj:`yaw.core.coordiante.Coordiante`, optional):
                Center coordinates of the patch. Computed automatically if not
                provided.
            radius (:obj:`yaw.core.coordiante.Distance`, optional):
                The angular size of the patch. Computed automatically if not
                provided.
        """
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
        """Whether the data is present in memory"""
        return self._data is not None

    def require_loaded(self) -> None:
        """Raise a :obj:`CachingError` if the data is not present in memory."""
        if not self.is_loaded():
            raise CachingError("data is not loaded")

    def load(self, use_threads: bool = True) -> None:
        """Load the data from the cache file into memory.

        Raises a :obj:`CachingError` if no cache file is sepcified."""
        if not self.is_loaded():
            if self.cachefile is None:
                raise CachingError("no datapath provided to load the data")
            self._data = pd.read_feather(self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        """Drop the data from memory.

        Raises a :obj:`CachingError` if no cache file is sepcified."""
        if self.cachefile is None:
            raise CachingError("no datapath provided to unload the data")
        self._data = None
        gc.collect()

    def has_redshifts(self) -> bool:
        """Whether the patch data include redshifts."""
        return self._has_z

    def has_weights(self) -> bool:
        """Whether the patch data include weights."""
        return self._has_weights

    @property
    def data(self) -> DataFrame:
        """Direct access to the underlying :obj:`pandas.DataFrame` which holds
        the patch data."""
        self.require_loaded()
        return self._data

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        return self._data["ra"].to_numpy()

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        return self._data["dec"].to_numpy()

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Raises a :obj:`CachingError` if data is not loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        self.require_loaded()
        return CoordSky(self.ra, self.dec)

    @property
    def redshifts(self) -> NDArray[np.float_]:
        """Get the redshifts as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        if self.has_redshifts():
            return self._data["redshift"].to_numpy()
        else:
            return None

    @property
    def weights(self) -> NDArray[np.float_]:
        """Get the object weights as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        if self.has_weights():
            return self._data["weights"].to_numpy()
        else:
            return None

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available.

        Available even if no data is loaded."""
        return self._total

    @property
    def center(self) -> CoordSky:
        """Get the patch centers in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return self._center.to_sky()

    @property
    def radius(self) -> DistSky:
        """Get the patch size in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return self._radius

    def iter_bins(
        self, z_bins: NDArray[np.float_], allow_no_redshift: bool = False
    ) -> Iterator[tuple[Interval, PatchCatalog]]:
        """Iterate the patch in bins of redshift.

        Args:
            z_bins (:obj:`NDArray`):
                Edges of the redshift bins.
            allow_no_redshift (:obj:`bool`):
                If true and the data has no redshifts, the iterator yields the
                whole patch at each iteration step.

        Yields:
            (tuple): tuple containing:
                - **intv** (:obj:`pandas.Interval`): the selection for this bin.
                - **cat** (:obj:`PatchCatalog`): instance containing the data
                  for this bin.
        """
        if not allow_no_redshift and not self.has_redshifts():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in pd.IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(pd.cut(self.redshifts, z_bins)):
                yield intv, PatchCatalog(
                    self.id,
                    bin_data,
                    degrees=False,
                    center=self._center,
                    radius=self._radius,
                )

    def get_tree(self, **kwargs) -> SphericalKDTree:
        """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
        tree = SphericalKDTree(self.pos, self.weights, **kwargs)
        tree._total = self.total  # no need to recompute this
        return tree


# Determine patch centers with k-means clustering. The implementation in
# treecorr is quite good, but might not be available. Implement a fallback using
# the scipy.cluster module.


def assign_patches(centers: Coordinate, position: Coordinate) -> NDArray[np.int_]:
    """Assign objects based on their coordinate to a list of points based on
    proximit."""
    patches, dist = vq.vq(position.to_3d().values, centers.to_3d().values)
    return patches


try:
    import treecorr

    def treecorr_patches(
        position: Coordinate, n_patches: int, **kwargs
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        """Use the *k*-means clustering algorithm of :obj:`treecorr.Catalog` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_sky()
        cat = treecorr.Catalog(
            ra=position.ra,
            ra_units="radians",
            dec=position.dec,
            dec_units="radians",
            npatch=n_patches,
        )
        xyz = np.atleast_2d(cat.patch_centers)
        centers = Coord3D.from_array(xyz)
        if n_patches == 1:
            patches = np.zeros(len(position), dtype=np.int_)
        else:
            patches = assign_patches(centers=centers, position=position)
        del cat  # might not be necessary
        return centers, patches

    create_patches = treecorr_patches

except ImportError:

    def scipy_patches(
        position: Coordinate, n_patches: int, n_max: int = 500_000
    ) -> tuple[Coord3D, NDArray[np.int_]]:
        """Use the *k*-means clustering algorithm of :obj:`scipy.cluster` to
        generate spatial patches and assigning objects to those patches.
        """
        position = position.to_3d()
        subset = np.random.randint(0, len(position), size=min(n_max, len(position)))
        # place on unit sphere to avoid coordinate distortions
        centers, _ = vq.kmeans2(position[subset].values, n_patches, minit="points")
        centers = Coord3D.from_array(centers)
        patches = assign_patches(centers=centers, position=position)
        return centers, patches

    create_patches = scipy_patches
