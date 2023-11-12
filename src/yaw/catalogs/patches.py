from __future__ import annotations

import gc
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow
from scipy.cluster import vq

from yaw.catalogs.kdtree import SphericalKDTree
from yaw.core.coordinates import Coord3D, Coordinate, CoordSky, Distance, DistSky

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray
    from pandas import DataFrame, Interval

__all__ = ["PatchCatalog"]


class NotAPatchFileError(Exception):
    pass


class CachingError(Exception):
    pass


def check_columns(data: DataFrame, extra: Iterable) -> None:
    required = {"ra", "dec"}
    optional = {"weight", "redshift"} | set(extra)
    existing = set(data.columns)
    for col in required - existing:
        raise KeyError("column 'ra' is required but missing")
    for col in existing - (required | optional):
        opt = ", ".join(optional)
        raise KeyError(f"unidentified column '{col}', allowed optionals are: {opt}")


def patch_id_from_path(fpath: str) -> int:
    """Extract the patch ID from the file name in the cache directory"""
    ext = ".feather"
    if not fpath.endswith(ext):
        raise NotAPatchFileError("input must be a .feather file")
    prefix, patch_id = fpath[: -len(ext)].rsplit("_", 1)
    return int(patch_id)


@dataclass
class PatchMeta:
    id: int
    zmin: float
    zmax: float
    length: int
    total: float
    has_z: bool
    has_w: bool
    center: Coordinate
    radius: Distance

    @classmethod
    def from_patch(cls, patch: PatchCatalog) -> PatchMeta:
        return cls(
            id=patch.id,
            zmin=patch.redshift.min(),
            zmax=patch.redshift.max(),
            center=patch.center,
            radius=patch.radius,
            length=len(patch),
            total=patch.total,
            has_z=patch.has_redshift(),
            has_w=patch.has_weight(),
        )

    def asdict(self) -> dict:
        meta = {k: v for k, v in asdict(self).items()}
        del meta["center"]
        del meta["radius"]
        center = self.center.to_3d()
        meta["x"] = center.x
        meta["y"] = center.y
        meta["z"] = center.z
        meta["r"] = self.radius.values
        return meta


class PatchCatalog:
    """Represents a single spatial patch of a :obj:`Catalog`.

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

    def __init__(
        self,
        id: int,
        data: DataFrame,
        cachefile: str | None = None,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> None:
        """Create a new patch from a data frame.

        Coordiantes are converted to radian. If a cache path is provided, a
        cache file is created and the data is dropped from memory.

        Args:
            id (:obj:`int`):
                Unique index of the patch.
            data (:obj:`pandas.DataFrame`):
                Data frame with columns ``ra`` (right ascension), ``dec``
                (declination) in radians and optionally ``weight``,
                ``redshift`` if either data is available.
            cachefile (:obj:`str`, optional):
                If provided, the data is cached as .feather file at this path.
            center (:obj:`yaw.core.coordiante.Coordiante`, optional):
                Center coordinates of the patch. Computed automatically if not
                provided.
            radius (:obj:`yaw.core.coordiante.Distance`, optional):
                The angular size of the patch in radians. Computed automatically
                if not provided.
        """
        self.id = id
        check_columns(data)
        self._data = data
        # if there is a file path, store the file
        if cachefile is not None:
            self.cachefile = cachefile
            self._data.to_feather(cachefile)
        # compute all metadata
        self._precompute_metadata(center, radius)

    def _precompute_metadata(self, center, radius) -> None:
        self._length = len(self._data)
        self._has_z = "redshift" in self._data
        try:
            self._total = float(self._data["weight"].sum())
            self._has_w = True
        except KeyError:
            self._total = len(self._data)
            self._has_w = False
        self._compute_center_radius(center, radius)

    def _compute_center_radius(
        self, center: Coordinate | None = None, radius: Distance | None = None
    ) -> None:
        # use provided values
        if center is not None and radius is not None:
            return center.to_3d(), radius.to_sky()

        # load a subset of the data to compute the center and/or radius
        self.load()
        SUBSET_SIZE = 1000  # seems a reasonable, fast but not too sparse
        if self._length < SUBSET_SIZE:
            positions = self.pos
        else:
            rng = np.random.default_rng(seed=12345)
            which = rng.integers(0, self._length, size=SUBSET_SIZE)
            positions = self.pos[which]
        positions = positions.to_3d()

        if center is None:
            center = positions.mean()
            self._center = center.to_sky().to_3d()  # ensure to be on unit sphere
        if radius is None:
            maxdist = positions.distance(self._center).max()
            self._radius = maxdist.to_sky()
        return center.to_3d(), radius.to_sky()

    @classmethod
    def from_cached(
        cls,
        cachefile: str,
        metadata: PatchMeta | None = None,
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
        # check the input file
        id = patch_id_from_path(cachefile)

        if metadata is None:
            try:
                data = pd.read_feather(cachefile)
            except Exception as e:
                raise NotAPatchFileError(cachefile) from e
            return cls(id, data, cachefile)

        else:
            try:
                pyarrow.ipc.open_file(cachefile)
            except Exception as e:
                raise NotAPatchFileError(cachefile) from e
            # create the patch without load the actual data
            new = cls.__new__(cls)
            new.id = id
            new.cachefile = cachefile
            new._data = None
            new._length = metadata.length
            new._total = metadata.total
            new._has_z = metadata.has_z
            new._has_w = metadata.has_w
            new._center = metadata.center.to_3d()
            new._radius = metadata.radius.to_sky()
            return new

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, loaded={self.is_loaded()})"
        return s

    def __len__(self) -> int:
        return self._length

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
                raise CachingError("there is no cached data")
            self._data = pd.read_feather(self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        """Drop the data from memory.

        Raises a :obj:`CachingError` if no cache file is sepcified."""
        if self.cachefile is None:
            raise CachingError("cachepath not set")
        self._data = None
        gc.collect()

    def has_redshift(self) -> bool:
        """Whether the patch data include redshifts."""
        return self._has_z

    def has_weight(self) -> bool:
        """Whether the patch data include weights."""
        return self._has_w

    @property
    def data(self) -> DataFrame:
        """Direct access to the underlying :obj:`pandas.DataFrame` which holds
        the patch data.

        Raises a :obj:`CachingError` if data is not loaded."""
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
    def redshift(self) -> NDArray[np.float_]:
        """Get the redshifts as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        if self.has_redshift():
            return self._data["redshift"].to_numpy()
        else:
            return None

    @property
    def weight(self) -> NDArray[np.float_]:
        """Get the object weights as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        self.require_loaded()
        if self.has_weight():
            return self._data["weight"].to_numpy()
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
        if not allow_no_redshift and not self.has_redshift():
            raise ValueError("no redshifts for iteration provdided")
        if allow_no_redshift:
            for intv in pd.IntervalIndex.from_breaks(z_bins, closed="left"):
                yield intv, self
        else:
            for intv, bin_data in self._data.groupby(pd.cut(self.redshift, z_bins)):
                yield intv, PatchCatalog(
                    self.id,
                    bin_data,
                    degrees=False,
                    center=self._center,
                    radius=self._radius,
                )

    def get_tree(self, **kwargs) -> SphericalKDTree:
        """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
        tree = SphericalKDTree(self.pos, self.weight, **kwargs)
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
