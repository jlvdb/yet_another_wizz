from __future__ import annotations

import gc
import os
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
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


def check_columns(data: DataFrame, optional: Iterable) -> None:
    required = {"ra", "dec"}
    optional = {"weight", "redshift"} | set(optional)
    existing = set(data.columns)
    for col in required - existing:
        raise KeyError("column 'ra' is required but missing")
    for col in existing - (required | optional):
        opt = ", ".join(optional)
        raise KeyError(f"unidentified column '{col}', allowed optionals are: {opt}")


def check_cache_file(path: str) -> None:
    try:
        pyarrow.ipc.open_file(path)
    except Exception as e:
        raise NotAPatchFileError(path) from e


def patch_id_from_path(path: str) -> int:
    """Extract the patch ID from the file name in the cache directory"""
    prefix, _ = os.path.splitext(path)
    prefix, patch_id = prefix.rsplit("_", 1)
    return int(patch_id)


def compute_zlims(data: DataFrame) -> tuple[float | None, float | None]:
    try:
        redshifts = data["redshift"]
        zmin = float(redshifts.min())
        zmax = float(redshifts.max())
    except KeyError:
        zmin, zmax = None, None
    return zmin, zmax


def compute_center_radius(
    data: DataFrame,
    center: Coordinate | None = None,
    radius: Distance | None = None,
    subset_size: int = 1000,
) -> tuple[Coord3D, DistSky]:
    # compute positions if needed
    if center is None or radius is None:
        if len(data) <= subset_size:
            ra_dec = data[["ra", "dec"]].to_numpy()
        else:
            rng = np.random.default_rng(seed=12345)
            take = rng.integers(0, len(data), size=subset_size)
            ra_dec = data[["ra", "dec"]][take].to_numpy()
        positions = CoordSky.from_array(ra_dec).to_3d()

    # compute the patch center
    if center is not None:
        center = center.to_3d()
    else:
        center = positions.mean()

    # compute the patch radius given patch center
    if radius is not None:
        radius = radius.to_sky()
    else:
        maxdist = positions.distance(center).max()
        radius = maxdist.to_sky()
    return center, radius


@dataclass
class PatchMeta:
    length: int
    total: float
    has_w: bool
    has_z: bool
    zmin: float | None
    zmax: float | None
    center: Coordinate
    radius: Distance

    @classmethod
    def build(
        cls,
        data: DataFrame,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> PatchMeta:
        length = len(data)
        # weight related metadata
        has_w = "weight" in data
        total = data["weight"].sum() if has_w else length
        has_z = "redshift" in data
        # redshift related metadata
        zmin, zmax = compute_zlims(data)
        # position related metadata
        center, radius = compute_center_radius(data, center, radius)
        # pack results and ensure native python types where possible
        return cls(
            length=length,
            total=float(total),
            has_w=has_w,
            has_z=has_z,
            zmin=zmin,
            zmax=zmax,
            center=center,
            radius=radius,
        )

    def asdict(self) -> dict:
        metadict = asdict(self)
        # replace center with x/y/z floats
        del metadict["center"]
        center = self.center.to_3d()
        metadict["x"] = float(center.x)
        metadict["y"] = float(center.y)
        metadict["z"] = center.z
        # convert radius to float
        metadict["radius"] = float(self.radius.values)
        return metadict

    def recompute_subset(
        self,
        data: DataFrame,
        zmin: float | None = None,
        zmax: float | None = None,
    ) -> PatchMeta:
        length = len(data)
        total = data["weight"].sum() if self.has_w else length
        zmin, zmax = compute_zlims(data)
        return self.__class__(
            length=length,
            total=float(total),
            has_w=self.has_w,
            has_z=self.has_z,
            zmin=zmin,
            zmax=zmax,
            center=self.center,
            radius=self.radius,
        )


@dataclass
class PatchCatalog:
    """Represents a single spatial patch of a :obj:`Catalog`.

    A patch holds the data from single patch of the catalogue and provides
    method to access this data. Furthermore, it implements the caching to and
    restoring from disk. Data is temporarily saved to a .feather file, making
    it easy to pass patches to new threads and processes, which can, load the
    data back into memory if necessary.
    """

    id: int
    """Unique index of the patch."""
    data: DataFrame | None
    metadata: PatchMeta | None = field(default=None)
    cachefile: str | None = field(default=None)
    """The patch to the cached .feather data file if caching is enabled."""

    def __postinit__(self) -> None:
        if self.data is None and self.cachefile is None:
            raise ValueError("either 'data' or 'cachefile' must be provided")

        # check the initial state
        data_in_memory = self.data is not None
        data_on_disk = not data_in_memory
        if data_in_memory:
            check_columns(self.data)

        # compute the meta data if needed, this requires loading the data
        if self.metadata is None:
            if data_on_disk:
                self.load()
            self.metadata = PatchMeta.build(self.data)

        # check that cached data is valid feather file if it was not loaded yet
        elif data_on_disk:
            check_cache_file(self.cachefile)

        # if there is a cachfile, store the data
        if not data_on_disk and self.cachefile is not None:
            self.unload()

    @classmethod
    def from_cached(
        cls,
        cachefile: str,
        metadata: PatchMeta | None = None,
    ) -> PatchCatalog:
        id = patch_id_from_path(cachefile)
        return cls(id=id, data=None, metadata=metadata, cachefile=cachefile)

    def __repr__(self) -> str:
        s = self.__class__.__name__
        s += f"(id={self.id}, length={len(self)}, loaded={self.is_loaded()})"
        return s

    def __len__(self) -> int:
        return self.metadata.length

    def is_loaded(self) -> bool:
        """Whether the data is present in memory"""
        return self.data is not None

    def load(self, use_threads: bool = True) -> None:
        """Load the data from the cache file into memory.

        Raises a :obj:`CachingError` if no cache file is sepcified."""
        if not self.is_loaded():
            if self.cachefile is None:
                raise CachingError("there is no cached data")
            self.data = pd.read_feather(self.cachefile, use_threads=use_threads)

    def unload(self) -> None:
        """Drop the data from memory.

        Raises a :obj:`CachingError` if no cache file is sepcified."""
        if self.cachefile is None:
            raise CachingError("cachepath not set")
        elif not os.path.exist(self.cachefile):
            self.data.to_feather(self.cachefile)
        self.data = None
        gc.collect()

    def has_redshift(self) -> bool:
        """Whether the patch data include redshifts."""
        return self.metadata.has_z

    def has_weight(self) -> bool:
        """Whether the patch data include weights."""
        return self.metadata.has_w

    def get_data(self) -> DataFrame:
        """Direct access to the underlying :obj:`pandas.DataFrame` which holds
        the patch data.

        Raises a :obj:`CachingError` if data is not loaded."""
        if self.is_loaded():
            return self.data
        else:
            return pd.read_feather(self.cachefile)

    def _get_columns(self, columns: Iterable[str]) -> dict[str, NDArray]:
        if self.data is None:
            data = pd.read_feather(self.cachefile, columns)
        else:
            data = self.data[columns]
        return {col: data[col].to_numpy() for col in columns}

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Raises a :obj:`CachingError` if data is not loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        data = self._get_columns(["ra", "dec"])
        return CoordSky(data["ra"], data["dec"])

    @property
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians.

        Raises a :obj:`CachingError` if data is not loaded."""
        return self._get_columns(["ra"])["ra"]

    @property
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians.

        Raises a :obj:`CachingError` if data is not loaded."""
        return self._get_columns(["dec"])["dec"]

    @property
    def redshift(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        if self.has_redshift():
            return self._get_columns(["redshift"])["redshift"]
        else:
            return None

    @property
    def weight(self) -> NDArray[np.float_] | None:
        """Get the object weights as array or ``None`` if not available.

        Raises a :obj:`CachingError` if data is not loaded."""
        if self.has_weight():
            return self._get_columns(["weight"])["weight"]
        else:
            return None

    @property
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available.

        Available even if no data is loaded."""
        return self.metadata.total

    @property
    def center(self) -> CoordSky:
        """Get the patch centers in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return self.metadata.center.to_sky()

    @property
    def radius(self) -> DistSky:
        """Get the patch size in radians.

        Available even if no data is loaded.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        return self.metadata.radius.to_sky()

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
            data = self.get_data()
            for intv, bin_data in self.data.groupby(pd.cut(data["redshift"], z_bins)):
                metadata = self.metadata.recompute_subset(self.id, bin_data)
                yield intv, PatchCatalog(
                    id=self.id,
                    data=bin_data,
                    metadata=metadata,
                )

    def get_tree(self, **kwargs) -> SphericalKDTree:
        """Build a :obj:`SphericalKDTree` from the patch data coordiantes."""
        pos_cols = ["ra", "dec"]
        # load only the strictly necessary data
        if self.has_weight():
            data = self._get_columns([*pos_cols, "weight"])
            weight = data["weight"].to_numpy()
        else:
            data = self._get_columns(pos_cols)
            weight = None
        position = CoordSky.from_array(data[pos_cols].to_numpy())
        tree = SphericalKDTree(position, weight, **kwargs)
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
