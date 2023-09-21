from __future__ import annotations

import pickle
from typing import TYPE_CHECKING

import numpy as np
from scipy.cluster import vq

from yaw.catalogs.patches import PatchBase
from yaw.catalogs.scipy.kdtree import SphericalKDTree
from yaw.core.coordinates import Coord3D, Coordinate, Distance

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

__all__ = ["ScipyPatch"]


class NotAPatchFileError(Exception):
    pass


def patch_id_from_path(fpath: str) -> int:
    """Extract the patch ID from the file name in the cache directory"""
    ext = ".feather"
    if not fpath.endswith(ext):
        raise NotAPatchFileError("input must be a .feather file")
    prefix, patch_id = fpath[: -len(ext)].rsplit("_", 1)
    return int(patch_id)


class ScipyPatch(PatchBase):
    """Represents a single spatial patch of a :obj:`ScipyCatalog`.

    A patch holds the data from single patch of the catalogue and provides
    method to access this data. Furthermore, it implements the caching to and
    restoring from disk. Data is temporarily saved to a .feather file, making
    it easy to pass patches to new threads and processes, which can, load the
    data back into memory if necessary.
    """

    _data: SphericalKDTree = None

    def _init(
        self, data: dict[str, NDArray[np.float_] | None], cachefile: str | None = None
    ) -> None:
        tree = SphericalKDTree(
            self.pos, weights=data["weights"], redshifts=data["redshifts"]
        )
        tree._total = self.total  # no need to recompute this
        self._data = tree

        if cachefile is not None:
            with open(cachefile, "wb") as f:
                pickle.dump(self, f)

    @classmethod
    def from_cached(
        cls,
        cachefile: str,
        center: Coordinate | None = None,
        radius: Distance | None = None,
    ) -> ScipyPatch:
        with open(cachefile, "rb") as f:
            patch = pickle.load(f)
        return patch

    def load(self, use_threads: bool = True) -> None:
        patch = self.from_cached(self.cachefile)
        self._data = patch._data

    @property
    def ra(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data.ra

    @property
    def dec(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data.dec

    @property
    def redshifts(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data.redshifts

    @property
    def weights(self) -> NDArray[np.float_]:
        self.require_loaded()
        return self._data.weights

    def get_tree(self) -> SphericalKDTree:
        """TODO"""
        return self._data


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
