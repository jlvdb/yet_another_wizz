from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from yaw.core.abc import PatchedQuantity
from yaw.core.containers import PatchIDs
from yaw.core.coordinates import Dist3D, DistSky
from yaw.core.cosmology import r_kpc_to_angle

if TYPE_CHECKING:  # pragma: no cover
    from yaw.catalogs import BaseCatalog
    from yaw.config import Config


logger = logging.getLogger(__name__)


LINK_ZMIN = 0.05


class PatchLinkage(PatchedQuantity):

    def __init__(
        self,
        patch_tuples: list[PatchIDs]
    ) -> None:
        self.pairs = patch_tuples

    @classmethod
    def from_setup(
        cls,
        config: Config,
        catalog: BaseCatalog
    ) -> PatchLinkage:
        # determine the additional overlap from the spatial query
        if config.backend.crosspatch:
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = max(LINK_ZMIN, config.binning.zmin)
            max_query_radius = r_kpc_to_angle(
                config.scales.as_array(), z_ref, config.cosmology).max()
        else:
            max_query_radius = 0.0  # only relevant for cross-patch
        max_query_radius = DistSky(max_query_radius)

        logger.debug(f"computing patch linkage with {max_query_radius=:.3e}")
        centers_3d = catalog.centers.to_3d().values
        radii = catalog.radii.values
        # compute distance between all patch centers
        dist_mat_3d = Dist3D(distance_matrix(centers_3d, centers_3d))
        # compare minimum separation required for patchs to not overlap
        size_sum = DistSky(np.add.outer(radii, radii))

        # check which patches overlap when factoring in the query radius
        overlaps = dist_mat_3d.to_sky() < (size_sum + max_query_radius)
        patch_pairs = []
        for id1, overlap in enumerate(overlaps):
            patch_pairs.extend((id1, id2) for id2 in np.where(overlap)[0])
        logger.debug(
            f"found {len(patch_pairs)} patch links "
            f"for {catalog.n_patches} patches")
        return cls(patch_pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(n_patches={self.n_patches}, n_pairs={len(self)})"

    @property
    def n_patches(self) -> int:
        patches = set()
        for p1, p2 in self.pairs:
            patches.add(p1)
            patches.add(p2)
        return len(patches)

    @property
    def density(self) -> float:
        n = self.n_patches
        return len(self) / (n*n)

    def get_pairs(
        self,
        auto: bool,
        crosspatch: bool = True
    ) -> list[PatchIDs]:
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
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None
    ) -> tuple[bool, BaseCatalog, BaseCatalog]:
        auto = collection2 is None
        if auto:
            collection2 = collection1
        return auto, collection1, collection2

    def get_matrix(
        self,
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        pairs = self.get_pairs(auto, crosspatch)
        # make a boolean matrix indicating the exisiting patch combinations
        n_patches = self.n_patches
        matrix = np.zeros((n_patches, n_patches), dtype=np.bool_)
        for pair in pairs:
            matrix[pair] = True
        return matrix

    def get_mask(
        self,
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        # make a boolean mask indicating all patch combinations
        n_patches = self.n_patches
        shape = (n_patches, n_patches)
        if crosspatch:
            mask = np.ones(shape, dtype=np.bool_)
            if auto:
                mask = np.triu(mask)
        else:
            mask = np.eye(n_patches, dtype=np.bool_)
        return mask

    def get_patches(
        self,
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None,
        crosspatch: bool = True
    ) -> tuple[list[BaseCatalog], list[BaseCatalog]]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        pairs = self.get_pairs(auto, crosspatch)
        # generate the patch lists
        patches1 = []
        patches2 = []
        for id1, id2 in pairs:
            patches1.append(collection1[id1])
            patches2.append(collection2[id2])
        return patches1, patches2
