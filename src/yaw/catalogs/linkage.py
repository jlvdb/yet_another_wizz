from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from yaw.core.containers import PatchIDs
from yaw.core.coordinates import Dist3D, DistSky
from yaw.core.cosmology import r_kpc_to_angle

if TYPE_CHECKING:  # pragma: no cover
    from yaw.catalogs import BaseCatalog
    from yaw.config import Configuration

__all__ = ["PatchLinkage"]


logger = logging.getLogger(__name__)


LINK_ZMIN = 0.05
"""The reference redshift at which the maximum angular size is computed."""


class PatchLinkage:
    """Class that links patches for pair counting, serves as task generator.

    This class is useful to generate pairs of patches that need to be paired
    for pair count measurements for a given maximum angular search radius.
    Patches that are separated by more than the sum of their radii and the
    maximum search radius do not contribute any pair counts and can be
    discarded.
    """

    def __init__(self, patch_tuples: list[PatchIDs]) -> None:
        """Populate a patch linkage container.

        To create a new linkage, use the :meth:`from_setup` method.

        Args:
            patch_tuples (list[:obj:`~yaw.core.containers.PatchIDs`]):
                List of patch pairs that need to be visited to count pairs.
        """
        self.pairs = patch_tuples

    @classmethod
    def from_setup(cls, config: Configuration, catalog: BaseCatalog) -> PatchLinkage:
        """Generate a new patch linkage.

        Compute a maximum angular separation for at low redshift for the scales
        provided in the configuration. Generate a list of all patch pairs that
        are separated by less than this maximum separation (factoring in the
        size of the patches).

        Args:
            config (`~yaw.Configuration`):
                Configuration object that defines the scales and cosmology
                needed to compute the maximum angular scale.
            catalog (:obj:`BaseCatalog`):
                Catalog instance with patch centers and sizes.

        Returns:
            :obj:`PatchLinkage`
        """
        # determine the additional overlap from the spatial query
        if config.backend.crosspatch:
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = max(LINK_ZMIN, config.binning.zmin)
            max_query_radius = r_kpc_to_angle(
                config.scales.as_array(), z_ref, config.cosmology
            ).max()
        else:
            max_query_radius = 0.0  # only relevant for cross-patch
        max_query_radius = DistSky(max_query_radius)

        logger.debug("computing patch linkage with %.3e", max_query_radius.values)
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
            "found %d patch links for %d patches", len(patch_pairs), catalog.n_patches
        )
        return cls(patch_pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(n_patches={self.n_patches}, n_pairs={len(self)})"

    @property
    def n_patches(self) -> int:
        """Get the total number of patches."""
        patches = set()
        for p1, p2 in self.pairs:
            patches.add(p1)
            patches.add(p2)
        return len(patches)

    @property
    def density(self) -> float:
        """Get ratio of the number of linked patch pairs compared to all
        possible combinations."""
        n = self.n_patches
        return len(self) / (n * n)

    def get_pairs(self, auto: bool, crosspatch: bool = True) -> list[PatchIDs]:
        """Get a list of linked patch pairs.

        Args:
            auto (:obj:`bool`):
                For autocorrelation measurements, only visit patch pairs where
                patch ID1 >= ID2.
            crosspatch (:obj:`bool`):
                If false, ignore all cross-patch pair counts and link patches
                just with themselves.

        Returns:
            :obj:`list[PatchIDs]`
        """
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
        collection1: BaseCatalog, collection2: BaseCatalog | None = None
    ) -> tuple[bool, BaseCatalog, BaseCatalog]:
        auto = collection2 is None
        if auto:
            collection2 = collection1
        return auto, collection1, collection2

    def get_matrix(
        self,
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None,
        crosspatch: bool = True,
    ) -> NDArray[np.bool_]:
        """Convert the list of linked patches to a boolean matrix indicating if
        two patches are linked.

        Depending on if one or two catalogs are provided as input, the result
        resembles a cross- or autocorrelation patch linkage. This is the only
        purpose of the inputs.

        Args:
            collection1 (:obj:`BaseCatalog`):
                First catalog for patch linkage.
            collection2 (:obj:`BaseCatalog`, optional):
                Second catalog for patch linkage. If not provided, returns a
                matrix for an autocorrelation case.
            crosspatch (:obj:`bool`):
                Link patches just with themselves and ignore cross-patch pairs.

        Returns:
            :obj:`NDArray`

        .. Warning::

            The patch centers of both catalogues must be (very close to)
            identical.
        """
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2
        )
        pairs = self.get_pairs(auto, crosspatch)
        # make a boolean matrix indicating the exisiting patch combinations
        n_patches = self.n_patches
        matrix = np.zeros((n_patches, n_patches), dtype=np.bool_)
        for pair in pairs:
            matrix[pair] = True
        return matrix

    def get_patches(
        self,
        collection1: BaseCatalog,
        collection2: BaseCatalog | None = None,
        crosspatch: bool = True,
    ) -> tuple[list[BaseCatalog], list[BaseCatalog]]:
        """Return linked pairs of patch data ready for processing.

        Instead of returning a list of patch index pairs, the actual patch data
        is returned in two aligned list, where item 1 form the first list is
        linked to item 1 of the second list. Depending on if one or two catalogs
        are provided as input, the result resembles a cross- or autocorrelation
        patch linkage.

        Args:
            collection1 (:obj:`BaseCatalog`):
                First catalog for patch linkage.
            collection2 (:obj:`BaseCatalog`, optional):
                Second catalog for patch linkage. If not provided, returns a
                matrix for an autocorrelation case.
            crosspatch (:obj:`bool`):
                Link patches just with themselves and ignore cross-patch pairs.

        Returns:
            list, list: Two lists with patch data from ``collection1`` and
            ``collection2`` (if provided, else ``collection1``) that are linked.

        .. Warning::

            The patch centers of both catalogues must be (very close to)
            identical.
        """
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2
        )
        pairs = self.get_pairs(auto, crosspatch)
        # generate the patch lists
        patches1 = []
        patches2 = []
        for id1, id2 in pairs:
            patches1.append(collection1[id1])
            patches2.append(collection2[id2])
        return patches1, patches2
