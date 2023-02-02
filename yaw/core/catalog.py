from __future__ import annotations

import logging
from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import astropandas as apd
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from yaw.core.coordinates import distance_sphere2sky, position_sphere2sky
from yaw.core.cosmology import r_kpc_to_angle
from yaw.core.utils import long_num_format

if TYPE_CHECKING:
    from pandas import DataFrame
    from yaw.core.config import Configuration
    from yaw.core.redshifts import NzTrue
    from yaw.core.resampling import PairCountResult
    from yaw.core.utils import (
        TypePatchKey, TypeScaleKey)


logger = logging.getLogger(__name__.replace(".core.", "."))


class CatalogBase(ABC):

    logger = logging.getLogger("yaw.Catalog")

    @abstractmethod
    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: CatalogBase | NDArray[np.float_] | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_file(
        cls,
        filepath: str,
        patches: int | CatalogBase | str,
        ra: str,
        dec: str,
        *,
        redshift: str | None = None,
        weight: str | None = None,
        sparse: int | None = None,
        cache_directory: str | None = None,
        file_ext: str | None = None,
        **kwargs
    ) -> CatalogBase:
        """
        TODO
        """
        columns = [c for c in [ra, dec, redshift, weight] if c is not None]
        if isinstance(patches, str):
            columns.append(patches)
            patch_kwarg = dict(patch_name=patches)
        elif isinstance(patches, CatalogBase):
            patch_kwarg = dict(
                patch_centers=position_sphere2sky(patches.centers))
        elif isinstance(patches, int):
            patch_kwarg = dict(n_patches=patches)
        else:
            raise TypeError(
                "'patches' must be either of type 'int', 'str', or 'Catalog'")

        cls.logger.info(f"reading catalog file '{filepath}'")
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
            cls.logger.debug(f"sparse sampling data {sparse}x")
            data = data[::sparse]
        return cls(
            data, ra, dec, **patch_kwarg,
            redshift_name=redshift,
            weight_name=weight,
            cache_directory=cache_directory)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            loaded=self.is_loaded(),
            nobjects=len(self),
            npatches=self.n_patches(),
            redshifts=self.has_redshifts())
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        pass

    @abstractproperty
    def ids(self) -> list[int]:
        """
        Get a list of patch IDs.
        """
        pass

    @abstractmethod
    def n_patches(self) -> int:
        """
        Get the number of patches (spatial regions).
        """
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Whether the data has been loaded into memory.
        """
        pass

    @abstractmethod
    def load(self) -> None:
        """
        Load data from a disk cache into memory.
        """
        self.logger.debug("bulk loading catalog")

    @abstractmethod
    def unload(self) -> None:
        """
        Unload data from memory if a disk cache is provided.
        """
        self.logger.debug("bulk unloading catalog")

    @abstractmethod
    def has_redshifts(self) -> bool:
        """
        Whether the catalogue has redshift data.
        """
        pass

    @abstractproperty
    def ra(self) -> NDArray[np.float_]:
        """
        Array of right ascensions in radians.
        """
        pass

    @abstractproperty
    def dec(self) -> NDArray[np.float_]:
        """
        Array of declinations in radians.
        """
        pass

    @abstractproperty
    def redshifts(self) -> NDArray[np.float_] | None:
        """
        Array of redshifts.
        """
        pass

    @abstractproperty
    def weights(self) -> NDArray[np.float_]:
        """
        Array of individual object weights.
        """
        pass

    @abstractproperty
    def patch(self) -> NDArray[np.int_]:
        """
        Array of patch memberships.
        """
        pass

    @abstractmethod
    def get_min_redshift(self) -> float:
        """
        Get the minimum redshift in the catalogue.
        """
        pass

    @abstractmethod
    def get_max_redshift(self) -> float:
        """
        Get the maximum redshift in the catalogue.
        """
        pass

    @abstractproperty
    def total(self) -> float:
        """
        Get the sum of object weights.
        """
        pass

    @abstractmethod
    def get_totals(self) -> NDArray[np.float_]:
        """
        Get the sum of object weights per patch.
        """
        pass

    @abstractproperty
    def centers(self) -> NDArray[np.float_]:
        """
        Get the patch centers in right ascension / declination (radians).
        """
        pass

    @abstractproperty
    def radii(self) -> NDArray[np.float_]:
        """
        Get the distance from the patches center to its farthest member in
        radians.
        """
        pass

    @abstractmethod
    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: CatalogBase = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False
    ) -> PairCountResult | dict[TypeScaleKey, PairCountResult]:
        n1 = long_num_format(len(self))
        n2 = long_num_format(len(self) if other is None else len(other))
        self.logger.debug(
            f"correlating with {'' if binned else 'un'}binned catalog "
            f"({n1}x{n2}) in {config.binning.zbin_num} redshift bins")

    @abstractmethod
    def true_redshifts(
        self,
        config: Configuration
    ) -> NzTrue:
        """
        Compute the a redshift distribution histogram.
        """
        self.logger.debug("computing true redshift distribution")


class PatchLinkage:

    def __init__(
        self,
        patch_tuples: list[TypePatchKey]
    ) -> None:
        self.pairs = patch_tuples

    @classmethod
    def from_setup(
        cls,
        config: Configuration,
        catalog: CatalogBase
    ) -> PatchLinkage:
        # determine the additional overlap from the spatial query
        if config.backend.crosspatch:
            # estimate maximum query radius at low, but non-zero redshift
            z_ref = max(0.05, config.binning.zmin)
            max_query_radius = r_kpc_to_angle(
                config.scales.scales, z_ref, config.cosmology).max()
        else:
            max_query_radius = 0.0  # only relevenat for cross-patch

        logger.debug(f"computing patch linkage with {max_query_radius=:.3e}")
        centers = catalog.centers  # in RA / Dec
        radii = catalog.radii  # radian, maximum distance measured from center
        # compute distance between all patch centers
        dist = distance_sphere2sky(distance_matrix(centers, centers))
        # compare minimum separation required for patchs to not overlap
        min_sep_deg = np.add.outer(radii, radii)

        # check which patches overlap when factoring in the query radius
        overlaps = (dist - max_query_radius) < min_sep_deg
        patch_pairs = []
        for id1, overlap in enumerate(overlaps):
            patch_pairs.extend((id1, id2) for id2 in np.where(overlap)[0])
        logger.debug(
            f"found {len(patch_pairs)} patch links "
            f"for {catalog.n_patches()} patches")
        return cls(patch_pairs)

    def __len__(self) -> int:
        return len(self.pairs)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}(npatches={self.n_patches()}, pairs={len(self)})"

    def n_patches(self) -> int:
        patches = set()
        for p1, p2 in self.pairs:
            patches.add(p1)
            patches.add(p2)
        return len(patches)

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
        collection1: CatalogBase,
        collection2: CatalogBase | None = None
    ) -> tuple[bool, CatalogBase, CatalogBase]:
        auto = collection2 is None
        if auto:
            collection2 = collection1
        return auto, collection1, collection2

    def get_matrix(
        self,
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        pairs = self.get_pairs(auto, crosspatch)
        # make a boolean matrix indicating the exisiting patch combinations
        n_patches = self.n_patches()
        matrix = np.zeros((n_patches, n_patches), dtype=np.bool_)
        for pair in pairs:
            matrix[pair] = True
        return matrix

    def get_mask(
        self,
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.bool_]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        # make a boolean mask indicating all patch combinations
        n_patches = self.n_patches()
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
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.float_]:
        auto, collection1, collection2 = self._parse_collections(
            collection1, collection2)
        n_patches = self.n_patches()
        # compute the product of the total weight per patch
        totals1 = np.zeros(n_patches)
        for i, total in zip(collection1.ids, collection1.get_totals()):
            totals1[i] = total
        totals2 = np.zeros(n_patches)
        for i, total in zip(collection2.ids, collection2.get_totals()):
            totals2[i] = total
        totals = np.multiply.outer(totals1, totals2)
        if auto:
            totals = np.triu(totals)  # (i, j) with i > j => 0
            totals[np.diag_indices(len(totals))] *= 0.5  # avoid double-counting
        return totals

    def get_patches(
        self,
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
        crosspatch: bool = True
    ) -> tuple[list[CatalogBase], list[CatalogBase]]:
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
