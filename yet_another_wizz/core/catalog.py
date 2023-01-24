from __future__ import annotations

from abc import ABC, abstractmethod, abstractproperty
from collections.abc import Iterator
from typing import Any

import astropandas as apd
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import distance_matrix

from yet_another_wizz.core.config import Configuration
from yet_another_wizz.core.coordinates import (
    distance_sphere2sky, position_sphere2sky)
from yet_another_wizz.core.redshifts import NzTrue
from yet_another_wizz.core.resampling import PairCountResult
from yet_another_wizz.core.utils import TypePatchKey, TypeScaleKey


class CatalogBase(ABC):

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
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
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
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        Unload data from memory if a disk cache is provided.
        """
        pass

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
        other: CatalogBase = None
    ) -> dict[TypeScaleKey, PairCountResult]:
        pass

    @abstractmethod
    def true_redshifts(
        self,
        config: Configuration
    ) -> NzTrue:
        """
        Compute the a redshift distribution histogram.
        """
        pass


class PatchLinkage:

    def __init__(
        self,
        patch_tuples: list[TypePatchKey]
    ) -> None:
        self.pairs = patch_tuples

    @classmethod
    def from_catalog(
        cls,
        catalog: CatalogBase,
        max_query_radius: float
    ) -> PatchLinkage:
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
        return cls(patch_pairs)

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
        collection1: CatalogBase,
        collection2: CatalogBase | None = None
    ) -> tuple[bool, CatalogBase, CatalogBase, int]:
        auto = collection2 is None
        if auto:
            collection2 = collection1
        n_patches = max(collection1.n_patches(), collection2.n_patches())
        return auto, collection1, collection2, n_patches

    def get_matrix(
        self,
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
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
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
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
        collection1: CatalogBase,
        collection2: CatalogBase | None = None,
        crosspatch: bool = True
    ) -> NDArray[np.float_]:
        auto, collection1, collection2, n_patches = self._parse_collections(
            collection1, collection2)
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
