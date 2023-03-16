from __future__ import annotations

import logging
from abc import ABC, abstractclassmethod, abstractmethod, abstractproperty
from collections.abc import Iterator, Sequence
from typing import TYPE_CHECKING, Any

import astropandas as apd
import numpy as np
from numpy.typing import NDArray

from yaw.coordinates import Coordinate, CoordSky, DistSky
from yaw.utils import PatchedQuantity, long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame
    from yaw.catalogs import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.correlation import HistgramData
    from yaw.paircounts import PairCountResult


class BackendError(Exception):
    pass


class BaseCatalog(Sequence, PatchedQuantity):

    logger = logging.getLogger("yaw.catalog")
    backends = dict()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not cls.__name__.endswith("Catalog"):
            raise BackendError(
                f"subclasses of 'BaseCatalog' must follow naming convention "
                f"'[Backend name]Catalog for registration (e.g. ScipyCatalog "
                f"-> 'scipy')")
        backend = cls.__name__.strip("Catalog").lower()
        cls.backends[backend] = cls

    @abstractmethod
    def __init__(
        self,
        data: DataFrame,
        ra_name: str,
        dec_name: str,
        *,
        patch_name: str | None = None,
        patch_centers: BaseCatalog | Coordinate | None = None,
        n_patches: int | None = None,
        redshift_name: str | None = None,
        weight_name: str | None = None,
        cache_directory: str | None = None,
        progress: bool = False
    ) -> None: pass

    @classmethod
    def from_file(
        cls,
        filepath: str,
        patches: str | int | BaseCatalog | Coordinate,
        ra: str,
        dec: str,
        *,
        redshift: str | None = None,
        weight: str | None = None,
        sparse: int | None = None,
        cache_directory: str | None = None,
        file_ext: str | None = None,
        progress: bool = False,
        **kwargs
    ) -> BaseCatalog:
        """
        TODO
        """
        columns = [c for c in [ra, dec, redshift, weight] if c is not None]
        if isinstance(patches, str):
            columns.append(patches)
            patch_kwarg = dict(patch_name=patches)
        elif isinstance(patches, int):
            patch_kwarg = dict(n_patches=patches)
        elif isinstance(patches, Coordinate):
            patch_kwarg = dict(patch_centers=patches)
        elif isinstance(patches, BaseCatalog):
            patch_kwarg = dict(patch_centers=patches.centers)
        else:
            raise TypeError(
                "'patches' must be either of type 'str' (col. name), 'int' "
                "(number of patches), or 'Catalog' or 'Coordinate' (specify "
                "centers)")

        cls.logger.info(f"reading catalog file '{filepath}'")
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
            cls.logger.debug(f"sparse sampling data {sparse}x")
            data = data[::sparse]
        return cls(
            data, ra, dec, **patch_kwarg,
            redshift_name=redshift,
            weight_name=weight,
            cache_directory=cache_directory,
            progress=progress)

    @abstractclassmethod
    def from_cache(
        cls,
        cache_directory: str,
        progress: bool = False
    ) -> BaseCatalog:
        cls.logger.info(f"restoring from cache directory '{cache_directory}'")

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            loaded=self.is_loaded(),
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshifts())
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    @abstractmethod
    def __len__(self) -> int: pass

    @abstractmethod
    def __getitem__(self, item: int) -> Any: pass

    @abstractproperty
    def ids(self) -> list[int]: pass

    @abstractmethod
    def __iter__(self) -> Iterator: pass

    @abstractmethod
    def is_loaded(self) -> bool: pass

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
    def has_redshifts(self) -> bool: pass

    def pos(self) -> CoordSky:
        return CoordSky(self.ra, self.dec)

    @abstractproperty
    def ra(self) -> NDArray[np.float_]: pass

    @abstractproperty
    def dec(self) -> NDArray[np.float_]: pass

    @abstractproperty
    def redshifts(self) -> NDArray[np.float_] | None: pass

    @abstractproperty
    def weights(self) -> NDArray[np.float_]: pass

    @abstractproperty
    def patch(self) -> NDArray[np.int_]: pass

    @abstractmethod
    def get_min_redshift(self) -> float: pass

    @abstractmethod
    def get_max_redshift(self) -> float: pass

    @abstractproperty
    def total(self) -> float: pass

    @abstractmethod
    def get_totals(self) -> NDArray[np.float_]: pass

    @abstractproperty
    def centers(self) -> CoordSky: pass

    @abstractproperty
    def radii(self) -> DistSky: pass

    @abstractmethod
    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: BaseCatalog = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False
    ) -> PairCountResult | dict[str, PairCountResult]:
        n1 = long_num_format(len(self))
        n2 = long_num_format(len(self) if other is None else len(other))
        self.logger.debug(
            f"correlating with {'' if binned else 'un'}binned catalog "
            f"({n1}x{n2}) in {config.binning.zbin_num} redshift bins")

    @abstractmethod
    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False
    ) -> HistgramData:
        """
        Compute the a redshift distribution histogram.
        """
        self.logger.info("computing true redshift distribution")
