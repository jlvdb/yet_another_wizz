from __future__ import annotations

import logging
from abc import abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, TypeVar

import astropandas as apd
import numpy as np
from numpy.typing import NDArray

from yaw.core.coordinates import Coordinate, CoordSky, DistSky
from yaw.core.utils import long_num_format

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame

    from yaw.catalogs import PatchLinkage
    from yaw.config import Configuration, ResamplingConfig
    from yaw.correlation.paircounts import NormalisedCounts
    from yaw.redshifts import HistData

__all__ = ["BaseCatalog"]


_Tcat = TypeVar("_Tcat", bound="BaseCatalog")


class BackendError(Exception):
    pass


class BaseCatalog:
    """The data catalog base class.

    Every new backend must implement a catalog class based on this abstract base
    class. On import this subclass is automatically registered and can be
    instantiated using the factory class :class:`yaw.NewCatalog`.

    .. Note::

        Base classes must follow the ``[Backendname]Catalog`` naming convention.
        The new backend is then registered with name ``backendname`` (lower
        case).
    """

    _logger = logging.getLogger("yaw.catalog")
    _backends = dict()

    def __init_subclass__(cls, **kwargs):
        """Handles the backend subclass registration."""
        super().__init_subclass__(**kwargs)
        if not cls.__name__.endswith("Catalog"):
            raise BackendError(
                "subclasses of 'BaseCatalog' must follow naming convention "
                "'[Backend name]Catalog for registration (e.g. ScipyCatalog "
                "-> 'scipy')"
            )
        backend = cls.__name__.strip("Catalog").lower()
        cls._backends[backend] = cls

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
        progress: bool = False,
    ) -> None:
        """Build a catalogue from in-memory data.

        Catalogs should be instantiated through the factory class, see
        :meth:`yaw.catalogs.NewCatalog.from_dataframe`."""
        pass

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
        **kwargs,
    ) -> BaseCatalog:
        """Build a catalogue from data file.

        Catalogs should be instantiated through the factory class, see
        :meth:`yaw.catalogs.NewCatalog.from_file`."""
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
                "centers)"
            )

        cls._logger.info("reading catalog file '%s'", filepath)
        data = apd.read_auto(filepath, columns=columns, ext=file_ext, **kwargs)
        if sparse is not None:
            cls._logger.debug("sparse sampling data %ix", sparse)
            data = data[::sparse]
        return cls(
            data,
            ra,
            dec,
            **patch_kwarg,
            redshift_name=redshift,
            weight_name=weight,
            cache_directory=cache_directory,
            progress=progress,
        )

    @classmethod
    @abstractmethod
    def from_cache(cls, cache_directory: str, progress: bool = False) -> BaseCatalog:
        """Restore the catalogue from its cache directory.

        Catalogs should be instantiated through the factory class, see
        :meth:`yaw.catalogs.NewCatalog.from_cache`."""
        cls._logger.info("restoring from cache directory '%s'", cache_directory)

    def __repr__(self) -> str:
        name = self.__class__.__name__
        args = dict(
            loaded=self.is_loaded(),
            nobjects=len(self),
            npatches=self.n_patches,
            redshifts=self.has_redshifts(),
        )
        arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
        return f"{name}({arg_str})"

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, item: int) -> Any:
        pass

    @property
    @abstractmethod
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog."""
        pass

    @abstractmethod
    def n_patches(self) -> int:
        """The number of spatial patches of this catalogue."""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator:
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """Indicates whether the catalog data is loaded.

        Always ``True`` if no cache is used. If the catalog is unloaded, data
        will be read from cache every time data is accessed."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Permanently load data from cache into memory.

        Raises a :obj:`~yaw.catalogs.scipy.patches.CachingError` if no cache
        is configured.
        """
        self._logger.debug("bulk loading catalog")

    @abstractmethod
    def unload(self) -> None:
        """Unload data from memory if a disk cache is provided."""
        self._logger.debug("bulk unloading catalog")

    @abstractmethod
    def has_redshifts(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""
        pass

    @abstractmethod
    def has_weights(self) -> bool:
        """Indicates whether the :meth:`weights` attribute holds data."""
        pass

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        return CoordSky(self.ra, self.dec)

    @property
    @abstractmethod
    def ra(self) -> NDArray[np.float_]:
        """Get an array of the right ascension values in radians."""
        pass

    @property
    @abstractmethod
    def dec(self) -> NDArray[np.float_]:
        """Get an array of the declination values in radians."""
        pass

    @property
    @abstractmethod
    def redshifts(self) -> NDArray[np.float_] | None:
        """Get the redshifts as array or ``None`` if not available."""
        pass

    @property
    @abstractmethod
    def weights(self) -> NDArray[np.float_]:
        """Get the object weights as array or ``None`` if not available."""
        pass

    @property
    @abstractmethod
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object as array."""
        pass

    @abstractmethod
    def get_min_redshift(self) -> float:
        """Get the minimum redshift or ``None`` if not available."""
        pass

    @abstractmethod
    def get_max_redshift(self) -> float:
        """Get the maximum redshift or ``None`` if not available."""
        pass

    @property
    @abstractmethod
    def total(self) -> float:
        """Get the sum of weights or the number of objects if weights are not
        available."""

    @abstractmethod
    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects in each
        patch."""

    @property
    @abstractmethod
    def centers(self) -> CoordSky:
        """Get a vector of sky coordinates of the patch centers in radians.

        Returns:
            :obj:`yaw.core.coordinates.CoordSky`
        """
        pass

    @property
    @abstractmethod
    def radii(self) -> DistSky:
        """Get a vector of angular separations in radians that describe the
        patch sizes.

        The radius of the patch is defined as the maximum angular distance of
        any object from the patch center.

        Returns:
            :obj:`yaw.core.coordinates.DistSky`
        """
        pass

    @abstractmethod
    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: _Tcat = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False,
    ) -> NormalisedCounts | dict[str, NormalisedCounts]:
        """Count pairs between objects at a given separation and in bins of
        redshift.

        If another catalog instance is passed to ``other``, then pairs are
        formed between these catalogues (cross), otherwise pairs are formed with
        the catalog (auto). Pairs are counted in bins of redshift, as defined in
        the configuration object (``config``). Pairs are only considered within
        fixed angular scales that are computed from the physical scales in the
        configuration and the mid of the current redshift bin.

        Args:
            config (:obj:`yaw.Configuration`):
                Configuration object that defines measurement scales, redshift
                binning, cosmological model, and various backend specific
                parameters.
            binned (:obj:`bool`):
                Whether to apply the redshift binning to the second catalogue
                (see ``other``).
            other (Catalog instance, optional):
                Second catalog instance used for cross-catalogue pair counting.
                Catalogue must use the same backend.
            linkage (:obj:`~yaw.catalogs.linkage.PatchLinkage`, optional):
                Linkage object that defines with patches must be correlated for
                a given scales and which patch combinations can be skipped. Can
                be used for the ``scipy`` backend to count pairs consistently
                between multiple catalogue instances.
            progress (:obj:`bool`):
                Show a progress indication, depends on backend.

        There are three different modes of operation that are determined by the
        combination of the ``binned`` and ``other`` parameters:

        1. If no second catalogue is provided, pairs are counted within the
           catalogue while applying the redshift binning.
        2. If a second catalogue is provided and ``binned=True``, pairs are
           counted between the catalogues and the binning is applied to both
           cataluges.
        3. If a second catalogue is provided and ``binned=False``, the redshift
           binning is not applied to the second catalogue, otherwise above.

        The catalogue from the calling instance of :meth:`correlate` has always
        redshift binning applied.
        """
        n1 = long_num_format(len(self))
        n2 = long_num_format(len(self) if other is None else len(other))
        self._logger.debug(
            "correlating with %sbinned catalog (%sx%s) in %d redshift bins",
            "" if binned else "un",
            n1,
            n2,
            config.binning.zbin_num,
        )

    @abstractmethod
    def true_redshifts(
        self,
        config: Configuration,
        sampling_config: ResamplingConfig | None = None,
        progress: bool = False,
    ) -> HistData:
        """
        Compute a histogram of the object redshifts from the binning defined in
        the provided configuration.

        Args:
            config (:obj:`~yaw.config.Configuration`):
                Defines the bin edges used for the histogram.
            sampling_config (:obj:`~yaw.config.ResamplingConfig`, optional):
                Specifies the spatial resampling for error estimates.
            progress (:obj:`bool`):
                Show a progress bar.

        Returns:
            HistData:
                Object holding the redshift histogram
        """
        self._logger.info("computing true redshift distribution")
