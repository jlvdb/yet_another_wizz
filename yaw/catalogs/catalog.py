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
    from yaw.correlation import HistogramData
    from yaw.paircounts import PairCountResult


class BackendError(Exception):
    pass


class BaseCatalog(Sequence, PatchedQuantity):
    """The data catalog base class.

    Every new backend must implement a catalog class. On creation this subclass
    is automatically registered and can be instantiated using the factory class
    :class:`~yaw.catalogs.NewCatalog`.
    """

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
    ) -> None:
        """Construct a catalog from a data frame.

        Args:
            data (:obj:`pandas.Dataframe`):
                Holds the catalog data.
            ra_name (str):
                Name of the column with right ascension data in degrees.
            dec_name (str):
                Name of the column with declination data in degress.
        
        Keyword Args:
            patch_name (str, optional):
                Name of the column that specifies the patch index, i.e.
                assigning each object to a spatial patch. Index starts counting
                from 0 (see :ref:`patches`).
            patch_centers (:obj:`BaseCatalog`, `Coordinate`, optional):
                Assign objects to existing patch centers based on their
                coordinates. Must be either a different catalog instance or a
                vector of coordinates.
            n_patches (int, optional):
                Assign objects to a given number of patches, generated using
                k-means clustering.
            redshift_name (str, optional):
                Name of the column with point-redshift estimates.
            weight_name (str, optional):
                Name of the column with object weights.
            cache_directory (str, optional):
                Path to directory used to cache patch data, must exists (see
                :ref:`caching`). If provided, patch data is automatically
                unloaded from memory.
            progress (bool, optional):
                Display a progress bar while creating patches.

        .. Note::
            Either of ``patch_name``, ``patch_centers``, or ``n_patches`` is
            required.

        Caching may significantly speed up parallel computations (e.g.
        :meth:`correlate`), accessing data attributes will trigger loading
        cached data as long as the catalog remains in the unloaded state (see
        :meth:`load` and :meth:`unload`).

        The underlying patch data can be accessed through indexing and
        iteration.
        ``TODO:`` add example.
        """
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
        **kwargs
    ) -> BaseCatalog:
        """
        Build catalog from data file.

        Loads the input file and constructs the catalog using the specified
        column names.

        Args:
            filepath (str):
                Path to the input data file.
            patches (str, int, :obj:`BaseCatalog`, :obj:`coordainte`):
                Specifies the construction of patches. If `str`, patch indices
                are read from the file. If `int`, generates this number of
                patches. Otherwise assign objects based on existing patch
                centers from a catalog instance or a coordinate vector.
            ra (str):
                Name of the column with right ascension data in degrees.
            dec (str):
                Name of the column with declination data in degress.
        
        Keyword Args:
            redshift (str, optional):
                Name of the column with point-redshift estimates.
            weight (str, optional):
                Name of the column with object weights.
            sparse (int, optional):
                Load every N-th row of the input data.
            cache_directory (str, optional):
                Path to directory used to cache patch data, must exists (see
                :ref:`caching`). If provided, patch data is automatically
                unloaded from memory.
            file_ext (str, optional):
                Hint for the input file type, if a uncommon file extension is
                used.
            progress (bool, optional):
                Display a progress bar while creating patches.

        Returns:
            :obj:`BaseCatalog`

        .. Note::
            Currently, the following file extensions are recognised
            automatically:

            - FITS: ``.fits``, ``.cat``
            - CSV: ``.csv``
            - HDF5: ``.hdf5``, ``.h5``,
            - Parquet: ``.pqt``, ``.parquet``
            - Feather: ``.feather``

            Otherwise provide the appropriate extension (including the dot)
            in the ``file_ext`` argument.
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
        """
        Restore the catalog from its cache directory.

        Args:
            cache_directory (str):
                Path to the cache directory.
            progress (bool, optional):
                Display a progress bar while restoring patches.

        Returns:
            :obj:`BaseCatalog`
        """
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
    def ids(self) -> list[int]:
        """Return a list of unique patch indices in the catalog"""
        pass

    @abstractmethod
    def __iter__(self) -> Iterator: pass

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
        self.logger.debug("bulk loading catalog")

    @abstractmethod
    def unload(self) -> None:
        """Unload data from memory if a disk cache is provided.

        Raises a :obj:`~yaw.catalogs.scipy.patches.CachingError` if no cache
        is configured.
        """
        self.logger.debug("bulk unloading catalog")

    @abstractmethod
    def has_redshifts(self) -> bool:
        """Indicates whether the :meth:`redshifts` attribute holds data."""

    @property
    def pos(self) -> CoordSky:
        """Get a vector of the object sky positions in radians."""
        return CoordSky(self.ra, self.dec)

    @abstractproperty
    def ra(self) -> NDArray[np.float_]:
        """Get the right ascension in radians."""
        pass

    @abstractproperty
    def dec(self) -> NDArray[np.float_]:
        """Get the declination in radians."""
        pass

    @abstractproperty
    def redshifts(self) -> NDArray[np.float_] | None:
        """Get the redshifts or ``None`` if not available."""
        pass

    @abstractproperty
    def weights(self) -> NDArray[np.float_]:
        """Get the redshifts or ``None`` if not available."""
        pass

    @abstractproperty
    def patch(self) -> NDArray[np.int_]:
        """Get the patch indices of each object."""
        pass

    @abstractmethod
    def get_min_redshift(self) -> float:
        """Get the minimum redshift or ``None`` if not available."""
        pass

    @abstractmethod
    def get_max_redshift(self) -> float:
        """Get the maximum redshift or ``None`` if not available."""
        pass

    @abstractproperty
    def total(self) -> float:
        """Get the sum of weights or the number of objects if not available."""

    @abstractmethod
    def get_totals(self) -> NDArray[np.float_]:
        """Get an array of the sum of weights or number of objects per patch."""

    @abstractproperty
    def centers(self) -> CoordSky:
        """Get a vector of sky coordinates of the patch centers in radians."""
        pass

    @abstractproperty
    def radii(self) -> DistSky:
        """Get a vector of distances in radians that describe the patch sizes.

        The radius of the patch is defined as the maximum angular distance of
        any object from the patch center."""
        pass

    @abstractmethod
    def correlate(
        self,
        config: Configuration,
        binned: bool,
        other: BaseCatalog = None,
        linkage: PatchLinkage | None = None,
        progress: bool = False
    ) -> PairCountResult | dict[str, PairCountResult]:
        """Compute the angular correlation between two catalogs.


        """
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
    ) -> HistogramData:
        """
        Compute a histogram of the object redshifts.

        Args:
            config (:obj:`~yaw.config.Configuration`):
                Defines the bin edges used for the histogram.
            sampling_config (:obj:`~yaw.config.ResamplingConfig`, optional):
                Specifies the spatial resampling for error estimates.
            progress (bool):
                Show a progress bar.

        Returns:
            HistogramData:
                Object holding the redshift histogram
        """
        self.logger.info("computing true redshift distribution")
