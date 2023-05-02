from __future__ import annotations

from typing import TYPE_CHECKING

from yaw.catalogs.catalog import BackendError, BaseCatalog

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame
    from yaw.core.coordinates import Coordinate


class NewCatalog:
    """Factory class for data catalogs implemented by the backends.

    Configured for a specific backend, this class provides a uniform interface
    to the different constructors of subclasses of
    :class:`~yaw.catalogs.BaseCatalog`.

    Args:
        backend (str):
            Specify the backend for which the catalog instances should be
            produced for. For availble options see
            :attr:`~yaw.catalogs.BACKEND_OPTIONS`.
    """

    def __init__(self, backend: str = "scipy") -> None:
        try:
            self.catalog: BaseCatalog = BaseCatalog.backends[backend]
            self.backend_name = backend
        except KeyError as e:
            raise BackendError(f"invalid backend '{backend}'") from e

    def from_dataframe(
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
    ) -> BaseCatalog:
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

        Returns:
            :obj:`BaseCatalog`

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
        return self.catalog(
            data,
            ra_name,
            dec_name,
            patch_name=patch_name,
            patch_centers=patch_centers,
            n_patches=n_patches,
            redshift_name=redshift_name,
            weight_name=weight_name,
            cache_directory=cache_directory,
            progress=progress)

    def from_file(
        self,
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
        return self.catalog.from_file(
            filepath,
            patches,
            ra,
            dec,
            redshift=redshift,
            weight=weight,
            sparse=sparse,
            cache_directory=cache_directory,
            file_ext=file_ext,
            progress=progress,
            **kwargs)

    def from_cache(
        self,
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
        return self.catalog.from_cache(cache_directory, progress=progress)
