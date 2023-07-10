from __future__ import annotations

from typing import TYPE_CHECKING

from yaw.catalogs.catalog import BackendError, BaseCatalog

if TYPE_CHECKING:  # pragma: no cover
    from pandas import DataFrame

    from yaw.core.coordinates import Coordinate

__all__ = ["NewCatalog"]


class NewCatalog:
    """Factory class for data catalogues implemented by the backends.

    A catalogue provides all the functionality to compute pair counts for
    correlation measurements by implementing an interface to the object
    positions, spatial patches for error estimation, and data management if the
    data is cached on disk. Aside from accessing the data directly, the most
    important methods are the :meth:`correlate` (pair counting) and
    :meth:`true_redshifts` (redshift histogram, if redshifts are provided).

    A new catalogue can be created using an instance of this factory class.
    The sole argument is the name of the backend for which catalogue instances
    should be produced. For example

    >>> yaw.NewCatalog("scipy")
    NewCatalog<scipy>()

    is the default factory, which produces catalogues for the ``scipy`` backend
    through its constructor methods.

    A key concept is :ref:`caching<caching>`, which can be used to reduce
    memory usage or even speed up the computation for some backends. A cache
    directory is a directory in which temporary data is stored in different
    formats (depending on the backend), such that parts of the data (typically
    individual spatial patches) can be read back into memory on demand.

    .. Warning::

        - The ``scipy`` backend does not preserve the order the input data, but
          instead groups objects by there spatial patch.
        - The ``treecorr`` backend does currently not support restoration from
          cache.
    """

    def __init__(self, backend: str = "scipy") -> None:
        """Create a new catalogue factory.

        Args:
            backend (:obj:`str`):
                Specify the backend for which the catalog instances should be
                produced for. For availble options see
                :attr:`~yaw.config.options.Options.backend`.
        """
        try:
            self.catalog: BaseCatalog = BaseCatalog._backends[backend]
            self.backend_name = backend
        except KeyError as e:
            raise BackendError(f"invalid backend '{backend}'") from e

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}<{self.backend_name}>()"

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
        progress: bool = False,
    ) -> BaseCatalog:
        """Build a catalogue from in-memory data.

        Specify the names of the required and or available columns in a
        :obj:`pandas.DataFrame`. Additional parameters control the creation
        spatial patches used for error estimates. Patches can be assigned based
        on a column in the data frame (``patch_name``), constructed from a set
        of existing patch centers (``patch_centers``), or generated with
        `k`-means clustering (``n_patches``).

        Args:
            data (:obj:`pandas.Dataframe`):
                Holds the catalog data.
            ra_name (:obj:`str`):
                Name of the column with right ascension data in degrees.
            dec_name (:obj:`str`):
                Name of the column with declination data in degress.

        Keyword Args:
            patch_name (:obj:`str`, optional):
                Name of the column that specifies the patch index, i.e.
                assigning each object to a spatial patch. Index starts counting
                from 0 (see :ref:`patches`).
            patch_centers (:obj:`~yaw.catalogs.BaseCatalog`, :obj:`~yaw.core.coordinates.Coordinate`, optional):
                Assign objects to existing patch centers based on their
                coordinates. Must be either a different catalog instance or a
                vector of coordinates.
            n_patches (:obj:`int`, optional):
                Assign objects to a given number of patches, generated using
                k-means clustering.
            redshift_name (:obj:`str`, optional):
                Name of the column with point-redshift estimates.
            weight_name (:obj:`str`, optional):
                Name of the column with object weights.
            cache_directory (:obj:`str`, optional):
                Path to directory used to cache patch data, must exists (see
                :ref:`caching`). If provided, patch data is automatically
                unloaded from memory.
            progress (:obj:`bool`, optional):
                Display a progress bar while creating patches.

        .. Note::
            Either of ``patch_name``, ``patch_centers``, or ``n_patches`` is
            required.

        Caching may significantly speed up parallel computations (e.g.
        :meth:`correlate`), accessing data attributes will trigger loading
        cached data as long as the catalog remains in the unloaded state (see
        :meth:`load` and :meth:`unload`).

        The underlying patch data can be accessed through indexing and
        iterating the Catalog instance.

        .. Note::
            TODO: Provide an example.
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
            progress=progress,
        )

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
        **kwargs,
    ) -> BaseCatalog:
        """
        Build catalogue from data file.

        Loads the input file and constructs the catalogue using the specified
        column names.

        Args:
            filepath (:obj:`str`):
                Path to the input data file.
            patches (:obj:`str`, :obj:`int`, :obj:`~yaw.catalogs.BaseCatalog`, :obj:`~yaw.core.coordinates.Coordinate`):
                Specifies the construction of patches. If `str`, patch indices
                are read from the file. If `int`, generates this number of
                patches. Otherwise assign objects based on existing patch
                centers from a catalog instance or a coordinate vector.
            ra (:obj:`str`):
                Name of the column with right ascension data in degrees.
            dec (:obj:`str`):
                Name of the column with declination data in degress.

        Keyword Args:
            redshift (:obj:`str`, optional):
                Name of the column with point-redshift estimates.
            weight (:obj:`str`, optional):
                Name of the column with object weights.
            sparse (:obj:`int`, optional):
                Load every N-th row of the input data.
            cache_directory (:obj:`str`, optional):
                Path to directory used to cache patch data, must exists (see
                :ref:`caching`). If provided, patch data is automatically
                unloaded from memory.
            file_ext (:obj:`str`, optional):
                Hint for the input file type, if a uncommon file extension is
                used.
            progress (:obj:`bool`, optional):
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
            **kwargs,
        )

    def from_cache(self, cache_directory: str, progress: bool = False) -> BaseCatalog:
        """
        Restore the catalogue from its cache directory.

        Args:
            cache_directory (:obj:`str`):
                Path to the cache directory.
            progress (:obj:`bool`, optional):
                Display a progress bar while restoring patches.

        Returns:
            :obj:`BaseCatalog`
        """
        return self.catalog.from_cache(cache_directory, progress=progress)
