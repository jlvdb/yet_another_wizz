from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.readers import DataFrameReader, new_filereader
from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import (
    PATCH_NAME_TEMPLATE,
    CatalogBase,
    InconsistentPatchesError,
    PatchBase,
    PatchData,
)
from yaw.catalog.writers import PATCH_INFO_FILE, PatchMode, create_patch_centers
from yaw.containers import YamlSerialisable, parse_binning
from yaw.options import Closed
from yaw.utils import AngularCoordinates, AngularDistances, parallel
from yaw.utils.logging import Indicator

if parallel.use_mpi():
    from yaw.catalog.writers.mpi4py import write_patches
else:
    from yaw.catalog.writers.multiprocessing import write_patches

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Union

    from numpy.typing import NDArray

    from yaw.catalog.utils import MockDataFrame as DataFrame

    TypePatchCenters = Union["Catalog", AngularCoordinates]

__all__ = [
    "Catalog",
    "Patch",
]

logger = logging.getLogger("yaw.catalog")


class Metadata(YamlSerialisable):
    """
    Container for patch meta data.

    Bundles the number of records stored in the patch, the sum of weights, and
    distribution of objects on sky through the center point and containing
    radius.

    Args:
        num_records:
            Number of data points in the patch.
        total:
            Sum of point weights or same as :obj:`num_records`.
        center:
            Center point (mean) of all data points,
            :obj:`~yaw.AngularCoordinates` in radian.
        radius:
            Radius around center point containing all data points,
            :obj:`~yaw.AngularDistances` in radian.
    """

    __slots__ = (
        "num_records",
        "total",
        "center",
        "radius",
    )

    num_records: int
    """Number of data points in the patch."""
    total: float
    """Sum of point weights."""
    center: AngularCoordinates
    """Center point (mean) of all data points."""
    radius: AngularDistances
    """Radius around center point containing all data points."""

    def __init__(
        self,
        *,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius

    def __repr__(self) -> str:
        items = (
            f"num_records={self.num_records}",
            f"total={self.total}",
            f"center={self.center}",
            f"radius={self.radius}",
        )
        return f"{type(self).__name__}({', '.join(items)})"

    @classmethod
    def compute(
        cls,
        coords: AngularCoordinates,
        *,
        weights: NDArray | None = None,
        center: AngularCoordinates | None = None,
    ) -> Metadata:
        """
        Compute the meta data from the patch data.

        If no weights are provided, the sum of weights will equal the number of
        data points. Weights are also used when computing the center point.

        Args:
            coords:
                Coordinates of patch data points, given as
                :obj:`~yaw.AngularCoordinates`.

        Keyword Args:
            weights:
                Optional, weights of data points.
            center:
                Optional, use this specific center point, e.g. when using an
                externally computed patch center.

        Returns:
            Final instance of meta data.
        """
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        if center is not None:
            if len(center) != 1:
                raise ValueError("'center' must be one single coordinate")
            new.center = center.copy()
        else:
            new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()

        return new

    @classmethod
    def from_dict(cls, kwarg_dict: dict) -> Metadata:
        center = AngularCoordinates(kwarg_dict.pop("center"))
        radius = AngularDistances(kwarg_dict.pop("radius"))
        return cls(center=center, radius=radius, **kwarg_dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


class Patch(PatchBase):
    """
    A single spatial patch of catalog data.

    Data has point coordinates and optionally weights and redshifts. This data
    is cached on disk in a binary file (``data.bin``) that is read when
    accessing any of the classes data attributes. Additionaly meta data, such as
    the patch center and radius, that describe the spatial distribution of the
    contained data points, are availble and stored as YAML file (``meta.yml``)

    The cached data is organised in a single directory as follows::

        [cache_path]/
          ├╴ data.bin
          ├╴ meta.yml
          ├╴ binning   (optional, see catalog.trees.BinnedTrees)
          └╴ trees.pkl (optional, see catalog.trees.BinnedTrees)

    Supports efficient pickeling as long as the cached data is not deleted or
    moved.
    """

    __slots__ = ("meta", "cache_path", "_has_weights", "_has_redshifts")

    meta: Metadata
    """Patch meta data; number of records stored in the patch, the sum of
    weights, and distribution of objects on sky through the center point and
    containing radius."""

    def __init__(
        self, cache_path: Path | str, center: AngularCoordinates | None = None
    ) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)
            with self.data_path.open(mode="rb") as f:
                self._has_weights, self._has_redshifts = PatchData.read_header(f)

        except FileNotFoundError:
            data = PatchData.from_file(self.data_path)
            self._has_weights = data.has_weights
            self._has_redshifts = data.has_redshifts

            self.meta = Metadata.compute(
                data.coords, weights=data.weights, center=center
            )
            self.meta.to_file(meta_data_file)

    def __repr__(self) -> str:
        items = (
            f"num_records={self.meta.num_records}",
            f"total={self.meta.total}",
            f"has_weights={self._has_weights}",
            f"has_redshifts={self._has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_path}"

    def __getstate__(self) -> dict:
        return dict(
            cache_path=self.cache_path,
            meta=self.meta,
            _has_weights=self._has_weights,
            _has_redshifts=self._has_redshifts,
        )

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    @staticmethod
    def id_from_path(cache_path: Path | str) -> int:
        """
        Extract the integer patch ID from the cache path.

        .. caution::
            This will fail if the patch has not been created through a
            :obj:`~yaw.Catalog` instance, which manages the patch creation.
        """
        _, id_str = Path(cache_path).name.split("_")
        return int(id_str)

    def load_data(self) -> PatchData:
        """
        Load the cached object data with coordinates and optional weights and
        redshifts.

        Returns:
            A special :obj:`PatchData` container that has the same
            :obj:`coords`, :obj:`weights`, and :obj:`redshifts` attributes as
            :obj:`Patch`.
        """
        return PatchData.from_file(self.data_path)

    @property
    def coords(self) -> AngularCoordinates:
        """Coordinates in right ascension and declination, in radian."""
        return self.load_data().coords

    @property
    def weights(self) -> NDArray | None:
        """Weights or ``None`` if there are no weights."""
        return self.load_data().weights

    @property
    def redshifts(self) -> NDArray | None:
        """Redshifts or ``None`` if there are no redshifts."""
        return self.load_data().redshifts

    def get_trees(self) -> BinnedTrees:
        """
        Try loading the binary search trees.

        Loads the tree(s) from the ``trees.pkl`` pickle file, other raises an
        error.

        Returns:
            :obj:`~yaw.catalog.trees.BinnedTrees` container with a single or
            multiple (when catalog is binned in redshift) binary search trees.

        Raises:
            FileNotFoundError:
                If the trees have not been build previously.
        """
        return BinnedTrees(self)


def write_catalog(
    cache_directory: Path | str,
    source: DataFrame | Path | str,
    *,
    ra_name: str,
    dec_name: str,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    patch_centers: TypePatchCenters | None = None,
    patch_name: str | None = None,
    patch_num: int | None = None,
    degrees: bool = True,
    chunksize: int | None = None,
    probe_size: int = -1,
    overwrite: bool = False,
    progress: bool = False,
    max_workers: int | None = None,
    buffersize: int = -1,
    **reader_kwargs,
) -> None:
    constructor = new_filereader if isinstance(source, (Path, str)) else DataFrameReader

    reader = None
    if parallel.on_root():
        actual_reader = constructor(
            source,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
            **reader_kwargs,
        )
        reader = actual_reader.get_dummy()

    reader = parallel.COMM.bcast(reader, root=0)
    if parallel.on_root():
        reader = actual_reader

    mode = PatchMode.determine(patch_centers, patch_name, patch_num)
    if mode == PatchMode.create:
        patch_centers = None
        if parallel.on_root():
            patch_centers = create_patch_centers(reader, patch_num, probe_size)
        patch_centers = parallel.COMM.bcast(patch_centers, root=0)

    # split the data into patches and create the cached Patch instances.
    write_patches(
        cache_directory,
        reader,
        patch_centers,
        overwrite=overwrite,
        progress=progress,
        max_workers=max_workers,
        buffersize=buffersize,
    )


def read_patch_ids(cache_directory: Path) -> list[int]:
    path = cache_directory / PATCH_INFO_FILE
    if not path.exists():
        raise InconsistentPatchesError("patch info file not found")
    return np.fromfile(path, dtype=np.int16).tolist()


def load_patches(
    cache_directory: Path,
    *,
    patch_centers: TypePatchCenters | None,
    progress: bool,
    max_workers: int | None = None,
) -> dict[int, Patch]:
    patch_ids = None
    if parallel.on_root():
        patch_ids = read_patch_ids(cache_directory)
    patch_ids = parallel.COMM.bcast(patch_ids, root=0)

    # instantiate patches, which triggers computing the patch meta-data
    path_template = str(cache_directory / PATCH_NAME_TEMPLATE)
    patch_paths = map(path_template.format, patch_ids)

    if patch_centers is not None:
        if isinstance(patch_centers, Catalog):
            patch_centers = patch_centers.get_centers()
        patch_arg_iter = zip(patch_paths, patch_centers)

    else:
        patch_arg_iter = zip(patch_paths)

    patch_iter = parallel.iter_unordered(
        Patch, patch_arg_iter, unpack=True, max_workers=max_workers
    )
    if progress:
        patch_iter = Indicator(patch_iter, len(patch_ids))

    patches = {Patch.id_from_path(patch.cache_path): patch for patch in patch_iter}
    return parallel.COMM.bcast(patches, root=0)


class Catalog(CatalogBase, Mapping[int, Patch]):
    """
    A container for catalog data.

    Catalogs are the core data structure for managing point data catalogs.
    Besides right ascension and declination coordinates, catalogs may have
    additional per-object weights and redshifts.

    Catalogs divided into spatial :obj:`~yaw.Patch` es, which each cache a
    portion of the data on disk to minimise the memory footprint when dealing
    with large data-sets, allowing to process the data in a patch-wise manner,
    only loading data from disk when they are needed. Additionally, the patches
    are used to estimate uncertainties using jackknife resampling.

    .. note::
        The number of patches should be sufficently large to support the
        redshift binning used for correlation measurements. The number of
        patches is also a trade-off between runtime and memory footprint during
        correlation measurements.

    The cached data is organised in a single directory, with one sub-directory
    for each spatial :obj:`~yaw.Patch`::

        [cache_directory]/
          ├╴ patch_ids.bin  # list of patch IDs for this catalog
          ├╴ patch_0/
          │    └╴ ...  # patch data
          ├╴ patch_1/
          │  ...
          └╴ patch_N/

    Args:
        cache_directory:
            The cache directory to use for this catalog, must exist and contain
            a valid catalog cache.

    Keyword Args:
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default).
    """

    __slots__ = ("cache_directory", "_patches")

    _patches: dict[int, Patch]

    def __init__(
        self, cache_directory: Path | str, *, max_workers: int | None = None
    ) -> None:
        if parallel.on_root():
            logger.info("restoring from cache directory: %s", cache_directory)

        self.cache_directory = Path(cache_directory)
        if not self.cache_directory.exists():
            raise OSError(f"cache directory not found: {self.cache_directory}")

        self._patches = load_patches(
            self.cache_directory,
            patch_centers=None,
            progress=False,
            max_workers=max_workers,
        )

    @classmethod
    def from_dataframe(
        cls,
        cache_directory: Path | str,
        dataframe: DataFrame,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_centers: TypePatchCenters | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        overwrite: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
        chunksize: int | None = None,
        probe_size: int = -1,
        **reader_kwargs,
    ) -> Catalog:
        """
        Create a new catalog instance from a :obj:`pandas.DataFrame`.

        Assign objects from the input data frame to spatial patches,
        write the patches to a cache on disk, and compute the patch meta data.

        .. note::
            One of the optional patch creation arguments (``patch_centers``,
            ``patch_name``, or ``patch_num``) must be provided.

        Args:
            cache_directory:
                The cache directory to use for this catalog. Created
                automatically or overwritten if requested.
            dataframe:
                The input data frame. May also be an object that supports
                mapping from string (column name) to data (numpy array-like).

        Keyword Args:
            ra_name:
                Column name in the data frame for right ascension.
            dec_name:
                Column name in the data frame for declination.
            weight_name:
                Optional column name in the data frame for weights.
            redshift_name:
                Optional column name in the data frame for redshifts.
            patch_centers:
                A list of patch centers to use when creating the patches. Can be
                either :obj:`~yaw.AngularCoordinates` or an other
                :obj:`~yaw.Catalog` as reference.
            patch_name:
                Optional column name in the data frame for a column with integer
                patch indices. Indices must be contiguous and starting from 0.
                Ignored if ``patch_centers`` is given.
            patch_num:
                Automatically compute patch centers from a sparse sample of the
                input data using `treecorr`. Requires an additional scan of the
                input file to read a sparse sampling of the object coordinates.
                Ignored if ``patch_centers`` or ``patch_name`` is given.
            degrees:
                Whether the input coordinates are given in degreees (default).
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).
            chunksize:
                The maximum number of records to load into memory at once when
                processing the input file in chunks.
            probe_size:
                The approximate number of records to read when generating
                patch centers (``patch_num``).

        Returns:
            A new catalog instance.

        Raises:
            FileExistsError:
                If the cache directory exists or is not a valid catalog when
                providing ``overwrite=True``.
        """
        write_catalog(
            cache_directory,
            source=dataframe,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_centers=patch_centers,
            patch_name=patch_name,
            patch_num=patch_num,
            degrees=degrees,
            chunksize=chunksize,
            probe_size=probe_size,
            overwrite=overwrite,
            progress=progress,
            max_workers=max_workers,
            **reader_kwargs,
        )

        if parallel.on_root():
            logger.info("computing patch metadata")
        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new._patches = load_patches(
            new.cache_directory,
            patch_centers=patch_centers,
            progress=progress,
            max_workers=max_workers,
        )
        return new

    @classmethod
    def from_file(
        cls,
        cache_directory: Path | str,
        path: Path | str,
        *,
        ra_name: str,
        dec_name: str,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        patch_centers: TypePatchCenters | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        overwrite: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
        chunksize: int | None = None,
        probe_size: int = -1,
        **reader_kwargs,
    ) -> Catalog:
        """
        Create a new catalog instance from a data file.

        Processes the input file in chunks, assign objects to spatial patches,
        write the patches to a cache on disk, and compute the patch meta data.
        Supported file formats are `FITS`, `Parquet`, and `HDF5`.

        .. note::
            One of the optional patch creation arguments (``patch_centers``,
            ``patch_name``, or ``patch_num``) must be provided.

        Args:
            cache_directory:
                The cache directory to use for this catalog. Created
                automatically or overwritten if requested.
            path:
                The path to the input data file.

        Keyword Args:
            ra_name:
                Column or path name in the file for right ascension.
            dec_name:
                Column or path name in the file for declination.
            weight_name:
                Optional column or path name in the file for weights.
            redshift_name:
                Optional column or path name in the file for redshifts.
            patch_centers:
                A list of patch centers to use when creating the patches. Can be
                either :obj:`~yaw.AngularCoordinates` or an other
                :obj:`~yaw.Catalog` as reference.
            patch_name:
                Optional column or path name for a column with integer patch
                indices. Indices must be contiguous and starting from 0.
                Ignored if ``patch_centers`` is given.
            patch_num:
                Automatically compute patch centers from a sparse sample of the
                input data using `treecorr`. Requires an additional scan of the
                input file to read a sparse sampling of the object coordinates.
                Ignored if ``patch_centers`` or ``patch_name`` is given.
            degrees:
                Whether the input coordinates are given in degreees (default).
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).
            chunksize:
                The maximum number of records to load into memory at once when
                processing the input file in chunks.
            probe_size:
                The approximate number of records to read when generating
                patch centers (``patch_num``).

        Returns:
            A new catalog instance.

        Raises:
            FileExistsError:
                If the cache directory exists or is not a valid catalog when
                providing ``overwrite=True``.

        Additional reader keyword arguments are passed on to the file reader
        class constuctor.
        """
        write_catalog(
            cache_directory,
            source=path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_centers=patch_centers,
            patch_name=patch_name,
            patch_num=patch_num,
            degrees=degrees,
            chunksize=chunksize,
            probe_size=probe_size,
            overwrite=overwrite,
            progress=progress,
            max_workers=max_workers,
            **reader_kwargs,
        )

        if parallel.on_root():
            logger.info("computing patch metadata")
        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new._patches = load_patches(
            new.cache_directory,
            patch_centers=patch_centers,
            progress=progress,
            max_workers=max_workers,
        )
        return new

    def __repr__(self) -> str:
        items = (
            f"num_patches={self.num_patches}",
            f"total={sum(self.get_totals())}",
            f"has_weights={self.has_weights}",
            f"has_redshifts={self.has_redshifts}",
        )
        return f"{type(self).__name__}({', '.join(items)}) @ {self.cache_directory}"

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self._patches[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self._patches.keys())

    @property
    def num_patches(self) -> int:
        """The number of patches of this catalog."""
        return len(self)

    @property
    def has_weights(self) -> bool:
        """Whether weights are available."""
        has_weights = tuple(patch.has_weights for patch in self.values())
        if all(has_weights):
            return True
        elif not any(has_weights):
            return False
        raise InconsistentPatchesError("'weights' not consistent")

    @property
    def has_redshifts(self) -> bool:
        """Whether redshifts are available."""
        has_redshifts = tuple(patch.has_redshifts for patch in self.values())
        if all(has_redshifts):
            return True
        elif not any(has_redshifts):
            return False
        raise InconsistentPatchesError("'redshifts' not consistent")

    def get_num_records(self) -> tuple[int]:
        """Get the number of records in each patches."""
        return tuple(patch.meta.num_records for patch in self.values())

    def get_totals(self) -> tuple[float]:
        """Get the sum of weights of the patches."""
        return tuple(patch.meta.total for patch in self.values())

    def get_centers(self) -> AngularCoordinates:
        """Get the center coordinates of the patches."""
        return AngularCoordinates.from_coords(
            patch.meta.center for patch in self.values()
        )

    def get_radii(self) -> AngularDistances:
        """Get the radii of the patches."""
        return AngularDistances.from_dists(patch.meta.radius for patch in self.values())

    def build_trees(
        self,
        binning: NDArray | None = None,
        *,
        closed: Closed | str = Closed.right,
        leafsize: int = 16,
        force: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
    ) -> None:
        """
        Build binary search trees on for each patch.

        The trees are cached in the patches' cache directory and can be
        retrieved from individual patches through :obj:`~yaw.Patch.get_trees()`.

        Args:
            binning:
                Optional array with redshift bin edges to apply to the data
                before building trees.

        Keyword Args:
            closed:
                Whether the bin edges are closed on the ``left`` or ``right``
                side.
            leafsize:
                Leafsize when building trees.
            force:
                Whether to overwrite any existing, cached trees.
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default). Takes precedence over the value in the configuration.
        """
        binning = parse_binning(binning, optional=True)
        closed = Closed(closed)  # parse early for error checks

        if parallel.on_root():
            logger.debug(
                "building patch-wise trees (%s)",
                "unbinned" if binning is None else f"using {len(binning) - 1} bins",
            )

        patch_tree_iter = parallel.iter_unordered(
            BinnedTrees.build,
            self.values(),
            func_args=(binning,),
            func_kwargs=dict(closed=str(closed), leafsize=leafsize, force=force),
            max_workers=max_workers,
        )
        if progress:
            patch_tree_iter = Indicator(patch_tree_iter, len(self))

        deque(patch_tree_iter, maxlen=0)


Catalog.get.__doc__ = "Return the :obj:`~yaw.Patch` for ID if exists, else default."
Catalog.keys.__doc__ = "A set-like object providing a view of all patch IDs."
Catalog.values.__doc__ = (
    "A set-like object providing a view of all :obj:`~yaw.Patch` es."
)
Catalog.items.__doc__ = "A set-like object providing a view of `(key, value)` pairs."
