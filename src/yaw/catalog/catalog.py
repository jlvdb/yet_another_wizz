"""
Implements data catalogs, which are the centeral container for row-data and
facilitate correlation measurements.

Catalogs are dictionary-like collections of patches, which each hold a portion
of the catalog data. Data is not permanently held in memory, instead each
catalog is tied to a cache directory on disk. To retrive data, access a patch
and manually load the data from its cache. This design allows flexibility while
minimising the memory footprint of large datasets.

Catalogs can be constructed directly from input files or random generators.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Mapping
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from shutil import rmtree
from typing import TYPE_CHECKING

import numpy as np
import treecorr
from scipy.cluster import vq

from yaw.binning import Binning
from yaw.catalog.patch import Patch, PatchWriter
from yaw.catalog.readers import (
    DataChunkReader,
    DataFrameReader,
    RandomReader,
    new_filereader,
)
from yaw.catalog.trees import BinnedTrees, groupby
from yaw.coordinates import AngularCoordinates, AngularDistances
from yaw.datachunk import (
    PATCH_ID_DTYPE,
    DataChunk,
    DataChunkInfo,
    HandlesDataChunk,
    check_patch_ids,
)
from yaw.options import Closed
from yaw.randoms import RandomsBase
from yaw.utils import format_long_num, parallel
from yaw.utils.logging import Indicator
from yaw.utils.parallel import EndOfQueue

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray
    from typing_extensions import Self

    from yaw.catalog.readers import DataFrame
    from yaw.datachunk import TypeDataChunk, TypePatchIDs

__all__ = [
    "Catalog",
    "create_patch_centers",
    "assign_patch_centers",
    "load_patches",
    "write_patches",
]


PATCH_NAME_TEMPLATE = "patch_{:d}"
"""Template to name patch directories in catalog cache directory."""

PATCH_INFO_FILE = "patch_ids.bin"
"""Name of file listing patch IDs in catalog cache directory."""

logger = logging.getLogger(__name__)


class InconsistentPatchesError(Exception):
    pass


class PatchMode(Enum):
    """Enumeration to specify the patch creation method."""

    apply = 0
    divide = 1
    create = 2

    @classmethod
    def determine(
        cls,
        patch_centers: AngularCoordinates | Catalog | None,
        patch_name: str | None,
        patch_num: int | None,
    ) -> PatchMode:
        """
        Determine the patch creation method to use.

        Method is determined from the three possible input parameters in the
        :obj:`~yaw.Catalog` creation routines, by checking which of the
        parameters are set in the following order of precedence:
        ``patch_centers`` > ``patch_name`` > ``patch_num``

        Args:
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

        Returns:
            The Enum value indicating which patch creation method to use.

        Raises:
            TypeError:
                If the input values are of an invalid type.
            ValueError:
                If the number of patches exceeds the maximum allowed number or
                none of the input parameters are provided.
        """
        log_sink = logger.debug if parallel.on_root() else lambda *x: x

        if patch_centers is not None:
            if not isinstance(patch_centers, (AngularCoordinates, Catalog)):
                raise TypeError(
                    "'patch_centers' must be a set of coordinates or another catalog"
                )
            check_patch_ids(len(patch_centers))

            log_sink("applying %d patches", len(patch_centers))
            return PatchMode.apply

        if patch_name is not None:
            if not isinstance(patch_name, str):
                raise TypeError("'patch_name' must be a string")

            log_sink("dividing patches based on '%s'", patch_name)
            return PatchMode.divide

        elif patch_num is not None:
            if not isinstance(patch_num, int):
                raise TypeError("'patch_num' must be an integer")
            check_patch_ids(patch_num)

            log_sink("creating %d patches", patch_num)
            return PatchMode.create

        raise ValueError("no patch method specified")


def get_patch_centers(instance: AngularCoordinates | Catalog) -> AngularCoordinates:
    """Extract the patch centers from a set of angular coordinates or a catalog
    instance, raises ``TypeError`` otherwise."""
    try:
        return instance.get_centers()
    except AttributeError as err:
        if isinstance(instance, AngularCoordinates):
            return instance
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        ) from err


def create_patch_centers(
    reader: DataChunkReader, patch_num: int, probe_size: int
) -> AngularCoordinates:
    """
    Automatically create new patch centers from a data source.

    Data source can be a file reader or random generator. Patch centers are
    computed from a small data subset using ``treecorr`` for optimal efficiency.

    Args:
        reader:
            A :obj:`DataChunkReader` instance that exposes a random generator or
            file reader.
        patch_num:
            The number of patches to create.
        probe_size:
            The size of the subsample from which the patches are computed.

    Returns:
        A new set of angular coordinates of the patch centers.
    """
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    if parallel.on_root():
        logger.info(
            "computing patch centers from subset of %s records",
            format_long_num(probe_size),
        )

    data_probe = reader.get_probe(probe_size)

    patch_centers = None
    if parallel.on_root():
        cat = treecorr.Catalog(
            ra=DataChunk.getattr(data_probe, "ra"),
            ra_units="radians",
            dec=DataChunk.getattr(data_probe, "dec"),
            dec_units="radians",
            w=DataChunk.getattr(data_probe, "weights", None),
            npatch=patch_num,
            config=dict(num_threads=parallel.get_size()),
        )
        patch_centers = AngularCoordinates.from_3d(cat.patch_centers)
    return parallel.COMM.bcast(patch_centers, root=0)


def assign_patch_centers(patch_centers: NDArray, data: TypeDataChunk) -> TypePatchIDs:
    """
    Computes the patch ID for a set of objects and patch center coordinates.

    Objects are assigned to the nearest patch center, expressed by the index of
    the patch center in the input list of patch centers.

    Args:
        patch_centers:
            Numpy array of patch center coordinates in radian and shape
            `(N, 2)`.
        data:
            Numpy array holding input object coordinates, i.e. a chunk of
            catalog data, must contain with fields ``ra`` and ``dec``.

    Returns:
        Array of 16-bit integer patch IDs for each input obejct.
    """
    coords = DataChunk.get_coords(data)
    ids, _ = vq.vq(coords.to_3d(), patch_centers)
    return ids.astype(PATCH_ID_DTYPE)


def split_into_patches(
    chunk: TypeDataChunk, patch_centers: NDArray | None
) -> dict[int, TypeDataChunk]:
    """
    Split a numpy array of catalog data into patches.

    If patch centers are provided, assigns patch IDs from nearest patch center.
    If a patch ID column is contained in the input data, uses that to assign
    objects to patches.

    Args:
        chunk:
            Numpy array holding input object coordinates, i.e. a chunk of
            catalog data, must contain with fields ``ra`` and ``dec``, and
            optionally ``patch_ids``.
        patch_centers:
            Optional, numpy array of patch center coordinates in radian and
            shape `(N, 2)`.

    Returns:
        Dictionary with patch IDs as keys and subset of input data chunk with
        objects belonging to the corresponding patch ID.

    Raises:
        RuntimeError:
            If neither patch centers nor patch IDs per object are provided.
    """
    has_patch_ids = DataChunk.hasattr(chunk, "patch_ids")

    # statement order matters
    if patch_centers is not None:
        patch_ids = assign_patch_centers(patch_centers, chunk)
        if has_patch_ids:
            chunk, _ = DataChunk.pop(chunk, "patch_ids")
    elif has_patch_ids:
        # patch IDs will be redundant information so we delete them
        chunk, patch_ids = DataChunk.pop(chunk, "patch_ids")
    else:  # pragma: no cover
        raise RuntimeError("found no way to obtain patch centers")

    return {
        int(patch_id): patch_data for patch_id, patch_data in groupby(patch_ids, chunk)
    }


def get_patch_path_from_id(cache_directory: Path | str, patch_id: int) -> Path:
    """
    Get the patch to a specific patch cache directory.

    Args:
        cache_directory:
            The cache directory used by the parent catalog.
        patch_id:
            ID of the patch for which to optain the patch cache directory.

    Returns:
        Path as a :obj:`pathlib.Path`.
    """
    return Path(cache_directory) / PATCH_NAME_TEMPLATE.format(patch_id)


def get_id_from_patch_path(cache_path: Path | str) -> int:
    """
    Extract the integer patch ID from a patch cache path.

    .. caution::
        This will fail if the patch has not been created through a
        :obj:`CatalogWriter` instance, which manages the patch creation.
    """
    _, id_str = Path(cache_path).name.split("_")
    return int(id_str)


def read_patch_ids(cache_directory: Path) -> list[int]:
    """Reads a list of patch IDs in a catalog from a metadata file stored in the
    catalog's cache directory."""
    path = cache_directory / PATCH_INFO_FILE
    if not path.exists():
        raise InconsistentPatchesError("patch info file not found")
    return np.fromfile(path, dtype=PATCH_ID_DTYPE).tolist()


def load_patches(
    cache_directory: Path,
    *,
    patch_centers: AngularCoordinates | Catalog | None,
    progress: bool,
    max_workers: int | None = None,
) -> dict[int, Patch]:
    """
    Instantiate all patches stored in a catalog's cache directory.

    Function is MPI aware, patches are loaded on the root worker and broadcasted
    to all workers. Computes patch metadata if not present. If patch centers
    are provided, only the patch radius is computed but not the patch centers.

    Args:
        cache_directory:
            Cache directory of the parent catalog.

    Keyword Args:
        patch_centers:
            Optional set of angular coordinates or catalog instance that
            defines the exact patch centers to use.
        progress:
            Show a progress on the terminal (disabled by default).
        max_workers:
            Limit the  number of parallel workers for this operation (all by
            default).
    """
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

    patches = {get_id_from_patch_path(patch.cache_path): patch for patch in patch_iter}
    return parallel.COMM.bcast(patches, root=0)


class CatalogWriter(AbstractContextManager, HandlesDataChunk):
    """
    A helper class that handles a stream of input catalog data and splits and
    writes it to patches.

    Args:
        cache_directory:
            Cache directory of the catalog.

    Keyword Args:
        overwrite:
            Whether to overwrite an existing catalog at the given cache
            location.
        chunk_info:
            An instance of :obj:`yaw.datachunk.DataChunkInfo` indicating which
            optional data attributes are processed by the pipeline.
        buffersize:
            Optional, maximum number of records to store in the internal cache
            of each patch writer.

    Attributes:
        cache_directory:
            Cache directory to use when creating the patches.
        writers:
            Dictionary of patch IDs / :obj:`~yaw.catalog.patch.PatchWriters`
            that delegates writing data for an individual patch.
        buffersize:
            Optional, maximum number of records to store in the internal cache
            of each patch writer.

    Raises:
        FileExistsError:
            If the cache directory already exists and ``overwrite==False``.
    """

    __slots__ = (
        "_chunk_info",
        "cache_directory",
        "buffersize",
        "writers",
    )

    def __init__(
        self,
        cache_directory: Path | str,
        *,
        chunk_info: DataChunkInfo,
        overwrite: bool = True,
        buffersize: int = -1,
    ) -> None:
        self._chunk_info = chunk_info
        self.cache_directory = Path(cache_directory)
        cache_exists = self.cache_directory.exists()

        if parallel.on_root():
            logger.info(
                "%s cache directory: %s",
                "overwriting" if cache_exists and overwrite else "using",
                cache_directory,
            )

        if self.cache_directory.exists():
            if overwrite:
                rmtree(self.cache_directory)
            else:
                raise FileExistsError(f"cache directory exists: {cache_directory}")

        self.buffersize = buffersize
        self.cache_directory.mkdir()
        self.writers: dict[int, PatchWriter] = {}

    def __repr__(self) -> str:
        items = (
            f"num_patches={self.num_patches}",
            f"max_buffersize={self.buffersize * self.num_patches}",
        )
        attrs = self._chunk_info.format()
        return f"{type(self).__name__}({', '.join(items)}, {attrs}) @ {self.cache_directory}"

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalize()

    @property
    def num_patches(self) -> int:
        """The number of unique patch IDs encountered so far."""
        return len(self.writers)

    def get_writer(self, patch_id: int) -> PatchWriter:
        """Get the patch writer for the given patch ID and create it if it does
        not yet exist."""
        try:
            return self.writers[patch_id]

        except KeyError:
            writer = PatchWriter(
                get_patch_path_from_id(self.cache_directory, patch_id),
                chunk_info=self.copy_chunk_info(),
                buffersize=self.buffersize,
            )
            self.writers[patch_id] = writer
            return writer

    def process_patches(self, patches: dict[int, TypeDataChunk]) -> None:
        """
        Process a dictionary of catalog data split into patches.

        Dictionary values are sent to the individual patch data writers which
        cache the data in memory temporarily or write them to disk.

        Args:
            patches:
                A dictionary of patch ID / numpy array with catalog data
                (containing ``ra``, ``dec``, or optionally ``weights`` and
                ``redshifts`` fields).
        """
        for patch_id, patch in patches.items():
            self.get_writer(patch_id).process_chunk(patch)

    def finalize(self) -> None:
        """
        Finalise the catalog cache directory.

        Flushes all patch writer caches and writes a list of patch IDs to
        the cache directory that simplifes loading the catalog instance later.

        Raises:
            ValueError:
                If any of the patches does not contain any data.
        """
        empty_patches = set()
        for patch_id, writer in self.writers.items():
            writer.close()
            if writer.num_processed == 0:
                empty_patches.add(patch_id)

        for patch_id in empty_patches:
            raise ValueError(f"patch with ID {patch_id} contains no data")

        patch_ids = np.fromiter(self.writers.keys(), dtype=np.int16)
        np.sort(patch_ids).tofile(self.cache_directory / PATCH_INFO_FILE)


def write_patches_unthreaded(
    path: Path | str,
    reader: DataChunkReader,
    patch_centers: AngularCoordinates | Catalog | None,
    *,
    overwrite: bool,
    progress: bool,
    buffersize: int = -1,
) -> None:
    """
    Read catalog from an input source and write the data to catalog cache
    directory.

    Creates patch centers automatically from data source if none are provided.
    This is a fallback implementation if parallel workers are disabled.

    Args:
        path:
            The target cache directory.
        reader:
            A :obj:`DataChunkReader` instance that exposes a random generator or
            file reader.
        patch_centers:
            Optional set of angular coordinates or catalog instance that
            defines the exact patch centers to use.

    Keyword Args:
        overwrite:
            Whether to overwrite an existing catalog at the given cache
            location. If the directory is not a valid catalog, a
            ``FileExistsError`` is raised.
        progress:
            Show a progress on the terminal (disabled by default).
        buffersize:
            Optional, maximum number of records to store in the internal cache
            of each patch writer.

    """
    with reader:
        if patch_centers is not None:
            patch_centers = get_patch_centers(patch_centers).to_3d()

        with CatalogWriter(
            cache_directory=path,
            chunk_info=reader.copy_chunk_info(),
            overwrite=overwrite,
            buffersize=buffersize,
        ) as writer:
            chunk_iter = Indicator(reader) if progress else iter(reader)
            for chunk in chunk_iter:
                patches = split_into_patches(chunk, patch_centers)
                writer.process_patches(patches)


if parallel.use_mpi():
    """Implementation of parallel input data processing based on OpenMPI."""

    from mpi4py import MPI

    if TYPE_CHECKING:
        from mpi4py.MPI import Comm

    class WorkerManager:
        """Contains information required by the MPI workers to coordinate
        parallel processing: rank that is responsible for reading, rank that is
        responsible for writing and which ranks are responsible for processing
        chunk data in parallel."""

        def __init__(self, max_workers: int | None, reader_rank: int = 0) -> None:
            self.reader_rank = reader_rank

            max_workers = parallel.get_size(max_workers)
            self.active_ranks = parallel.ranks_on_same_node(reader_rank, max_workers)

            self.active_ranks.discard(reader_rank)
            self.writer_rank = self.active_ranks.pop()
            self.active_ranks.add(reader_rank)

        def get_comm(self) -> Comm:
            rank = parallel.COMM.Get_rank()
            if rank in self.active_ranks:
                return parallel.COMM.Split(1, rank)
            else:
                return parallel.COMM.Split(MPI.UNDEFINED, rank)

    def scatter_data_chunk(comm: Comm, reader_rank: int, chunk: DataChunk) -> DataChunk:
        """Takes a chunk of catalog data, splits it into chunks and broadcasts
        the chunks to the parallel chunk processing tasks."""
        num_ranks = comm.Get_size()

        if comm.Get_rank() == reader_rank:
            splits = np.array_split(chunk, num_ranks)

            for rank, split in enumerate(splits):
                if rank != reader_rank:
                    comm.send(split, dest=rank, tag=2)

            return splits[reader_rank]

        else:
            return comm.recv(source=0, tag=2)

    def chunk_processing_task(
        comm: Comm,
        worker_config: WorkerManager,
        patch_centers: AngularCoordinates | Catalog | None,
        chunk_iter: Iterator[DataChunk],
    ) -> None:
        """A dedicated parallel worker task which splits catalog data into
        paches and sends the data to the writer process."""
        if patch_centers is not None:
            patch_centers = patch_centers.to_3d()

        reader_rank = parallel.world_to_comm_rank(comm, worker_config.reader_rank)

        for chunk in chunk_iter:
            worker_chunk = scatter_data_chunk(comm, reader_rank, chunk)
            patches = split_into_patches(worker_chunk, patch_centers)
            parallel.COMM.send(patches, dest=worker_config.writer_rank, tag=1)

        comm.Barrier()

    def writer_task(
        cache_directory: Path | str,
        *,
        chunk_info: DataChunkInfo,
        overwrite: bool = True,
        buffersize: int = -1,
    ) -> None:
        """A dedicated writer process that recieves a dictionary with patch IDs
        and patch data to write using a :obj:`CatalogWriter`, terminated when
        receiving :obj:`EndOfQueue` sentinel."""
        recv = parallel.COMM.recv
        with CatalogWriter(
            cache_directory,
            chunk_info=chunk_info,
            overwrite=overwrite,
            buffersize=buffersize,
        ) as writer:
            while (patches := recv(source=MPI.ANY_SOURCE, tag=1)) is not EndOfQueue:
                writer.process_patches(patches)

    def write_patches(
        path: Path | str,
        reader: DataChunkReader,
        patch_centers: AngularCoordinates | Catalog | None,
        *,
        overwrite: bool,
        progress: bool,
        max_workers: int | None = None,
        buffersize: int = -1,
    ) -> None:
        """
        Read catalog from an input source and write the data to catalog cache
        directory.

        Creates patch centers automatically from data source if none are
        provided. This is an implementation with MPI parallelsim. The root rank
        is responsible for reading data from the source, one rank is responsible
        for writing to the cache directory, any remaining ranks process the
        input data.

        .. Note::
            The code tries to schedule all work only on the same node that
            hosts the root tasks to avoid inter-node communication.

        Args:
            path:
                The target cache directory.
            reader:
                A :obj:`DataChunkReader` instance that exposes a random
                generator or file reader.
            patch_centers:
                Optional set of angular coordinates or catalog instance that
                defines the exact patch centers to use.

        Keyword Args:
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).
            buffersize:
                Optional, maximum number of records to store in the internal
                cache of each patch writer.
        """
        max_workers = parallel.get_size(max_workers)
        if max_workers < 2:
            raise ValueError("catalog creation requires at least two workers")
        if parallel.on_root():
            logger.debug("running preprocessing on %d workers", max_workers)

        rank = parallel.COMM.Get_rank()
        worker_config = WorkerManager(max_workers, 0)
        worker_comm = worker_config.get_comm()

        if rank == worker_config.writer_rank:
            writer_task(
                cache_directory=path,
                chunk_info=reader.copy_chunk_info(),
                overwrite=overwrite,
                buffersize=buffersize,
            )

        elif rank in worker_config.active_ranks:
            if patch_centers is not None:
                patch_centers = get_patch_centers(patch_centers)

            with reader:
                chunk_iter = Indicator(reader) if progress else iter(reader)
                chunk_processing_task(
                    worker_comm,
                    worker_config,
                    patch_centers,
                    chunk_iter,
                )

            worker_comm.Free()

        if parallel.COMM.Get_rank() == worker_config.reader_rank:
            parallel.COMM.send(EndOfQueue, dest=worker_config.writer_rank, tag=1)
        parallel.COMM.Barrier()

else:
    """Implementation of parallel input data processing based on python's
    multiprocessing."""

    import multiprocessing
    from dataclasses import dataclass, field

    if TYPE_CHECKING:
        from multiprocessing import Queue

    class ChunkProcessingTask:
        """Defines the worker task which splits catalog data into paches and
        puts the data into the writer process queue."""

        def __init__(
            self,
            patch_queue: Queue[dict[int, TypeDataChunk] | EndOfQueue],
            patch_centers: AngularCoordinates | None,
        ) -> None:
            self.patch_queue = patch_queue

            if isinstance(patch_centers, AngularCoordinates):
                self.patch_centers = patch_centers.to_3d()
            else:
                self.patch_centers = None

        def __call__(self, chunk: DataChunk) -> dict[int, TypeDataChunk]:
            patches = split_into_patches(chunk, self.patch_centers)
            self.patch_queue.put(patches)

    @dataclass
    class WriterProcess(AbstractContextManager):
        """A dedicated writer process that recieves a dictionary with patch IDs
        and patch data to write using a :obj:`CatalogWriter`, terminated when
        receiving :obj:`EndOfQueue` sentinel."""

        patch_queue: Queue[dict[int, TypeDataChunk] | EndOfQueue]
        cache_directory: Path | str
        chunk_info: DataChunkInfo = field(kw_only=True)
        overwrite: bool = field(default=True, kw_only=True)
        buffersize: int = field(default=-1, kw_only=True)

        def __post_init__(self) -> None:
            self.process = multiprocessing.Process(target=self.task)

        def __enter__(self) -> Self:
            self.start()
            return self

        def __exit__(self, *args, **kwargs) -> None:
            self.join()

        def task(self) -> None:
            with CatalogWriter(
                self.cache_directory,
                overwrite=self.overwrite,
                chunk_info=self.chunk_info,
                buffersize=self.buffersize,
            ) as writer:
                while (patches := self.patch_queue.get()) is not EndOfQueue:
                    writer.process_patches(patches)

        def start(self) -> None:
            self.process.start()

        def join(self) -> None:
            self.process.join()

    def write_patches(
        path: Path | str,
        reader: DataChunkReader,
        patch_centers: AngularCoordinates | Catalog | None,
        *,
        overwrite: bool,
        progress: bool,
        max_workers: int | None = None,
        buffersize: int = -1,
    ) -> None:
        """
        Read catalog from an input source and write the data to catalog cache
        directory.

        Creates patch centers automatically from data source if none are
        provided. This is an implementation with MPI parallelsim. There is a
        dedicated process that handles writing data to the catalog cache
        directory.

        Args:
            path:
                The target cache directory.
            reader:
                A :obj:`DataChunkReader` instance that exposes a random
                generator or file reader.
            patch_centers:
                Optional set of angular coordinates or catalog instance that
                defines the exact patch centers to use.

        Keyword Args:
            overwrite:
                Whether to overwrite an existing catalog at the given cache
                location. If the directory is not a valid catalog, a
                ``FileExistsError`` is raised.
            progress:
                Show a progress on the terminal (disabled by default).
            max_workers:
                Limit the  number of parallel workers for this operation (all by
                default).
            buffersize:
                Optional, maximum number of records to store in the internal
                cache of each patch writer.
        """
        max_workers = parallel.get_size(max_workers)

        if max_workers == 1:
            logger.debug("running preprocessing sequentially")
            return write_patches_unthreaded(
                path,
                reader,
                patch_centers,
                overwrite=overwrite,
                progress=progress,
                buffersize=buffersize,
            )

        else:
            logger.debug("running preprocessing on %d workers", max_workers)

        with (
            reader,
            multiprocessing.Manager() as manager,
            multiprocessing.Pool(max_workers) as pool,
        ):
            patch_queue = manager.Queue()

            if patch_centers is not None:
                patch_centers = get_patch_centers(patch_centers)
            chunk_processing_task = ChunkProcessingTask(patch_queue, patch_centers)

            with WriterProcess(
                patch_queue,
                cache_directory=path,
                chunk_info=reader.copy_chunk_info(),
                overwrite=overwrite,
                buffersize=buffersize,
            ):
                chunk_iter = Indicator(reader) if progress else iter(reader)
                for chunk in chunk_iter:
                    pool.map(chunk_processing_task, np.array_split(chunk, max_workers))

                patch_queue.put(EndOfQueue)


class Catalog(Mapping[int, Patch]):
    """
    A container for catalog data.

    Catalogs are the core data structure for managing point data catalogs.
    Besides right ascension and declination coordinates, catalogs may have
    additional per-object weights and redshifts.

    Catalogs divided into spatial :obj:`~yaw.catalog.Patch` es, which each cache
    a portion of the data on disk to minimise the memory footprint when dealing
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

    .. caution::
        Empty patches are currently not supported and the catalog creation will
        fail if a patch without any data is encountered (e.g. if the input
        catalog is too sparse or inhomogeneous).

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
        patch_centers: AngularCoordinates | Catalog | None = None,
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
        reader = DataFrameReader(
            dataframe,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
            **reader_kwargs,
        )
        mode = PatchMode.determine(patch_centers, patch_name, patch_num)
        if mode == PatchMode.create:
            patch_centers = create_patch_centers(reader, patch_num, probe_size)

        # split the data into patches and create the cached Patch instances.
        write_patches(
            cache_directory,
            reader,
            patch_centers,
            overwrite=overwrite,
            progress=progress,
            max_workers=max_workers,
            buffersize=-1,
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
        patch_centers: AngularCoordinates | Catalog | None = None,
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
        reader = new_filereader(
            path,
            ra_name=ra_name,
            dec_name=dec_name,
            weight_name=weight_name,
            redshift_name=redshift_name,
            patch_name=patch_name,
            chunksize=chunksize,
            degrees=degrees,
            **reader_kwargs,
        )
        mode = PatchMode.determine(patch_centers, patch_name, patch_num)
        if mode == PatchMode.create:
            patch_centers = create_patch_centers(reader, patch_num, probe_size)

        # split the data into patches and create the cached Patch instances.
        write_patches(
            cache_directory,
            reader,
            patch_centers,
            overwrite=overwrite,
            progress=progress,
            max_workers=max_workers,
            buffersize=-1,
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
    def from_random(
        cls,
        cache_directory: Path | str,
        generator: RandomsBase,
        num_randoms: int,
        *,
        patch_centers: AngularCoordinates | Catalog | None = None,
        patch_num: int | None = None,
        overwrite: bool = False,
        progress: bool = False,
        max_workers: int | None = None,
        chunksize: int | None = None,
        probe_size: int = -1,
    ) -> Catalog:
        """
        Create a new catalog instance from a data file.

        Generate a catalog from uniform random data points in chunks, assign
        objects to spatial patches, write the patches to a cache on disk, and
        compute the patch meta data.

        The :ref:`generator object<generator>` must be created separately by the
        user.

        .. note::
            One of the optional patch creation arguments (``patch_centers``, or
            ``patch_num``) must be provided (``patch_name`` is not supported).

        Args:
            cache_directory:
                The cache directory to use for this catalog. Created
                automatically or overwritten if requested.
            generator:
                A random generator (:obj:`~yaw.catalog.generator.RandomsBase`)
                instance from which samples are drawn.
            num_randoms:
                The number of randoms to generate.

        Keyword Args:
            patch_centers:
                A list of patch centers to use when creating the patches. Can be
                either :obj:`~yaw.AngularCoordinates` or an other
                :obj:`~yaw.Catalog` as reference.
            patch_num:
                Automatically compute patch centers from a sparse sample of the
                input data using `treecorr`. Requires an additional scan of the
                input file to read a sparse sampling of the object coordinates.
                Ignored if ``patch_centers`` or ``patch_name`` is given.
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
                The maximum number of records to generate and write at once.
            probe_size:
                The number of initial random samples to draw read when
                generating patch centers (``patch_num``).

        Returns:
            A new catalog instance.

        Raises:
            FileExistsError:
                If the cache directory exists or is not a valid catalog when
                providing ``overwrite=True``.
        """
        rand_iter = RandomReader(generator, num_randoms, chunksize)

        mode = PatchMode.determine(patch_centers, None, patch_num)
        if mode == PatchMode.create:
            patch_centers = create_patch_centers(rand_iter, patch_num, probe_size)

        # split the data into patches and create the cached Patch instances.
        write_patches(
            cache_directory,
            rand_iter,
            patch_centers,
            overwrite=overwrite,
            progress=progress,
            max_workers=max_workers,
            buffersize=-1,
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
            f"num_records={sum(self.get_num_records())}",
        )
        patch = next(iter(self.values()))
        attrs = patch._chunk_info.format()
        return f"{type(self).__name__}({', '.join(items)}, {attrs}) @ {self.cache_directory}"

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

    def get_num_records(self) -> tuple[int, ...]:
        """Get the number of records in each patches."""
        return tuple(patch.meta.num_records for patch in self.values())

    def get_sum_weights(self) -> tuple[float, ...]:
        """Get the sum of weights of the patches."""
        return tuple(patch.meta.sum_weights for patch in self.values())

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
        retrieved through ``yaw.trees.BinnedTrees(patch)``.

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
        if binning is not None:
            binning = Binning(binning, closed=closed)

        if parallel.on_root():
            logger.debug(
                "building patch-wise trees (%s)",
                "unbinned" if binning is None else f"using {len(binning)} bins",
            )

        patch_tree_iter = parallel.iter_unordered(
            BinnedTrees.build,
            self.values(),
            func_args=(binning,),
            func_kwargs=dict(leafsize=leafsize, force=force),
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
