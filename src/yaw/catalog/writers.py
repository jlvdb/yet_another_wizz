from __future__ import annotations

import logging
import multiprocessing
from collections.abc import Callable
from contextlib import AbstractContextManager
from enum import Enum
from functools import partial
from itertools import repeat
from pathlib import Path
from shutil import rmtree
from typing import Protocol, Union, get_args

import numpy as np
import treecorr
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.cluster import vq
from typing_extensions import Self

from yaw.catalog.readers import BaseReader, DataFrameReader, new_filereader
from yaw.catalog.utils import DATA_PATH, PATCH_NAME_TEMPLATE, DataChunk
from yaw.catalog.utils import MockDataFrame as DataFrame
from yaw.catalog.utils import write_column_info
from yaw.containers import Tpath
from yaw.utils import AngularCoordinates, parallel
from yaw.utils.logging import Indicator
from yaw.utils.parallel import EndOfQueue

__all__ = [
    "create_patches",
]

Tcenters = Union["HasPatchCenters", AngularCoordinates]

CHUNKSIZE = 65_536

logger = logging.getLogger(__name__)


class HasPatchCenters(Protocol):
    def get_patch_centers() -> AngularCoordinates: ...


class PatchWriter:
    __slots__ = ("cache_path", "buffersize", "_cachesize", "_shards", "_file")

    def __init__(
        self,
        cache_path: Tpath,
        *,
        has_weights: bool,
        has_redshifts: bool,
        buffersize: int = -1,
    ) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)
        self._file = None

        write_column_info(cache_path, has_weights, has_redshifts)

        self.buffersize = CHUNKSIZE if buffersize < 0 else int(buffersize)
        self._cachesize = 0
        self._shards = []

    def open(self) -> None:
        if self._file is None:
            self._file = open(self.cache_path / DATA_PATH, mode="ab")

    def close(self) -> None:
        self.flush()
        self._file.close()
        self._file = None

    def process_chunk(self, chunk: DataChunk) -> None:
        self._shards.append(chunk.data)
        self._cachesize += len(chunk)

        if self._cachesize >= self.buffersize:
            self.flush()

    def flush(self) -> None:
        if len(self._shards) > 0:
            self.open()  # ensure file is ready for writing

            data = np.concatenate(self._shards)
            self._shards = []

            data.tofile(self._file)

        self._cachesize = 0


class PatchMode(Enum):
    apply = 0
    divide = 1
    create = 2

    @classmethod
    def determine(
        cls,
        patch_centers: Tcenters | None,
        patch_name: str | None,
        patch_num: int | None,
    ) -> PatchMode:
        log_sink = logger.debug if parallel.on_root() else lambda *x: x

        if patch_centers is not None:
            log_sink("applying patch %d centers", len(patch_centers))
            return PatchMode.apply

        if patch_name is not None:
            log_sink("dividing patches based on '%s'", patch_name)
            return PatchMode.divide

        elif patch_num is not None:
            log_sink("creating %d patches", patch_num)
            return PatchMode.create

        raise ValueError("no patch method specified")


def create_patch_centers(
    reader: BaseReader, patch_num: int, probe_size: int
) -> AngularCoordinates:
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    sparse_factor = np.ceil(reader.num_records / probe_size)
    test_sample = reader.read(int(sparse_factor))

    if parallel.on_root():
        logger.info("computing patch centers from %dx sparse sampling", sparse_factor)

    cat = treecorr.Catalog(
        ra=test_sample.ra,
        ra_units="radians",
        dec=test_sample.dec,
        dec_units="radians",
        w=test_sample.weights,
        npatch=patch_num,
        config=dict(num_threads=parallel.get_size()),
    )
    xyz = np.atleast_2d(cat.patch_centers)
    return AngularCoordinates.from_3d(xyz)


class ChunkProcessor:
    __slots__ = ("patch_centers",)

    def __init__(self, patch_centers: AngularCoordinates | None) -> None:
        if patch_centers is None:
            self.patch_centers = None
        else:
            self.patch_centers = patch_centers.to_3d()

    def _compute_patch_ids(self, chunk: DataChunk) -> NDArray[np.int32]:
        patches, _ = vq.vq(chunk.coords.to_3d(), self.patch_centers)
        return patches.astype(np.int32, copy=False)

    def execute(self, chunk: DataChunk) -> dict[int, DataChunk]:
        if self.patch_centers is not None:
            patch_ids = self._compute_patch_ids(chunk)
            chunk.set_patch_ids(patch_ids)

        return chunk.split_patches()

    def execute_send(self, queue: multiprocessing.Queue, chunk: DataChunk) -> None:
        queue.put(self.execute(chunk))


class CatalogWriter(AbstractContextManager):
    def __init__(
        self,
        cache_directory: Tpath,
        *,
        overwrite: bool = True,
        has_weights: bool,
        has_redshifts: bool,
        buffersize: int = -1,
    ) -> None:
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

        self.has_weights = has_weights
        self.has_redshifts = has_redshifts

        self.buffersize = buffersize
        self.cache_directory.mkdir()
        self._writers: dict[int, PatchWriter] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalize()

    def get_writer_path(self, patch_id: int) -> Path:
        return self.cache_directory / PATCH_NAME_TEMPLATE.format(patch_id)

    def get_writer(self, patch_id: int) -> PatchWriter:
        try:
            return self._writers[patch_id]

        except KeyError:
            writer = PatchWriter(
                self.get_writer_path(patch_id),
                has_weights=self.has_weights,
                has_redshifts=self.has_redshifts,
                buffersize=self.buffersize,
            )
            self._writers[patch_id] = writer
            return writer

    def process_patches(self, patches: dict[int, DataChunk]) -> None:
        for patch_id, patch in patches.items():
            self.get_writer(patch_id).process_chunk(patch)

    def finalize(self) -> None:
        for writer in self._writers.values():
            writer.close()


def get_patch_centers(instance: Tcenters) -> AngularCoordinates:
    try:
        return instance.get_centers()
    except AttributeError as err:
        if isinstance(instance, AngularCoordinates):
            return instance
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        ) from err


def _writer_process(
    get_method: Callable[[], dict[int, DataChunk] | EndOfQueue],
    cache_directory: Tpath,
    *,
    overwrite: bool = True,
    has_weights: bool,
    has_redshifts: bool,
    buffersize: int = -1,
) -> None:
    writer = CatalogWriter(
        cache_directory,
        overwrite=overwrite,
        has_weights=has_weights,
        has_redshifts=has_redshifts,
        buffersize=buffersize,
    )
    with writer:
        while True:
            patches = get_method()
            if patches is EndOfQueue:
                break
            writer.process_patches(patches)


def write_patches_multiprocessing(
    path: Tpath,
    reader: BaseReader,
    patch_centers: Tcenters,
    *,
    overwrite: bool,
    progress: bool,
    max_workers: int | None = None,
    buffersize: int = -1,
) -> None:

    patch_centers = get_patch_centers(patch_centers)
    preprocess = ChunkProcessor(patch_centers)

    manager = multiprocessing.Manager()
    max_workers = parallel.get_size(max_workers)
    pool = multiprocessing.Pool(max_workers)

    with reader, manager, pool:
        patch_queue = manager.Queue()

        writer = multiprocessing.Process(
            target=_writer_process,
            kwargs=dict(
                get_method=patch_queue.get,
                cache_directory=path,
                overwrite=overwrite,
                has_weights=reader.has_weights,
                has_redshifts=reader.has_redshifts,
                buffersize=buffersize,
            ),
        )
        writer.start()

        chunk_iter = iter(reader)
        if progress:
            chunk_iter = Indicator(reader)

        for chunk in chunk_iter:
            pool.starmap(
                preprocess.execute_send,
                zip(repeat(patch_queue), chunk.split(max_workers)),
            )

        patch_queue.put(EndOfQueue)
        writer.join()


def write_patches_mpi(
    path: Tpath,
    reader: BaseReader,
    patch_centers: Tcenters,
    *,
    overwrite: bool,
    progress: bool,
    max_workers: int | None = None,
    buffersize: int = -1,
) -> None:
    rank = parallel.COMM.Get_rank()
    patch_centers = get_patch_centers(patch_centers)
    preprocess = ChunkProcessor(patch_centers)

    # run all processing on the same node as the root (which handles reading)
    max_workers = parallel.get_size(max_workers)
    active_ranks = parallel.ranks_on_same_node(0, max_workers)
    # choose any rank that will handle writing the output
    for writer_rank in active_ranks:
        if writer_rank != 0:
            break
    active_ranks.remove(writer_rank)
    active_comm = parallel.COMM.Split(
        1 if rank in active_ranks else MPI.UNDEFINED, rank
    )

    if rank == writer_rank:
        _writer_process(
            get_method=partial(parallel.COMM.recv, source=MPI.ANY_SOURCE, tag=2),
            cache_directory=path,
            overwrite=overwrite,
            has_weights=reader.has_weights,
            has_redshifts=reader.has_redshifts,
            buffersize=buffersize,
        )

    elif rank in active_ranks:
        with reader:
            chunk_iter = iter(reader)
            if progress:
                chunk_iter = Indicator(reader)

            for chunk in chunk_iter:
                if rank == 0:
                    splitted = chunk.split(len(active_ranks))
                    split_assignments = dict(zip(active_ranks, splitted))

                    for dest, split in split_assignments.items():
                        if dest != 0:
                            parallel.COMM.send(split, dest=dest, tag=1)
                    split = split_assignments[0]

                else:
                    split = parallel.COMM.recv(source=0, tag=1)

                patches = preprocess.execute(split)
                parallel.COMM.send(patches, dest=writer_rank, tag=2)

            active_comm.Barrier()
            if rank == 0:
                parallel.COMM.send(EndOfQueue, dest=writer_rank, tag=2)

        active_comm.Free()

    parallel.COMM.Barrier()


def create_patches(
    cache_directory: Tpath,
    source: DataFrame | Tpath,
    *,
    ra_name: str,
    dec_name: str,
    weight_name: str | None = None,
    redshift_name: str | None = None,
    patch_centers: Tcenters | None = None,
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
    constructor = (
        new_filereader if isinstance(source, get_args(Tpath)) else DataFrameReader
    )

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

    args = (cache_directory, reader, patch_centers)
    kwargs = dict(
        overwrite=overwrite,
        progress=progress,
        max_workers=max_workers,
        buffersize=buffersize,
    )
    if parallel.use_mpi():
        write_patches_mpi(*args, **kwargs)
    else:
        write_patches_multiprocessing(*args, **kwargs)
