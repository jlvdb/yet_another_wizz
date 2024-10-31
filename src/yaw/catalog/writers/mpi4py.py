from __future__ import annotations

from typing import TYPE_CHECKING

from mpi4py import MPI

from yaw.catalog.readers import DataChunk
from yaw.catalog.writers.base import (
    CatalogWriter,
    get_patch_centers,
    logger,
    split_into_patches,
)
from yaw.utils import parallel
from yaw.utils.logging import Indicator
from yaw.utils.parallel import EndOfQueue

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from mpi4py.MPI import Comm

    from yaw.catalog.containers import TypePatchCenters
    from yaw.catalog.generators import ChunkGenerator


class WorkerManager:
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
    num_ranks = comm.Get_size()

    if comm.Get_rank() == reader_rank:
        splits = chunk.split(num_ranks)

        for rank, split in enumerate(splits):
            if rank != reader_rank:
                comm.send(split, dest=rank, tag=2)

        return splits[reader_rank]

    else:
        return comm.recv(source=0, tag=2)


def chunk_processing_task(
    comm: Comm,
    worker_config: WorkerManager,
    patch_centers: TypePatchCenters | None,
    chunk_iter: Iterator[DataChunk],
) -> None:
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
    has_weights: bool,
    has_redshifts: bool,
    overwrite: bool = True,
    buffersize: int = -1,
) -> None:
    recv = parallel.COMM.recv
    with CatalogWriter(
        cache_directory,
        has_weights=has_weights,
        has_redshifts=has_redshifts,
        overwrite=overwrite,
        buffersize=buffersize,
    ) as writer:
        while (patches := recv(source=MPI.ANY_SOURCE, tag=1)) is not EndOfQueue:
            writer.process_patches(patches)


def write_patches(
    path: Path | str,
    generator: ChunkGenerator,
    patch_centers: TypePatchCenters,
    *,
    overwrite: bool,
    progress: bool,
    max_workers: int | None = None,
    buffersize: int = -1,
) -> None:
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
            has_weights=generator.has_weights,
            has_redshifts=generator.has_redshifts,
            overwrite=overwrite,
            buffersize=buffersize,
        )

    elif rank in worker_config.active_ranks:
        if patch_centers is not None:
            patch_centers = get_patch_centers(patch_centers)

        with generator:
            chunk_iter = Indicator(generator) if progress else iter(generator)
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
