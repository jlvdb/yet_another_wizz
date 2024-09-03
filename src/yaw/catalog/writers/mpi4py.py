from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

from mpi4py import MPI

from yaw.catalog.writers.base import CatalogWriter, ChunkProcessor, get_patch_centers
from yaw.containers import Tpath
from yaw.utils import parallel
from yaw.utils.logging import Indicator
from yaw.utils.parallel import EndOfQueue

if TYPE_CHECKING:
    from collections.abc import Callable

    from yaw.catalog.containers import Tcenters
    from yaw.catalog.readers import BaseReader
    from yaw.catalog.utils import DataChunk


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


def write_patches(
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
