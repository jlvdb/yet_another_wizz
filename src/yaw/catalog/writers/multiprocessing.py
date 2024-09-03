from __future__ import annotations

import multiprocessing
from itertools import repeat
from typing import TYPE_CHECKING

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
