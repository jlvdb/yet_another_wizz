from __future__ import annotations

import multiprocessing
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from yaw.catalog.utils import PatchData
from yaw.catalog.writers.base import (
    CatalogWriter,
    get_patch_centers,
    logger,
    split_into_patches,
    write_patches_unthreaded,
)
from yaw import parallel
from yaw.utils import AngularCoordinates
from yaw.logging import Indicator
from yaw.parallel import EndOfQueue

if TYPE_CHECKING:
    from multiprocessing import Queue
    from pathlib import Path

    from typing_extensions import Self

    from yaw.catalog.containers import TypePatchCenters
    from yaw.catalog.generators import ChunkGenerator, DataChunk


class ChunkProcessingTask:
    def __init__(
        self,
        patch_queue: Queue[dict[int, PatchData] | EndOfQueue],
        patch_centers: AngularCoordinates | None,
    ) -> None:
        self.patch_queue = patch_queue

        if isinstance(patch_centers, AngularCoordinates):
            self.patch_centers = patch_centers.to_3d()
        else:
            self.patch_centers = None

    def __call__(self, chunk: DataChunk) -> dict[int, PatchData]:
        patches = split_into_patches(chunk, self.patch_centers)
        self.patch_queue.put(patches)


@dataclass
class WriterProcess(AbstractContextManager):
    patch_queue: Queue[dict[int, PatchData] | EndOfQueue]
    cache_directory: Path | str
    has_weights: bool = field(kw_only=True)
    has_redshifts: bool = field(kw_only=True)
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
            has_weights=self.has_weights,
            has_redshifts=self.has_redshifts,
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
    generator: ChunkGenerator,
    patch_centers: TypePatchCenters,
    *,
    overwrite: bool,
    progress: bool,
    max_workers: int | None = None,
    buffersize: int = -1,
) -> None:
    max_workers = parallel.get_size(max_workers)

    if max_workers == 1:
        logger.debug("running preprocessing sequentially")
        return write_patches_unthreaded(
            path,
            generator,
            patch_centers,
            overwrite=overwrite,
            progress=progress,
            buffersize=buffersize,
        )

    else:
        logger.debug("running preprocessing on %d workers", max_workers)

    with (
        generator,
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
            has_weights=generator.has_weights,
            has_redshifts=generator.has_redshifts,
            overwrite=overwrite,
            buffersize=buffersize,
        ):
            chunk_iter = Indicator(generator) if progress else iter(generator)
            for chunk in chunk_iter:
                pool.map(chunk_processing_task, chunk.split(max_workers))

            patch_queue.put(EndOfQueue)
