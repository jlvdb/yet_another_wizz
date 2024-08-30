from __future__ import annotations

import logging
import multiprocessing
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from enum import Enum
from functools import partial
from itertools import repeat
from pathlib import Path
from shutil import rmtree
from typing import Union, get_args

import numpy as np
import treecorr
from mpi4py import MPI
from numpy.typing import NDArray
from scipy.cluster import vq
from typing_extensions import Self

from yaw.catalog.patch import BinnedTrees, Patch, PatchWriter
from yaw.catalog.readers import BaseReader, DataFrameReader, new_filereader
from yaw.catalog.utils import DataChunk
from yaw.catalog.utils import MockDataFrame as DataFrame
from yaw.containers import Tclosed, Tpath, default_closed, parse_binning
from yaw.utils import AngularCoordinates, AngularDistances, parallel
from yaw.utils.logging import Indicator
from yaw.utils.parallel import EndOfQueue

__all__ = [
    "Catalog",
]

Tcenters = Union["Catalog", AngularCoordinates]

PATCH_NAME_TEMPLATE = "patch_{:}"
PATCHFILE_NAME = "num_patches"

logger = logging.getLogger("yaw.catalog")


class InconsistentPatchesError(Exception):
    pass


class InconsistentTreesError(Exception):
    pass


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
    max_workers = parallel.get_size(max_workers)

    if isinstance(patch_centers, Catalog):
        patch_centers = patch_centers.get_centers()
    elif not isinstance(patch_centers, AngularCoordinates):
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        )

    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(max_workers)
    with reader, manager, pool:
        patch_queue = manager.Queue()

        preprocess = ChunkProcessor(patch_centers)

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
    max_workers = parallel.get_size(max_workers)
    rank = parallel.COMM.Get_rank()

    if isinstance(patch_centers, Catalog):
        patch_centers = patch_centers.get_centers()
    elif not isinstance(patch_centers, AngularCoordinates):
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        )

    preprocess = ChunkProcessor(patch_centers)

    # run all processing on the same node as the root (which handles reading)
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


def patch_id_from_path(patch_path: Tpath) -> int:
    _, id_str = Path(patch_path).name.split("_")
    return int(id_str)


def create_patchfile(cache_directory: Tpath, num_patches: int) -> None:
    with (cache_directory / PATCHFILE_NAME).open("w") as f:
        f.write(str(num_patches))


def verify_patchfile(cache_directory: Tpath, num_expect: int) -> None:
    path = Path(cache_directory) / PATCHFILE_NAME
    if not path.exists():
        raise InconsistentPatchesError("patch indicator file not found")

    with path.open() as f:
        num_patches = int(f.read())
    if num_expect != num_patches:
        raise ValueError(f"expected {num_expect} patches but found {num_patches}")


def compute_patch_metadata(
    cache_directory: Tpath, progress: bool = False
) -> dict[int, Patch]:
    if parallel.on_root():
        logger.info("computing patch metadata")

    cache_directory = Path(cache_directory)
    patch_paths = tuple(cache_directory.glob(PATCH_NAME_TEMPLATE.format("*")))
    create_patchfile(cache_directory, len(patch_paths))

    # instantiate patches, which trigger computing the patch meta-data
    patch_iter = parallel.iter_unordered(Patch, patch_paths)
    if progress:
        patch_iter = Indicator(patch_iter, len(patch_paths))

    patches = {patch_id_from_path(patch.cache_path): patch for patch in patch_iter}
    return parallel.COMM.bcast(patches, root=0)


class Catalog(Mapping[int, Patch]):
    patches = dict[int, Patch]

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)

        patches = None

        if parallel.on_root():
            logger.info("restoring from cache directory: %s", cache_directory)

            template = PATCH_NAME_TEMPLATE.format("*")
            patch_paths = tuple(self.cache_directory.glob(template))
            verify_patchfile(self.cache_directory, len(patch_paths))

            patches = {patch_id_from_path(cache): Patch(cache) for cache in patch_paths}

        self.patches = parallel.COMM.bcast(patches, root=0)

    @classmethod
    def from_dataframe(
        cls,
        cache_directory: Tpath,
        dataframe: DataFrame,
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
        **reader_kwargs,
    ) -> Catalog:
        create_patches(
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
            **reader_kwargs,
        )

        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = compute_patch_metadata(cache_directory, progress)
        return new

    @classmethod
    def from_file(
        cls,
        cache_directory: Tpath,
        path: Tpath,
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
        **reader_kwargs,
    ) -> Catalog:
        create_patches(
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
            **reader_kwargs,
        )

        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = compute_patch_metadata(cache_directory, progress)
        return new

    def __repr__(self) -> str:
        num_patches = len(self)
        weights = self.has_weights
        redshifts = self.has_redshifts
        return f"{type(self).__name__}({num_patches=}, {weights=}, {redshifts=})"

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self.patches[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self.patches.keys())

    @property
    def has_weights(self) -> bool:
        has_weights = tuple(patch.meta.has_weights for patch in self.values())
        if all(has_weights):
            return True
        elif not any(has_weights):
            return False
        raise InconsistentPatchesError("'weights' not consistent")

    @property
    def has_redshifts(self) -> bool:
        has_redshifts = tuple(patch.meta.has_redshifts for patch in self.values())
        if all(has_redshifts):
            return True
        elif not any(has_redshifts):
            return False
        raise InconsistentPatchesError("'redshifts' not consistent")

    def get_num_records(self) -> tuple[int]:
        return tuple(patch.meta.num_records for patch in self.values())

    def get_totals(self) -> tuple[float]:
        return tuple(patch.meta.total for patch in self.values())

    def get_centers(self) -> AngularCoordinates:
        return AngularCoordinates.from_coords(
            patch.meta.center for patch in self.values()
        )

    def get_radii(self) -> AngularDistances:
        return AngularDistances.from_dists(patch.meta.radius for patch in self.values())

    def build_trees(
        self,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = default_closed,
        leafsize: int = 16,
        force: bool = False,
        progress: bool = False,
    ) -> None:
        binning = parse_binning(binning, optional=True)

        if parallel.on_root():
            logger.debug(
                "building patch-wise trees (%s)",
                "unbinned" if binning is None else f"using {len(binning) - 1} bins",
            )

        patch_tree_iter = parallel.iter_unordered(
            BinnedTrees.build,
            self.values(),
            func_args=(binning,),
            func_kwargs=dict(closed=closed, leafsize=leafsize, force=force),
        )
        if progress:
            patch_tree_iter = Indicator(patch_tree_iter, len(self))

        deque(patch_tree_iter, maxlen=0)
