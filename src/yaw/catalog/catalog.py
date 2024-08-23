from __future__ import annotations

import logging
import multiprocessing
from collections import deque
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from shutil import rmtree
from typing import Union

import numpy as np
import treecorr
from numpy.typing import NDArray
from scipy.cluster import vq
from typing_extensions import Self

from yaw.catalog.patch import BinnedTrees, Patch, PatchWriter
from yaw.catalog.readers import BaseReader, DataFrameReader, new_filereader
from yaw.catalog.utils import DataChunk
from yaw.catalog.utils import MockDataFrame as DataFrame
from yaw.containers import Tclosed, Tpath, default_closed, parse_binning
from yaw.utils import AngularCoordinates, AngularDistances, ParallelHelper
from yaw.utils.progress import Indicator

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
        log_sink = logger.debug

        if patch_centers is not None:
            log_sink("applying patch %i centers", len(patch_centers))
            return PatchMode.apply

        if patch_name is not None:
            log_sink("dividing patches based on '%s'", patch_name)
            return PatchMode.divide

        elif patch_num is not None:
            log_sink("creating %i patches", patch_num)
            return PatchMode.create

        raise ValueError("no patch method specified")


def create_patch_centers(
    reader: BaseReader, patch_num: int, probe_size: int
) -> AngularCoordinates:
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    sparse_factor = np.ceil(reader.num_records / probe_size)
    test_sample = reader.read(int(sparse_factor))

    logger.info("computing patch centers from %ix sparse sampling", sparse_factor)

    cat = treecorr.Catalog(
        ra=test_sample.coords.ra,
        ra_units="radians",
        dec=test_sample.coords.dec,
        dec_units="radians",
        npatch=patch_num,
        config=dict(num_threads=ParallelHelper.num_threads),
    )
    xyz = np.atleast_2d(cat.patch_centers)
    return AngularCoordinates.from_3d(xyz)


class ChunkProcessor(Callable):
    __slots__ = ("patch_centers",)

    def __init__(self, patch_centers: AngularCoordinates | None) -> None:
        if patch_centers is None:
            self.patch_centers = None
        else:
            self.patch_centers = patch_centers.to_3d()

    def _compute_patch_ids(self, chunk: DataChunk) -> NDArray[np.int32]:
        patches, _ = vq.vq(chunk.coords.to_3d(), self.patch_centers)
        return patches.astype(np.int32, copy=False)

    def __call__(self, chunk: DataChunk) -> dict[int, DataChunk]:
        if self.patch_centers is not None:
            patch_ids = self._compute_patch_ids(chunk)
            chunk.set_patch_ids(patch_ids)

        return chunk.split_patches()


class CatalogWriter(AbstractContextManager):
    def __init__(self, cache_directory: Tpath, overwrite: bool = True) -> None:
        self.cache_directory = Path(cache_directory)
        if self.cache_directory.exists():
            if overwrite:
                logger.info("overwriting cache directory: %s", cache_directory)
                rmtree(self.cache_directory)
            else:
                raise FileExistsError(f"cache directory exists: {cache_directory}")
        else:
            logger.info("using cache directory: %s", cache_directory)

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
            writer = PatchWriter(self.get_writer_path(patch_id))
            self._writers[patch_id] = writer
            return writer

    def process_patches(self, patches: dict[int, DataChunk]) -> None:
        for patch_id, patch in patches.items():
            self.get_writer(patch_id).process_chunk(patch)

    def finalize(self) -> None:
        for writer in self._writers.values():
            writer.finalize()


def write_patches(
    path: Tpath,
    reader: BaseReader,
    patch_centers: Tcenters,
    overwrite: bool,
    progress: bool,
    num_threads: int | None = None,
) -> None:
    num_threads = num_threads or ParallelHelper.num_threads

    if isinstance(patch_centers, Catalog):
        patch_centers = patch_centers.get_centers()
    elif not isinstance(patch_centers, AngularCoordinates):
        raise TypeError(
            "'patch_centers' must be of type 'Catalog' or 'AngularCoordinates'"
        )
    preprocess = ChunkProcessor(patch_centers)

    writer = CatalogWriter(path, overwrite=overwrite)
    pool = multiprocessing.Pool(num_threads)
    with reader, writer, pool:
        chunk_iter = iter(reader)
        if progress:
            chunk_iter = Indicator(reader, description="I/O")

        for chunk in chunk_iter:
            thread_chunks = chunk.split(num_threads)

            for patch_chunks in pool.imap_unordered(preprocess, thread_chunks):
                writer.process_patches(patch_chunks)


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
    if ParallelHelper.on_root():
        cache_directory = Path(cache_directory)
        patch_paths = tuple(cache_directory.glob(PATCH_NAME_TEMPLATE.format("*")))
        create_patchfile(cache_directory, len(patch_paths))
    else:
        patch_paths = None
    patch_paths = ParallelHelper.comm.bcast(patch_paths, root=0)

    # instantiate patches, which trigger computing the patch meta-data
    patch_iter = ParallelHelper.iter_unordered(Patch, patch_paths)
    if progress:
        patch_iter = Indicator(patch_iter, len(patch_paths), "metadata")

    patches = {patch_id_from_path(patch.cache_path): patch for patch in patch_iter}
    return ParallelHelper.comm.bcast(patches, root=0)


class Catalog(Mapping[int, Patch]):
    patches = dict[int, Patch]

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)
        logger.info("restoring from cache directory: %s", cache_directory)

        if ParallelHelper.on_root():
            template = PATCH_NAME_TEMPLATE.format("*")
            patch_paths = tuple(self.cache_directory.glob(template))
            verify_patchfile(self.cache_directory, len(patch_paths))

            patches = {patch_id_from_path(cache): Patch(cache) for cache in patch_paths}

        else:
            patches = None

        self.patches = ParallelHelper.comm.bcast(patches, root=0)

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
        if ParallelHelper.on_root():
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

            write_patches(cache_directory, reader, patch_centers, overwrite, progress)
        ParallelHelper.comm.Barrier()

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
        if ParallelHelper.on_root():
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

            write_patches(cache_directory, reader, patch_centers, overwrite, progress)
        ParallelHelper.comm.Barrier()

        new = cls.__new__(cls)
        new.cache_directory = Path(cache_directory)
        new.patches = compute_patch_metadata(cache_directory, progress)
        return new

    def __repr__(self) -> str:
        num_patches = len(self)
        weights = self.has_weights()
        redshifts = self.has_redshifts()
        return f"{type(self).__name__}({num_patches=}, {weights=}, {redshifts=})"

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self.patches[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self.patches.keys())

    def has_weights(self) -> bool:
        has_weights = tuple(patch.has_weights() for patch in self.values())
        if all(has_weights):
            return True
        elif not any(has_weights):
            return False
        raise InconsistentPatchesError("'weights' not consistent")

    def has_redshifts(self) -> bool:
        has_redshifts = tuple(patch.has_redshifts() for patch in self.values())
        if all(has_redshifts):
            return True
        elif not any(has_redshifts):
            return False
        raise InconsistentPatchesError("'redshifts' not consistent")

    def get_redshift_range(self) -> tuple[float, float]:
        if not self.has_redshifts():
            raise ValueError("no 'redshifts' attached")

        min_redshifts = []
        max_redshifts = []
        for patch in self.values():
            redshifts = patch.redshifts  # triggers I/O
            min_redshifts.append(redshifts.min())
            max_redshifts.append(redshifts.max())

        return float(min(min_redshifts)), float(max(max_redshifts))

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
        patches = self.values()

        patch_tree_iter = ParallelHelper.iter_unordered(
            BinnedTrees.build,
            patches,
            func_args=(binning,),
            func_kwargs=dict(closed=closed, leafsize=leafsize, force=force),
        )
        if progress:
            patch_tree_iter = Indicator(patch_tree_iter, len(self), description="trees")

        deque(patch_tree_iter, maxlen=0)
