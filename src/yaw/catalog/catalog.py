from __future__ import annotations

import multiprocessing
from collections import deque
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from enum import Enum
from pathlib import Path
from shutil import get_terminal_size, rmtree
from typing import Union

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np
import treecorr
from numpy.typing import NDArray
from pandas import DataFrame
from scipy.cluster import vq
from tqdm import tqdm

from yaw.catalog.patch import BinnedTrees, Patch, PatchWriter
from yaw.catalog.readers import BaseReader, DataFrameReader, new_filereader
from yaw.catalog.trees import parse_binning
from yaw.catalog.utils import DataChunk, Tclosed
from yaw.coordinates import Coordinates, Coords3D, CoordsSky, DistsSky
from yaw.parallel import ParallelHelper

__all__ = [
    "Catalog",
]

Tpath = Union[Path, str]

PATCH_NAME_TEMPLATE = "patch_{:}"


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
        patch_centers: Catalog | Coordinates | None,
        patch_name: str | None,
        patch_num: int | None,
    ) -> PatchMode:
        if patch_centers is not None:
            return PatchMode.apply
        if patch_name is not None:
            return PatchMode.divide
        elif patch_num is not None:
            return PatchMode.create
        raise ValueError("no patch method specified")


def create_patch_centers(
    reader: BaseReader, patch_num: int, probe_size: int
) -> CoordsSky:
    if probe_size < 10 * patch_num:
        probe_size = int(100_000 * np.sqrt(patch_num))
    sparse_factor = np.ceil(reader.num_records / probe_size)
    test_sample = reader.read(int(sparse_factor))

    cat = treecorr.Catalog(
        ra=test_sample.coords.ra,
        ra_units="radians",
        dec=test_sample.coords.dec,
        dec_units="radians",
        npatch=patch_num,
        num_threads=ParallelHelper.num_threads(),
    )
    xyz = np.atleast_2d(cat.patch_centers)
    return Coords3D(xyz).to_sky()


def compute_patch_ids(
    coord_sky: NDArray, patch_centers_3d: NDArray
) -> NDArray[np.int32]:
    coords = CoordsSky(coord_sky).to_3d()
    patch_centers = Coords3D(patch_centers_3d)
    patches, _ = vq.vq(coords, patch_centers)
    return patches.astype(np.int32, copy=False)


def compute_patch_ids_parallel(
    chunk: DataChunk, patch_centers: CoordsSky, num_threads: int = 4
) -> NDArray[np.int32]:
    coord_chunks = np.array_split(chunk.coords.data, multiprocessing.cpu_count())
    patch_centers_3d = patch_centers.to_3d()
    with multiprocessing.Pool(processes=num_threads) as pool:
        args = ((chunk, patch_centers_3d.data) for chunk in coord_chunks)
        patches = pool.starmap(compute_patch_ids, args)
    return np.concatenate(patches, dtype=np.int32)


class CatalogWriter(AbstractContextManager):
    def __init__(
        self, cache_directory: Tpath, overwrite: bool = True, progress: bool = False
    ) -> None:
        self.cache_directory = Path(cache_directory)
        if self.cache_directory.exists():
            if overwrite:
                rmtree(self.cache_directory)
            else:
                raise FileExistsError(f"cache directory exists: {cache_directory}")
        self.cache_directory.mkdir()
        self._writers: dict[int, PatchWriter] = {}
        self.progress = progress

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalize()

    def get_writer(self, patch_id: int) -> PatchWriter:
        if patch_id not in self._writers:
            cache_path = self.cache_directory / PATCH_NAME_TEMPLATE.format(patch_id)
            self._writers[patch_id] = PatchWriter(cache_path)
        return self._writers[patch_id]

    def process_patches(self, patches: dict[int, DataChunk]) -> None:
        for patch_id, patch in patches.items():
            self.get_writer(patch_id).process_chunk(patch)

    def finalize(self) -> None:
        for writer in self._writers.values():
            writer.finalize()


def write_patches(
    path: Tpath,
    reader: BaseReader,
    mode: PatchMode,
    patch_centers: Catalog | Coordinates | None,
    overwrite: bool,
    progress: bool,
) -> None:
    if isinstance(patch_centers, Catalog):
        patch_centers = patch_centers.get_centers()
    if isinstance(patch_centers, Coordinates):
        patch_centers = patch_centers.to_sky()

    with reader, CatalogWriter(path, overwrite=overwrite, progress=progress) as writer:
        chunk_iter_progress_optional = tqdm(
            reader,
            total=reader.num_chunks,
            ncols=min(80, get_terminal_size()[0]),
            disable=(not progress),
        )
        for chunk in chunk_iter_progress_optional:
            if mode != PatchMode.divide:
                patch_ids = compute_patch_ids_parallel(chunk, patch_centers)
                chunk.set_patch_ids(patch_ids)
            patch_chunks = chunk.split_patches()
            writer.process_patches(patch_chunks)


def compute_patch_metadata(path: Tpath, progress: bool = False):
    patch_paths = tuple(Path(path).glob(PATCH_NAME_TEMPLATE.format("*")))
    # instantiate patches, which trigger computing the patch meta-data
    deque(
        ParallelHelper.iter_unordered(
            Patch,
            patch_paths,
            progress=progress,
            total=len(patch_paths),
        ),
        maxlen=0,
    )


class Catalog(Mapping[int, Patch]):
    _patches = dict[int, Patch]

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)

        if ParallelHelper.on_root():
            patches = {}
            for cache in self.cache_directory.glob(PATCH_NAME_TEMPLATE.format("*")):
                patch_id = int(cache.name.split("_")[1])
                patches[patch_id] = Patch(cache)
            if len(patches) == 0:
                raise FileNotFoundError(f"no patches found in '{cache_directory}'")
        else:
            patches = None

        if ParallelHelper.use_mpi():
            self._patches = ParallelHelper.comm.bcast(patches, root=0)
        else:
            self._patches = patches

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
        patch_centers: Catalog | Coordinates | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        chunksize: int = 1_000_000,
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
            write_patches(
                cache_directory, reader, mode, patch_centers, overwrite, progress
            )
        ParallelHelper.comm.Barrier()

        compute_patch_metadata(cache_directory, progress)
        return cls(cache_directory)

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
        patch_centers: Catalog | Coordinates | None = None,
        patch_name: str | None = None,
        patch_num: int | None = None,
        degrees: bool = True,
        chunksize: int = 1_000_000,
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
            write_patches(
                cache_directory, reader, mode, patch_centers, overwrite, progress
            )
        ParallelHelper.comm.Barrier()

        compute_patch_metadata(cache_directory, progress)
        return cls(cache_directory)

    def __repr__(self) -> str:
        num_patches = len(self)
        weights = self.has_weights()
        redshifts = self.has_redshifts()
        return f"{type(self).__name__}({num_patches=}, {weights=}, {redshifts=})"

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self._patches[patch_id]

    def __iter__(self) -> Iterator[int]:
        yield from sorted(self._patches.keys())

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

    def get_centers(self) -> CoordsSky:
        return CoordsSky.from_coords(patch.meta.center for patch in self.values())

    def get_radii(self) -> DistsSky:
        return DistsSky.from_dists(patch.meta.radius for patch in self.values())

    def build_trees(
        self,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = "left",
        leafsize: int = 16,
        force: bool = False,
        progress: bool = False,
    ) -> None:
        binning = parse_binning(binning)
        patches = self.values()

        deque(
            ParallelHelper.iter_unordered(
                BinnedTrees.build,
                patches,
                func_args=(binning,),
                func_kwargs=dict(closed=closed, leafsize=leafsize, force=force),
                progress=progress,
                total=len(patches),
            ),
            maxlen=0,
        )
