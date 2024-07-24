from __future__ import annotations

from collections.abc import Iterator, Sequence
from enum import Enum
from pathlib import Path
from shutil import rmtree
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

from yaw.catalog.coordinates import Coordinates, Coords3D, CoordsSky, DistsSky
from yaw.catalog.patch import DataChunk, Patch, PatchWriter
from yaw.catalog.readers import BaseReader, MemoryReader, get_filereader

__all__ = [
    "Catalog",
]

Tpath = Union[Path, str]

PATCH_NAME_TEMPLATE = "patch_{:}"


class InconsistentPatchesError(Exception):
    pass


def get_column(
    dataframe: DataFrame, column: str | None, required: bool = False
) -> NDArray | None:
    if column is None:
        if required:
            raise ValueError("column is required but no column name provided")
        return None
    return dataframe[column].to_numpy()


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
    )
    xyz = np.atleast_2d(cat.patch_centers)
    return Coords3D(xyz).to_sky()


def assign_patch_centers(
    chunk: DataChunk, patch_centers: CoordsSky
) -> NDArray[np.int32]:
    patches, _ = vq.vq(chunk.coords.to_3d().values, patch_centers.to_3d().values)
    return patches


class CatalogWriter:
    def __init__(self, path: Tpath, overwrite: bool = True) -> None:
        self.path = Path(path)
        if self.path.exists():
            if overwrite:
                rmtree(self.path)
            else:
                raise FileExistsError(f"cache directory exists: {path}")
        self.path.mkdir()
        self._writers: dict[int, PatchWriter] = {}

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args, **kwargs) -> None:
        self.finalize()

    def get_writer(self, patch_id: int) -> PatchWriter:
        if patch_id not in self._writers:
            path = self.path / PATCH_NAME_TEMPLATE.format(patch_id)
            self._writers[patch_id] = PatchWriter(path)
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
) -> None:
    if isinstance(patch_centers, Catalog):
        patch_centers = patch_centers.get_centers()
    if isinstance(patch_centers, Coordinates):
        patch_centers = patch_centers.to_sky()

    with reader, CatalogWriter(path, overwrite=overwrite) as writer:
        for chunk in reader:
            if mode == PatchMode.apply:
                chunk.set_patch(assign_patch_centers(chunk, patch_centers))
            patch_chunks = chunk.split_patches()
            writer.process_patches(patch_chunks)


class Catalog(Sequence):
    _patches = dict[int, Patch]

    def __init__(self, cache_directory: Tpath) -> None:
        self.cache_directory = Path(cache_directory)
        patches = {}
        for cache in self.cache_directory.glob(PATCH_NAME_TEMPLATE.format("*")):
            _, str_pid = cache.name.split("_")
            patches[int(str_pid)] = Patch(cache)
        self._patches = patches

    @classmethod
    def from_dataframe(
        cls,
        cache_directory: Tpath,
        data: DataFrame,
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
    ) -> Catalog:
        mode = PatchMode.determine(patch_centers, patch_name, patch_num)
        data_chunk = DataChunk.from_columns(
            ra=get_column(data, ra_name, required=True),
            dec=get_column(data, dec_name, required=True),
            weight=get_column(data, weight_name),
            redshift=get_column(data, redshift_name),
            patch=get_column(data, patch_name if mode == PatchMode.divide else None),
            degrees=degrees,
        )
        reader = MemoryReader(data_chunk, chunksize=chunksize)
        if mode == PatchMode.create:
            patch_centers = create_patch_centers(reader, patch_num, probe_size)
        write_patches(cache_directory, reader, mode, patch_centers, overwrite)
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
        **reader_kwargs,
    ) -> Catalog:
        mode = PatchMode.determine(patch_centers, patch_name, patch_num)
        reader = get_filereader(path)(
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
        if mode == PatchMode.create:
            patch_centers = create_patch_centers(reader, patch_num, probe_size)
        write_patches(cache_directory, reader, mode, patch_centers, overwrite)
        return cls(cache_directory)

    def __repr__(self) -> str:
        num_patches = len(self)
        weights = self.has_weight()
        redshifts = self.has_redshift()
        return f"{type(self).__name__}({num_patches=}, {weights=}, {redshifts=})"

    def __len__(self) -> int:
        return len(self._patches)

    def __getitem__(self, patch_id: int) -> Patch:
        return self._patches[patch_id]

    def __iter__(self) -> Iterator[Patch]:
        for patch_id in sorted(self._patches.keys()):
            yield self._patches[patch_id]

    def has_weight(self) -> bool:
        has_weight = tuple(patch.has_weight() for patch in iter(self))
        if all(has_weight):
            return True
        elif not any(has_weight):
            return False
        raise InconsistentPatchesError("'weight' not consistent")

    def has_redshift(self) -> bool:
        has_redshift = tuple(patch.has_redshift() for patch in iter(self))
        if all(has_redshift):
            return True
        elif not any(has_redshift):
            return False
        raise InconsistentPatchesError("'redshift' not consistent")

    def get_redshift_range(self) -> tuple[float, float]:
        if not self.has_redshift():
            raise ValueError("no 'redshift' attached")
        zmin = np.inf
        zmax = -np.inf
        for patch in iter(self):
            z = patch.redshift
            zmin = min(zmin, z.min())
            zmax = max(zmax, z.max())
        return float(zmin), float(zmax)

    def get_num_records(self) -> tuple[int]:
        return tuple(patch.meta.num_records for patch in iter(self))

    def get_totals(self) -> tuple[float]:
        return tuple(patch.meta.total for patch in iter(self))

    def get_centers(self) -> CoordsSky:
        return CoordsSky.from_coords(patch.meta.center for patch in iter(self))

    def get_radii(self) -> DistsSky:
        return DistsSky.from_dists(patch.meta.radius for patch in iter(self))
