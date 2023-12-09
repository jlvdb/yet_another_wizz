from __future__ import annotations

import multiprocessing
from collections.abc import Generator, Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np

from yaw.catalog.base import (
    Catalog,
    IpcData,
    ParallelContext,
    PatchMode,
    parse_path_or_none,
)
from yaw.catalog.kdtree import build_trees_binned
from yaw.catalog.patch import PatchDataCached
from yaw.catalog.readers import ChunkReader, Reader, get_reader
from yaw.catalog.utils import DataChunk, patch_id_from_path, read_pickle, write_pickle
from yaw.core.containers import Binning
from yaw.core.coordinates import Coordinate

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.kdtree import SphericalKDTree
    from yaw.core.utils import TypePathStr

__all__ = ["CatalogCached"]


class IpcDataCached(IpcData):
    def __init__(self, tree_path: Path) -> None:
        self.path = tree_path

    def __enter__(self) -> Self:
        pass

    def __exit__(self, *args, **kwargs) -> None:
        pass

    def get_trees(self) -> list[SphericalKDTree] | SphericalKDTree:
        return read_pickle(self.path)


def _worker_build_tree(args: tuple[TypePathStr, TypePathStr, Binning | None]) -> None:
    patch_path, tree_path, binning = args
    patch_id = patch_id_from_path(patch_path)
    patch = PatchDataCached.restore(patch_id, patch_path)
    trees = build_trees_binned(patch, binning)
    write_pickle(tree_path, trees)


class ParallelContextCached(ParallelContext):
    def __init__(
        self,
        catalog: CatalogCached,
        binning: Binning | Iterable | None,
        num_threads: int,
    ) -> None:
        super().__init__(catalog, binning, num_threads)

    def _get_path_patch(self, patch_id: int) -> Path:
        return self._path_trees_template.format(patch_id)

    @property
    def _path_lock(self) -> Path:
        return self.catalog.cache_directory / "lock"

    @property
    def _path_binning(self) -> Path:
        return self.catalog.cache_directory / "binning.pickle"

    @property
    def _path_trees_template(self) -> str:
        return str(self.catalog.cache_directory / "trees_{:d}.pickle")

    def _get_path_trees(self, patch_id: int) -> Path:
        return self._path_trees_template.format(patch_id)

    def __enter__(self) -> Self:
        if self._path_lock.exists():
            raise FileExistsError(
                "catalog is currently locked from a differenct operation"
            )
        self._path_lock.touch()

        if self._path_binning.exists():
            current_binning: Binning | None = read_pickle(self._path_binning)
        building_required = current_binning != self.binning
        if building_required:
            write_pickle(self._path_binning, self.binning)

        args = []
        for patch in self.catalog:
            args.append([patch.path, self._get_path_trees(patch.id), self.binning])
        with multiprocessing.Pool(self.num_threads) as pool:
            pool.imap_unordered(_worker_build_tree, args)

        return self

    def __exit__(self, *args, **kwargs) -> None:
        self._path_lock.unlink()

    def get_patches_ipc(self) -> list[IpcDataCached]:
        return [self._get_path_trees(patch_id) for patch_id in self.catalog.ids]


class CatalogCached(Catalog):
    def __init__(
        self,
        patches: Mapping[int, PatchDataCached] | Iterable[PatchDataCached],
        cache_directory: TypePathStr,
    ) -> None:
        super().__init__(patches)
        self.cache_directory = parse_path_or_none(cache_directory)

    @classmethod
    def from_records(
        cls,
        cache_directory: TypePathStr,
        ra: NDArray,
        dec: NDArray,
        patches: NDArray | Catalog | Coordinate | int,
        *,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
        degrees: bool = True,
        n_per_patch: int | None = None,
    ) -> Self:
        data = DataChunk(
            ra=np.asarray(np.deg2rad(ra) if degrees else ra),
            dec=np.asarray(np.deg2rad(dec) if degrees else dec),
            weight=np.asarray(weight),
            redshift=np.asarray(redshift),
        )
        patch_mode = PatchMode.get(patches)
        if patch_mode == PatchMode.divide:
            data.set_patch(patches)

        reader_kwargs = dict(data=data, patch_name=patches, degrees=degrees)
        return cls._from_reader(
            reader=ChunkReader,
            reader_kwargs=reader_kwargs,
            patch_mode=patch_mode,
            patch_data=patches,
            cache_directory=cache_directory,
            n_per_patch=n_per_patch,
        )

    @classmethod
    def from_file(
        cls,
        cache_directory: TypePathStr,
        path: str,
        ra_name: str,
        dec_name: str,
        patches: str | int | Catalog | Coordinate,
        *,
        weight_name: str | None = None,
        redshift_name: str | None = None,
        degrees: bool = True,
        n_per_patch: int | None = None,
        reader: type[Reader] | None = None,
        reader_kwargs: dict | None = None,
    ) -> Self:
        patch_mode = PatchMode.get(patches)

        if reader is None:
            reader = get_reader(path)
        if reader_kwargs is None:
            reader_kwargs = dict()
        else:
            reader_kwargs = {k: v for k, v in reader_kwargs.items()}
        if patch_mode == PatchMode.divide:
            reader_kwargs["patch_name"] = patches
        reader_kwargs.update(
            dict(
                path=path,
                ra_name=ra_name,
                dec_name=dec_name,
                weight_name=weight_name,
                redshift_name=redshift_name,
                degrees=degrees,
            )
        )

        return cls._from_reader(
            reader=reader,
            reader_kwargs=reader_kwargs,
            patch_mode=patch_mode,
            patch_data=patches,
            cache_directory=cache_directory,
            n_per_patch=n_per_patch,
        )

    @classmethod
    def from_cache(cls, cache_directory: TypePathStr, progress: bool = False) -> Self:
        cache_directory = Path(cache_directory)
        if not cache_directory.exists():
            raise FileNotFoundError(
                f"cache directory does not exist: {cache_directory}"
            )
        patches = {}
        for patch_path in cache_directory.glob("patch_*"):
            patch_id = patch_id_from_path(patch_path)
            patches[patch_id] = PatchDataCached.restore(patch_id, patch_path)
        return cls(patches, cache_directory)

    def __iter__(self) -> Generator[PatchDataCached]:
        return super().__iter__()

    def parallel_context(
        self,
        binning: Binning | Iterable | None,
    ) -> ParallelContextCached:
        return ParallelContextCached(self, binning)
