from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from yaw.catalog.base import Catalog, PatchMode, parse_path_or_none
from yaw.catalog.patch import PatchDataCached
from yaw.catalog.readers import ChunkReader, Reader, get_reader
from yaw.catalog.utils import DataChunk, patch_id_from_path
from yaw.core.coordinates import Coordinate

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.catalog.patch.base import PatchData
    from yaw.core.utils import TypePathStr

__all__ = ["CatalogCached"]


class CatalogCached(Catalog):
    def __init__(
        self,
        patches: Mapping[int, PatchData] | Iterable[PatchData],
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
    ) -> Catalog:
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
    ) -> Catalog:
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
    def from_cache(
        cls, cache_directory: TypePathStr, progress: bool = False
    ) -> Catalog:
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
