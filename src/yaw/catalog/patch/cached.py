from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

import numpy as np

from yaw.catalog import utils
from yaw.catalog.patch.base import Collector, PatchData, PatchMetadata
from yaw.catalog.utils import DataChunk, patch_path_from_id

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import NDArray

    from yaw.core.utils import TypePathStr


def get_and_check_data_size(
    path: TypePathStr,
    *,
    must_exist: bool,
    require_size: int | None = None,
    itemsize: int = 8,
) -> int | None:
    if path.exists():
        nbytes = os.path.getsize(path)
        size = nbytes // itemsize
        if size != require_size:
            raise ValueError("file size does not match expected size")
        return size

    elif must_exist:
        raise FileNotFoundError(str(path))


@dataclass
class CacheWriter:
    path: TypePathStr
    has_weight: bool = field(default=False)
    has_redshift: bool = field(default=False)
    chunksize: int = field(default=65_536)

    def __post_init__(self) -> None:
        self.cachesize = 0
        self.cache = {"ra": utils.ArrayCache(), "dec": utils.ArrayCache()}
        if self.has_weight:
            self.cache["weight"] = utils.ArrayCache()
        if self.has_redshift:
            self.cache["redshift"] = utils.ArrayCache()

        self.path = Path(self.path)
        if self.path.exists():
            raise FileExistsError(f"directory already exists: {self.path}")
        self.path.mkdir(parents=True)

    def flush(self):
        for key, cache in self.cache.items():
            with open(self.path / key, mode="a") as f:
                cache.get_values().tofile(f)
            cache.clear()
        self.cachesize = 0

    def append_chunk(self, chunk: DataChunk) -> None:
        chunk_dict = chunk.to_dict(drop_patch=True)
        utils.check_optional_args(chunk_dict["weight"], self.has_weight, "weight")
        utils.check_optional_args(chunk_dict["redshift"], self.has_redshift, "redshift")

        self.cachesize += len(chunk)
        for key, cache in self.cache.items():
            cache.append(chunk_dict[key])
        if self.cachesize > self.chunksize:
            self.flush()

    def append_data(
        self,
        ra: NDArray,
        dec: NDArray,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
    ) -> None:
        self.append_chunk(DataChunk(ra=ra, dec=dec, weight=weight, redshift=redshift))

    def finalize(self) -> None:
        self.flush()


class PatchWriter(Collector):
    def __init__(self, cache_directory: TypePathStr) -> None:
        self._writers: dict[int, CacheWriter] = dict()

        self.cache_directory = Path(cache_directory)
        self.cache_directory.mkdir(parents=True, exist_ok=True)

    def _get_and_delete_cache_path(self, patch_id: int) -> Path:
        cachepath = patch_path_from_id(self.cache_directory, patch_id)
        if os.path.exists(cachepath):
            shutil.rmtree(cachepath)

    def process(self, chunk: DataChunk) -> None:
        for patch_id, patch_chunk in chunk.groupby():
            if patch_id not in self._writers:
                cachepath = self._get_and_delete_cache_path(patch_id)
                self._writers[patch_id] = CacheWriter(
                    cachepath,
                    has_weight=patch_chunk.weight is not None,
                    has_redshift=patch_chunk.redshift is not None,
                )
            self._writers[patch_id].append_chunk(patch_chunk)

    def get_patches(self) -> dict[int, PatchDataCached]:
        for writer in self._writers.values():
            writer.finalize()
        return {
            patch_id: PatchDataCached.restore(patch_id, writer.path)
            for patch_id, writer in self._writers.items()
        }


class PatchDataCached(PatchData):
    def __init__(
        self,
        path: TypePathStr,
        id: int,
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.floating] | None = None,
        redshift: NDArray[np.float64] | None = None,
        metadata: PatchMetadata | None = None,
    ) -> None:
        self.id = id
        has_weight = weight is not None
        has_redshift = redshift is not None
        # create memory maps for the input data
        writer = CacheWriter(path, has_weight, has_redshift)
        writer.append_data(ra, dec, weight, redshift)
        writer.finalize()
        self.path = writer.path
        # populate the data attributes
        self._init_from_disk(metadata)

    @classmethod
    def restore(cls, id: int, path: TypePathStr) -> Self:
        new = cls.__new__(cls)
        new.path = Path(path)
        if not new.path.exists():
            raise FileNotFoundError(f"cache directory des not exist: {new.path}")
        new.id = id
        # populate the data attributes
        new._init_from_disk(metadata=None)
        return new

    @property
    def _path_metadata(self) -> Path:
        return self.path / "metadata.json"

    def _write_metadata(self) -> None:
        the_dict = self.metadata.to_dict()
        with open(self._path_metadata, "w") as f:
            json.dump(the_dict, f)

    _update_metadata_callback = _write_metadata

    def _check_data_files(self) -> None:
        # required data
        length = get_and_check_data_size("ra", must_exist=True)
        get_and_check_data_size("dec", must_exist=True, require_size=length)
        # optional data
        get_and_check_data_size("weight", must_exist=False, require_size=length)
        get_and_check_data_size("redshift", must_exist=False, require_size=length)
        return length

    def _init_metadata(self, data_length: int, metadata: PatchMetadata | None) -> None:
        overwrite = metadata is not None
        load_existing = self._path_metadata.exists()

        if overwrite:
            self.metadata = metadata
            self._write_metadata()

        elif load_existing:
            with open(self._path_metadata) as f:
                the_dict = json.load(f)
            self.metadata = PatchMetadata.from_dict(the_dict)

        else:
            self.metadata = PatchMetadata(data_length)
            self._write_metadata()

    def _init_from_disk(self, metadata: PatchMetadata | None) -> None:
        length = self._check_data_files()
        self._init_metadata(length, metadata)

    @property
    def ra(self) -> NDArray[np.float64]:
        np.fromfile(self.path / "ra", dtype=np.float64)

    @property
    def dec(self) -> NDArray[np.float64]:
        np.fromfile(self.path / "dec", dtype=np.float64)

    @property
    def weight(self) -> NDArray[np.float64] | None:
        try:
            np.fromfile(self.path / "weight", dtype=np.float64)
        except FileNotFoundError:
            return None

    @property
    def redshift(self) -> NDArray[np.float64] | None:
        try:
            np.fromfile(self.path / "redshift", dtype=np.float64)
        except FileNotFoundError:
            return None
