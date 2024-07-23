from __future__ import annotations

import json
from collections.abc import Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from yaw.coordinates import CoordsSky, DistsSky

__all__ = [
    "DataChunk",
    "PatchWriter",
    "Patch",
]

TypePathStr = Union[Path, str]


def split_array(
    array: NDArray | None,
    idx_sort: NDArray[np.int64],
    idx_split: NDArray[np.int64],
) -> list[NDArray] | list[None]:
    if array is None:
        return [None] * (len(idx_split) + 1)
    array_sorted = array[idx_sort]
    return np.split(array_sorted, idx_split)


class DataChunk:
    def __init__(
        self,
        coords: CoordsSky,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
        patch: NDArray[np.int32] | None = None,
    ) -> None:
        self.coords = coords
        self.weight = weight
        self.redshift = redshift
        self.set_patch(patch)

    @classmethod
    def from_columns(
        cls,
        ra: NDArray,
        dec: NDArray,
        weight: NDArray | None = None,
        redshift: NDArray | None = None,
        patch: NDArray | None = None,
        degrees: bool = True
    ):
        if degrees:
            ra = np.deg2rad(ra)
            dec = np.deg2rad(dec)
        coords = CoordsSky(np.column_stack((ra, dec)))
        return cls(coords, weight, redshift, patch)

    @classmethod
    def from_chunks(cls, chunks: Sequence[DataChunk]) -> DataChunk:
        def concat_attr(attr: str) -> NDArray | None:
            values = tuple(getattr(chunk, attr) for chunk in chunks)
            value_set = tuple(value is not None for value in values)
            if all(value_set):
                return np.concatenate(values)
            elif not any(value_set):
                return None
            raise ValueError(f"not all chunks have '{attr}' set")

        return DataChunk(
            coords=CoordsSky.from_coords(chunk.coords for chunk in chunks),
            weight=concat_attr("weight"),
            redshift=concat_attr("redshift"),
            patch=concat_attr("patch"),
        )

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, index: ArrayLike) -> DataChunk:
        return DataChunk(
            coords=self.coords[index],
            weight=self.weight[index] if self.weight is not None else None,
            redshift=self.redshift[index] if self.redshift is not None else None,
            patch=self.patch[index] if self.patch is not None else None,
        )

    def set_patch(self, patch: NDArray | None):
        if patch is not None:
            patch = np.asarray(patch, copy=False)
            if patch.shape != (len(self),):
                raise ValueError("'patch' has an invalid shape")
            patch = patch.astype(np.int32, casting="same_kind", copy=False)
        self.patch = patch

    def split_patches(self) -> dict[int, DataChunk]:
        if self.patch is None:
            raise ValueError("'patch' not provided")

        idx_sort = np.argsort(self.patch)
        patch_sorted = self.patch[idx_sort]
        idx_split = np.where(np.diff(patch_sorted) != 0)[0] + 1

        coords = split_array(self.coords.values, idx_sort, idx_split)
        weight = split_array(self.weight, idx_sort, idx_split)
        redshift = split_array(self.redshift, idx_sort, idx_split)

        chunks = {}
        for i, pid in enumerate(np.unique(patch_sorted)):
            chunks[int(pid)] = DataChunk(
                coords=CoordsSky(coords[i]),
                weight=weight[i],
                redshift=redshift[i],
            )
        return chunks


class ArrayCache:
    def __init__(self):
        self._shards = []

    def append(self, data: NDArray) -> None:
        data = np.asarray(data, copy=False)
        self._shards.append(data)

    def get_values(self) -> NDArray:
        return np.concatenate(self._shards, dtype=np.float64)

    def clear(self) -> None:
        self._shards = []


class PatchWriter:
    def __init__(self, path: TypePathStr, chunksize: int = 65_536) -> None:
        self.path = Path(path)
        if self.path.exists():
            raise FileExistsError(f"directory already exists: {self.path}")
        self.path.mkdir(parents=True)

        self.chunksize = chunksize
        self._cachesize = 0
        self._caches: dict[str, ArrayCache] = {}

    def process_chunk(self, chunk: DataChunk) -> None:
        has_weight = chunk.weight is not None
        has_redshift = chunk.redshift is not None

        if len(self._caches) == 0:
            self._caches["coords"] = ArrayCache()
            if has_weight:
                self._caches["weight"] = ArrayCache()
            if has_redshift:
                self._caches["redshift"] = ArrayCache()

        self._caches["coords"].append(chunk.coords.values)
        if "weight" in self._caches:
            if not has_weight:
                raise ValueError("chunk has no 'weight' attached")
            self._caches["weight"].append(chunk.weight)
        if "redshift" in self._caches:
            if not has_redshift:
                raise ValueError("chunk has no 'redshift' attached")
            self._caches["redshift"].append(chunk.redshift)

        self._cachesize += len(chunk)
        if self._cachesize > self.chunksize:
            self.flush()

    def flush(self):
        for key, cache in self._caches.items():
            with open(self.path / key, mode="a") as f:
                cache.get_values().tofile(f)
            cache.clear()
        self._cachesize = 0

    def finalize(self) -> None:
        self.flush()
        Patch(self.path)  # computes metadata


META_FILE_NAME = "meta.json"


@dataclass
class Metadata:
    num_records: int
    total: float
    center: CoordsSky
    radius: DistsSky

    @classmethod
    def compute(cls, coords: CoordsSky, weight: NDArray |  None = None) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weight is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weight))

        new.center = coords.mean()
        new.radius = coords.distance(new.center).max().to_sky()
        return new

    @classmethod
    def from_file(cls, cache_path: TypePathStr) -> Metadata:
        with open(Path(cache_path) / META_FILE_NAME) as f:
            meta: dict = json.load(f)
        center = CoordsSky(meta.pop("center"))
        radius = DistsSky(meta.pop("radius"))
        return cls(center=center, radius=radius, **meta)

    def to_file(self, cache_path: TypePathStr) -> None:
        meta = dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.to_sky().values.tolist(),
            radius=self.radius.values.tolist(),
        )
        with open(Path(cache_path) / META_FILE_NAME, "w") as f:
            json.dump(meta, f)


class Patch(Sized):
    meta: Metadata

    def __init__(self, cache_path: TypePathStr) -> None:
        self.cache_path = Path(cache_path)
        try:
            self.meta = Metadata.from_file(self.cache_path)
        except FileNotFoundError:
            self.meta = Metadata.compute(self.coords, self.weight)
            self.meta.to_file(self.cache_path)

    def __len__(self) -> int:
        return self.meta.num_records

    @property
    def coords(self) -> CoordsSky:
        data = np.fromfile(self.cache_path / "coords").reshape((-1, 2))
        return CoordsSky(data)

    @property
    def weight(self) -> NDArray | None:
        if self.has_weight():
            return np.fromfile(self.cache_path / "weight")
        return None

    @property
    def redshift(self) -> NDArray | None:
        if self.has_redshift():
            return np.fromfile(self.cache_path / "redshift")
        return None

    def has_weight(self) -> bool:
        path = self.cache_path / "weight"
        return path.exists()

    def has_redshift(self) -> bool:
        path = self.cache_path / "redshift"
        return path.exists()
