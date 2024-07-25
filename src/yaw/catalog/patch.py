from __future__ import annotations

import json
from collections.abc import Sequence, Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator, Literal, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from yaw.catalog.coordinates import CoordsSky, DistsSky

__all__ = [
    "DataChunk",
    "PatchWriter",
    "Patch",
]

Tpath = Union[Path, str]


def groupby_value(
    values: NDArray,
    **optional_arrays: NDArray | None,
) -> Generator[tuple[Any, dict[str, NDArray]], None, None]:
    idx_sort = np.argsort(values)
    values_sorted = values[idx_sort]
    uniques, _idx_split = np.unique(values_sorted, return_index=True)
    idx_split = _idx_split[1:]

    splitted_arrays = {}
    for name, array in optional_arrays.items():
        if array is not None:
            array_sorted = array[idx_sort]
            splitted_arrays[name] = np.split(array_sorted, idx_split)

    for i, value in enumerate(uniques):
        yield value, {name: splits[i] for name, splits in splitted_arrays.items()}


def groupby_binning(
    values: NDArray,
    binning: NDArray,
    closed: Literal["left", "right"] = "left",
    **optional_arrays: NDArray | None,
) -> Generator[tuple[NDArray, dict[str, NDArray]], None, None]:
    binning = np.asarray(binning)
    bin_idx = np.digitize(values, binning, right=(closed == "right"))
    for i, bin_array in groupby_value(bin_idx, **optional_arrays):
        if i == 0 or i == len(binning):  # skip values outside of binning range
            continue
        yield binning[i - 1 : i + 1], bin_array


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
        degrees: bool = True,
    ):
        if degrees:
            ra = np.deg2rad(ra)
            dec = np.deg2rad(dec)
        coords = CoordsSky(np.column_stack((ra, dec)))
        return cls(coords, weight, redshift, patch)

    @classmethod
    def from_chunks(cls, chunks: Sequence[DataChunk]) -> DataChunk:
        def concat_optional_attr(attr: str) -> NDArray | None:
            values = tuple(getattr(chunk, attr) for chunk in chunks)
            value_is_set = tuple(value is not None for value in values)
            if all(value_is_set):
                return np.concatenate(values)
            elif not any(value_is_set):
                return None
            raise ValueError(f"not all chunks have '{attr}' set")

        return DataChunk(
            coords=CoordsSky.from_coords(chunk.coords for chunk in chunks),
            weight=concat_optional_attr("weight"),
            redshift=concat_optional_attr("redshift"),
            patch=concat_optional_attr("patch"),
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
        chunks = {}
        for patch_id, attr_dict in groupby_value(
            self.patch,
            coords=self.coords.values,
            weight=self.weight,
            redshift=self.redshift,
        ):
            coords = CoordsSky(attr_dict.pop("coords"))
            chunks[int(patch_id)] = DataChunk(coords, **attr_dict)
        return chunks


class ArrayCache:
    def __init__(self):
        self._shards = []

    def append(self, data: NDArray) -> None:
        data = np.asarray(data, copy=False)
        self._shards.append(data)

    def get_values(self) -> NDArray:
        return np.concatenate(self._shards)

    def clear(self) -> None:
        self._shards = []


class PatchWriter:
    def __init__(self, cache_path: Tpath, chunksize: int = 65_536) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)

        self.chunksize = chunksize
        self._cachesize = 0
        self._caches: dict[str, ArrayCache] = {}

    def _init_caches(self, chunk: DataChunk) -> None:
        self._caches["coords"] = ArrayCache()
        if chunk.weight is not None:
            self._caches["weight"] = ArrayCache()
        if chunk.redshift is not None:
            self._caches["redshift"] = ArrayCache()

    def process_chunk(self, chunk: DataChunk) -> None:
        if len(self._caches) == 0:
            self._init_caches(chunk)

        for attr, cache in self._caches.items():
            values = getattr(chunk, attr)
            if values is None:
                raise ValueError(f"chunk has no '{attr}' attached")
            if attr == "coords":
                values = values.values
            cache.append(values)

        self._cachesize += len(chunk)
        if self._cachesize > self.chunksize:
            self.flush()

    def flush(self):
        for attr, cache in self._caches.items():
            cache_path = self.cache_path / attr
            with cache_path.open(mode="a") as f:
                cache.get_values().tofile(f)
            cache.clear()
        self._cachesize = 0

    def finalize(self) -> None:
        self.flush()
        Patch(self.cache_path)  # computes metadata


@dataclass
class Metadata:
    num_records: int
    total: float
    center: CoordsSky
    radius: DistsSky

    @classmethod
    def compute(cls, coords: CoordsSky, weight: NDArray | None = None) -> Metadata:
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
    def from_file(cls, fpath: Tpath) -> Metadata:
        with Path(fpath).open() as f:
            meta: dict = json.load(f)
        center = CoordsSky(meta.pop("center"))
        radius = DistsSky(meta.pop("radius"))
        return cls(center=center, radius=radius, **meta)

    def to_file(self, fpath: Tpath) -> None:
        meta = dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.to_sky().values.tolist(),
            radius=self.radius.values.tolist(),
        )
        with Path(fpath).open(mode="w") as f:
            json.dump(meta, f)


class Patch(Sized):
    meta: Metadata

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.json"
        try:
            self.meta = Metadata.from_file(meta_data_file)
        except FileNotFoundError:
            self.meta = Metadata.compute(self.coords, self.weight)
            self.meta.to_file(meta_data_file)

    def __len__(self) -> int:
        return self.meta.num_records

    def has_weight(self) -> bool:
        path = self.cache_path / "weight"
        return path.exists()

    def has_redshift(self) -> bool:
        path = self.cache_path / "redshift"
        return path.exists()

    @property
    def coords(self) -> CoordsSky:
        data = np.fromfile(self.cache_path / "coords")
        return CoordsSky(data.reshape((-1, 2)))

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
