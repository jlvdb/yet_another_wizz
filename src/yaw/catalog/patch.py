from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import DataChunk, JsonSerialisable
from yaw.coordinates import CoordsSky, DistsSky

__all__ = [
    "PatchWriter",
    "Patch",
]

Tpath = Union[Path, str]


class ArrayBuffer:
    __slots__ = ("_shards")

    def __init__(self):
        self._shards = []

    def is_empty(self) -> bool:
        return len(self._shards) == 0

    def append(self, data: NDArray) -> None:
        data = np.asarray(data)
        self._shards.append(data)

    def get_values(self) -> NDArray:
        return np.concatenate(self._shards)

    def clear(self) -> None:
        self._shards = []


class PatchWriter:
    __slots__ = ("cache_path", "coords", "weights", "redshifts", "chunksize", "cachesize")

    def __init__(self, cache_path: Tpath, chunksize: int = 65_536) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)

        self.chunksize = chunksize
        self.cachesize = 0

        self.coords = ArrayBuffer()
        self.weights = ArrayBuffer()
        self.redshifts = ArrayBuffer()

    def process_chunk(self, chunk: DataChunk) -> None:
        coords = chunk.coords.data
        self.coords.append(coords)

        weights = chunk.weights
        if weights is not None:
            self.weights.append(weights)

        redshifts = chunk.redshifts
        if redshifts is not None:
            self.redshifts.append(redshifts)

        self.cachesize += len(coords)
        if self.cachesize > self.chunksize:
            self.flush()

    def flush(self) -> None:
        def flush_cache(cache: ArrayBuffer, cache_path: Path) -> None:
            with cache_path.open(mode="a") as f:
                cache.get_values().tofile(f)
            cache.clear()

        if self.cachesize > 0:
            flush_cache(self.coords, self.cache_path / "coords")
            if not self.weights.is_empty():
                flush_cache(self.weights, self.cache_path / "weights")
            if not self.redshifts.is_empty():
                flush_cache(self.redshifts, self.cache_path / "redshifts")
            self.cachesize = 0

    def finalize(self) -> None:
        self.flush()


@dataclass
class Metadata(JsonSerialisable):
    num_records: int
    total: float
    center: CoordsSky
    radius: DistsSky

    @classmethod
    def compute(cls, coords: CoordsSky, weights: NDArray | None = None) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        new.center = coords.mean()
        new.radius = coords.distance(new.center).max().to_sky()
        return new

    @classmethod
    def from_dict(cls, kwarg_dict: dict) -> Metadata:
        center = CoordsSky(kwarg_dict.pop("center"))
        radius = DistsSky(kwarg_dict.pop("radius"))
        return cls(center=center, radius=radius, **kwarg_dict)

    def to_dict(self) -> dict:
        return dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.to_sky().tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


class Patch(Sized):
    meta: Metadata

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.json"
        try:
            self.meta = Metadata.from_file(meta_data_file)
        except FileNotFoundError:
            self.meta = Metadata.compute(self.coords, self.weights)
            self.meta.to_file(meta_data_file)

    def __getstate__(self) -> dict:
        return dict(cache_path=self.cache_path, meta=self.meta)

    def __setstate__(self, state) -> None:
        self.__dict__.update(state)

    def __len__(self) -> int:
        return self.meta.num_records

    def has_weights(self) -> bool:
        path = self.cache_path / "weights"
        return path.exists()

    def has_redshifts(self) -> bool:
        path = self.cache_path / "redshifts"
        return path.exists()

    @property
    def coords(self) -> CoordsSky:
        data = np.fromfile(self.cache_path / "coords")
        return CoordsSky(data.reshape((-1, 2)))

    @property
    def weights(self) -> NDArray | None:
        if self.has_weights():
            return np.fromfile(self.cache_path / "weights")
        return None

    @property
    def redshifts(self) -> NDArray | None:
        if self.has_redshifts():
            return np.fromfile(self.cache_path / "redshifts")
        return None

    def get_trees(self) -> BinnedTrees:
        return BinnedTrees(self)
