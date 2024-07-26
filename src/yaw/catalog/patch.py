from __future__ import annotations

import json
from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import DataChunk
from yaw.coordinates import CoordsSky, DistsSky

__all__ = [
    "PatchWriter",
    "Patch",
]

Tpath = Union[Path, str]


class ArrayBuffer:
    def __init__(self):
        self._shards = []

    def append(self, data: NDArray) -> None:
        data = np.asarray(data)
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
        self._caches: dict[str, ArrayBuffer] = {}

    def _init_caches(self, chunk: DataChunk) -> None:
        self._caches["coords"] = ArrayBuffer()
        if chunk.weights is not None:
            self._caches["weights"] = ArrayBuffer()
        if chunk.redshifts is not None:
            self._caches["redshifts"] = ArrayBuffer()

    def process_chunk(self, chunk: DataChunk) -> None:
        if len(self._caches) == 0:
            self._init_caches(chunk)

        for attr, cache in self._caches.items():
            values = getattr(chunk, attr)
            if values is None:
                raise ValueError(f"chunk has no '{attr}' attached")
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
            center=self.center.to_sky().tolist(),
            radius=self.radius.tolist(),
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
            self.meta = Metadata.compute(self.coords, self.weights)
            self.meta.to_file(meta_data_file)

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
