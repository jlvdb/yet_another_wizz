from __future__ import annotations

from collections.abc import Sized
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import DataChunk
from yaw.containers import JsonSerialisable, Serialisable, Tpath
from yaw.utils import AngularCoordinates, AngularDistances

__all__ = [
    "PatchWriter",
    "Patch",
]

CHUNKSIZE = 65_536


class ArrayWriter:
    __slots__ = ("path", "_cachesize", "chunksize", "_shards")

    def __init__(self, path: Tpath, *, chunksize: int = CHUNKSIZE):
        self.path = path
        self._cachesize = 0
        self.chunksize = chunksize
        self._shards = []

    def append(self, data: NDArray) -> None:
        data = np.asarray(data)
        self._shards.append(data)
        self._cachesize += len(data)

        if self._cachesize >= self.chunksize:
            self.flush()

    def flush(self) -> None:
        if len(self._shards) > 0:
            data = np.concatenate(self._shards)
            with self.path.open("a") as f:
                data.tofile(f)
            self._shards = []

        self._cachesize = 0


class PatchWriter:
    __slots__ = ("cache_path", "coords", "weights", "redshifts")

    def __init__(self, cache_path: Tpath, chunksize: int = CHUNKSIZE) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)

        self.coords = ArrayWriter(self.cache_path / "coords", chunksize=chunksize)
        self.weights = ArrayWriter(self.cache_path / "weights", chunksize=chunksize)
        self.redshifts = ArrayWriter(self.cache_path / "redshifts", chunksize=chunksize)

    def process_chunk(self, chunk: DataChunk) -> None:
        coords = chunk.coords.data
        self.coords.append(coords)

        weights = chunk.weights
        if weights is not None:
            self.weights.append(weights)

        redshifts = chunk.redshifts
        if redshifts is not None:
            self.redshifts.append(redshifts)

    def finalize(self) -> None:
        self.coords.flush()
        if self.weights:
            self.weights.flush()
        if self.redshifts:
            self.redshifts.flush()


@dataclass
class Metadata(JsonSerialisable):
    __slots__ = ("num_records", "total", "center", "radius")

    def __init__(
        self,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius

    @classmethod
    def compute(
        cls, coords: AngularCoordinates, weights: NDArray | None = None
    ) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()
        return new

    @classmethod
    def from_dict(cls, kwarg_dict: dict) -> Metadata:
        center = AngularCoordinates(kwarg_dict.pop("center"))
        radius = AngularDistances(kwarg_dict.pop("radius"))
        return cls(center=center, radius=radius, **kwarg_dict)

    def to_dict(self) -> dict:
        return dict(
            num_records=int(self.num_records),
            total=float(self.total),
            center=self.center.tolist()[0],  # 2-dim by default
            radius=self.radius.tolist()[0],  # 1-dim by default
        )


class Patch(Sized, Serialisable):
    __slots__ = ("meta", "cache_path")

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
        for key, value in state.items():
            setattr(self, key, value)

    def __len__(self) -> int:
        return self.meta.num_records

    def to_dict(self) -> dict[str, Any]:
        return self.__getstate__()

    def has_weights(self) -> bool:
        path = self.cache_path / "weights"
        return path.exists()

    def has_redshifts(self) -> bool:
        path = self.cache_path / "redshifts"
        return path.exists()

    @property
    def coords(self) -> AngularCoordinates:
        data = np.fromfile(self.cache_path / "coords")
        return AngularCoordinates(data.reshape((-1, 2)))

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
