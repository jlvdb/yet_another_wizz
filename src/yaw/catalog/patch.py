from __future__ import annotations

import json
import pickle
from collections.abc import Iterable, Iterator, Sized
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Union

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import AngularTree
from yaw.catalog.utils import DataChunk, Tclosed, groupby_binning
from yaw.coordinates import CoordsSky, DistsSky

__all__ = [
    "PatchWriter",
    "Patch",
    "BinnedTree",
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
    _trees: BinnedTrees | None

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.json"
        try:
            self.meta = Metadata.from_file(meta_data_file)
        except FileNotFoundError:
            self.meta = Metadata.compute(self.coords, self.weights)
            self.meta.to_file(meta_data_file)
        self._trees = None

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

    def get_trees(
        self,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = "left",
        leafsize: int = 16,
        force_build: bool = False,
    ) -> BinnedTrees:
        kwargs = dict(binning=binning, closed=closed, leafsize=leafsize, force=force_build)

        if self._trees is None:
            try:
                self._trees = BinnedTrees(self)
            except FileNotFoundError:
                self._trees = BinnedTrees.create(self, **kwargs)
                return self._trees

        self._trees.rebuild(**kwargs)
        return self._trees


def build_binned_trees(
    patch: Patch,
    binning: NDArray,
    closed: str,
    leafsize: int,
) -> tuple[AngularTree]:
    if not patch.has_redshifts():
        raise ValueError("patch has no 'redshifts' attached")

    trees = []
    for _, bin_data in groupby_binning(
        patch.redshifts,
        binning,
        closed=closed,
        coords=patch.coords,
        weights=patch.weights,
    ):
        bin_data["coords"] = CoordsSky(bin_data["coords"])
        tree = AngularTree(**bin_data, leafsize=leafsize)
        trees.append(tree)
    return tuple(trees)


class BinnedTrees(Iterable):
    _patch: Patch

    def __init__(self, patch: Patch) -> None:
        self._patch = patch
        if not self.binning_file.exists():
            raise FileNotFoundError(f"no trees found for patch at '{self.cache_path}'")

        with self.binning_file.open() as f:
            binning = json.load(f)
        if binning is None:
            self.binning = None
        else:
            self.binning = np.asarray(binning)

    @classmethod
    def create(
        cls,
        patch: Patch,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = "left",
        leafsize: int = 16,
        force: bool = False,
    ) -> BinnedTrees:
        new = cls.__new__(cls)
        new._patch = patch
        new.rebuild(binning, closed=closed, leafsize=leafsize, force=force)
        return new

    def rebuild(
        self,
        binning: NDArray | None = None,
        *,
        closed: Tclosed = "left",
        leafsize: int = 16,
        force: bool = False,
    ) -> None:
        if binning is not None:
            binning = np.asarray(binning, dtype=np.float64)
        if not force and self.binning_file.exists() and self.binning_equal(binning):
            return

        with self.trees_file.open(mode="wb") as f:
            patch = self._patch
            if binning is not None:
                trees = build_binned_trees(patch, binning, closed, leafsize)
            else:
                trees = AngularTree(patch.coords, patch.weights, leafsize=leafsize)
            pickle.dump(trees, f)

        with self.binning_file.open(mode="w") as f:
            try:
                json.dump(binning.tolist(), f)
            except AttributeError:
                json.dump(binning, f)
        self.binning = binning

    @property
    def cache_path(self) -> Path:
        return self._patch.cache_path

    @property
    def binning_file(self) -> Path:
        return self.cache_path / "binning.json"

    @property
    def trees_file(self) -> Path:
        return self.cache_path / "trees.pkl"

    def is_binned(self) -> bool:
        return self.binning is not None

    def binning_equal(self, binning: NDArray | None) -> bool:
        if self.binning is None and binning is None:
            return True
        elif np.array_equal(self.binning, binning):
            return True
        return False

    @property
    def trees(self) -> AngularTree | tuple[AngularTree]:
        with self.trees_file.open(mode="rb") as f:
            return pickle.load(f)

    def __iter__(self) -> Iterator[AngularTree]:
        if self.is_binned():
            yield from self.trees
        else:
            yield from repeat(self.trees)

    def count_binned(
        self,
        other: BinnedTrees,
        ang_min: NDArray,
        ang_max: NDArray,
        weight_scale: float | None = None,
        weight_res: int = 50,
    ) -> NDArray[np.float64]:
        is_binned = (self.is_binned(), other.is_binned())
        if not any(is_binned):
            raise ValueError("at least one of the trees must be binned")
        elif all(is_binned) and not self.binning_equal(other.binning):
            raise ValueError("binning of trees does not match")

        binned_counts = []
        for tree_self, tree_other in zip(iter(self), iter(other)):
            counts = tree_self.count(
                tree_other,
                ang_min,
                ang_max,
                weight_scale=weight_scale,
                weight_res=weight_res,
            )
            binned_counts.append(counts)
        return np.transpose(binned_counts)
