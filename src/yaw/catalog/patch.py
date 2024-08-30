from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from yaw.catalog.trees import BinnedTrees
from yaw.catalog.utils import DataChunk
from yaw.containers import Tpath, YamlSerialisable
from yaw.utils import AngularCoordinates, AngularDistances

__all__ = [
    "PatchWriter",
    "Patch",
]

CHUNKSIZE = 65_536
COLUMN_FILE_NAME = "patch.columns"
DATA_PATH = "data.bin"


def write_column_info(
    cache_path: Tpath, has_weights: bool, has_redshifts: bool
) -> None:
    info = (has_weights << 0) | (has_redshifts << 1)

    with open(Path(cache_path) / COLUMN_FILE_NAME, mode="wb") as f:
        info_bytes = info.to_bytes(1, byteorder="big")
        f.write(info_bytes)


def read_and_delete_column_info(cache_path: Tpath) -> tuple[bool, bool]:
    with open(Path(cache_path) / COLUMN_FILE_NAME, "rb") as f:
        info_bytes = f.read()
        info = int.from_bytes(info_bytes, byteorder="big")

    has_weights = info & (1 << 0)
    has_redshifts = info & (1 << 1)
    return has_weights, has_redshifts


def read_patch_data(
    cache_path: Tpath,
    has_weights: bool,
    has_redshifts: bool,
) -> DataChunk:
    columns = ["ra", "dec"]
    if has_weights:
        columns.append("weights")
    if has_redshifts:
        columns.append("redshifts")

    path = Path(cache_path) / DATA_PATH
    rawdata = np.fromfile(path)
    num_records = len(rawdata) // len(columns)

    dtype = np.dtype([(col, "f8") for col in columns])
    data = rawdata.view(dtype).reshape((num_records,))

    return DataChunk(data)


class PatchWriter:
    __slots__ = ("cache_path", "buffersize", "_cachesize", "_shards", "_file")

    def __init__(
        self,
        cache_path: Tpath,
        *,
        has_weights: bool,
        has_redshifts: bool,
        buffersize: int = -1,
    ) -> None:
        self.cache_path = Path(cache_path)
        if self.cache_path.exists():
            raise FileExistsError(f"directory already exists: {self.cache_path}")
        self.cache_path.mkdir(parents=True)
        self._file = None

        write_column_info(cache_path, has_weights, has_redshifts)

        self.buffersize = CHUNKSIZE if buffersize < 0 else int(buffersize)
        self._cachesize = 0
        self._shards = []

    def open(self) -> None:
        if self._file is None:
            self._file = open(self.cache_path / DATA_PATH, mode="ab")

    def close(self) -> None:
        self.flush()
        self._file.close()
        self._file = None

    def process_chunk(self, chunk: DataChunk) -> None:
        self._shards.append(chunk.data)
        self._cachesize += len(chunk)

        if self._cachesize >= self.buffersize:
            self.flush()

    def flush(self) -> None:
        if len(self._shards) > 0:
            self.open()  # ensure file is ready for writing

            data = np.concatenate(self._shards)
            self._shards = []

            data.tofile(self._file)

        self._cachesize = 0


@dataclass
class Metadata(YamlSerialisable):
    __slots__ = (
        "num_records",
        "total",
        "center",
        "radius",
        "has_weights",
        "has_redshifts",
    )

    def __init__(
        self,
        *,
        num_records: int,
        total: float,
        center: AngularCoordinates,
        radius: AngularDistances,
        has_weights: bool,
        has_redshifts: bool,
    ) -> None:
        self.num_records = num_records
        self.total = total
        self.center = center
        self.radius = radius
        self.has_weights = has_weights
        self.has_redshifts = has_redshifts

    @classmethod
    def compute(
        cls,
        coords: AngularCoordinates,
        *,
        weights: NDArray | None = None,
        redshifts: NDArray | None = None,
    ) -> Metadata:
        new = super().__new__(cls)
        new.num_records = len(coords)
        if weights is None:
            new.total = float(new.num_records)
        else:
            new.total = float(np.sum(weights))

        new.center = coords.mean(weights)
        new.radius = coords.distance(new.center).max()

        new.has_weights = weights is not None
        new.has_redshifts = redshifts is not None
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
            has_weights=bool(self.has_weights),
            has_redshifts=bool(self.has_redshifts),
        )


class Patch:
    __slots__ = ("meta", "cache_path")

    def __init__(self, cache_path: Tpath) -> None:
        self.cache_path = Path(cache_path)
        meta_data_file = self.cache_path / "meta.yml"

        try:
            self.meta = Metadata.from_file(meta_data_file)

        except FileNotFoundError:
            has_weights, has_redshifts = read_and_delete_column_info(self.cache_path)
            data = read_patch_data(self.cache_path, has_weights, has_redshifts)

            self.meta = Metadata.compute(
                data.coords, weights=data.weights, redshifts=data.redshifts
            )
            self.meta.to_file(meta_data_file)

    def __getstate__(self) -> dict:
        return dict(cache_path=self.cache_path, meta=self.meta)

    def __setstate__(self, state) -> None:
        for key, value in state.items():
            setattr(self, key, value)

    def load_data(self) -> DataChunk:
        return read_patch_data(
            self.cache_path, self.meta.has_weights, self.meta.has_redshifts
        )

    @property
    def coords(self) -> AngularCoordinates:
        return self.load_data().coords

    @property
    def weights(self) -> NDArray | None:
        return self.load_data().weights

    @property
    def redshifts(self) -> NDArray | None:
        return self.load_data().redshifts

    def get_trees(self) -> BinnedTrees:
        return BinnedTrees(self)
