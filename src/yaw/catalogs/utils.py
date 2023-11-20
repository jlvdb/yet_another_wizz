from __future__ import annotations

import pickle
from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from yaw.core.coordinates import Coord3D, Coordinate, DistSky
from yaw.core.utils import TypePathStr

from ._utils import _compute_center, _compute_radius, _minmax

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike, NDArray
    from polars import DataFrame

__all__ = []  # TODO

# type annotations for C code


def compute_center(ra: NDArray[np.float64], dec: NDArray[np.float64]) -> Coord3D:
    xyz = _compute_center(ra, dec)
    return Coord3D(*xyz)


def compute_radius(
    ra: NDArray[np.float64], dec: NDArray[np.float64], coord: Coordinate
) -> DistSky:
    coord_3d = coord.to_3d()
    dist = _compute_radius(ra, dec, coord_3d.x, coord_3d.y, coord_3d.z)
    return DistSky(dist)


def minmax(array: NDArray[np.float64 | np.float32]) -> tuple[float, float]:
    return _minmax(array)


# python functions


def memmap_init(path: str, dtype: DTypeLike, shape: tuple[int] | int) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="w+", shape=shape)


def memmap_load(path: str, dtype: DTypeLike, readonly: bool = True) -> np.memmap:
    return np.memmap(path, dtype=dtype, mode="r" if readonly else "r+")


def memmap_resize(memmap: np.memmap, new_shape: tuple[int] | int) -> np.memmap:
    itemsize = memmap.itemsize
    if isinstance(new_shape, int):
        new_shape *= itemsize
    else:
        new_shape = tuple(n * itemsize for n in new_shape)
    memmap.base.resize(new_shape)
    memmap.flush()
    # reopen
    path = memmap.filename
    dtype = memmap.dtype
    readonly = memmap.mode == "r"
    del memmap
    return memmap_load(path, dtype, readonly)


def dataframe_to_numpy_dict(dataframe: DataFrame) -> dict[str, NDArray]:
    the_dict = {}
    for col in dataframe.columns:
        the_dict[col] = dataframe[col].to_numpy()
    return the_dict


def concat_numpy_dicts(dicts: Iterable[dict[str, NDArray]]) -> dict[str, NDArray]:
    chunk_iter = iter(dicts)
    chunk_dict = {key: [data] for key, data in next(chunk_iter).items()}
    for chunk in chunk_iter:
        for col, chunk_list in chunk_dict.items():
            chunk_list.append(chunk[col])
    return {col: np.concatenate(data) for col, data in chunk_dict.items()}


def groupby(
    index: NDArray[np.int64], **arrays: NDArray | None
) -> Generator[tuple[int, dict[str, NDArray]]]:
    order = index.argsort()
    items, _split = np.unique(index[order], return_index=True)
    split = _split[1:]
    grouped = {
        col: np.split(data[order], split)
        for col, data in arrays.items()
        if data is not None
    }
    for i, key in enumerate(items):
        yield key, {col: gdata[i] for col, gdata in grouped.items()}


@dataclass
class DataChunk:
    ra: NDArray
    dec: NDArray
    weight: NDArray | None = field(default=None)
    redshift: NDArray | None = field(default=None)
    patch: NDArray | None = field(default=None)

    @classmethod
    def from_chunks(cls, chunks: Iterable[DataChunk]) -> DataChunk:
        merged = concat_numpy_dicts([chunk.to_dict() for chunk in chunks])
        return cls(**merged)

    def __len__(self) -> int:
        return len(self.ra)

    @property
    def size(self) -> int:
        return sum(val.size for val in self.to_dict().values())

    def __getitem__(self, index) -> DataChunk:
        kwargs = {key: values[index] for key, values in self.to_dict().items()}
        return DataChunk(**kwargs)

    def groupby(self) -> Generator[tuple[int, DataChunk]]:
        col_kwargs = self.to_dict(drop_patch=True)
        for patch_id, patch_data in groupby(self.patch, **col_kwargs):
            yield patch_id, self.__class__(**patch_data)

    def to_dict(self, drop_patch: bool = False) -> DataChunk:
        the_dict = dict(ra=self.ra, dec=self.dec)
        if self.weight is not None:
            the_dict["weight"] = self.weight
        if self.redshift is not None:
            the_dict["redshift"] = self.redshift
        if not drop_patch and self.patch is not None:
            the_dict["patch"] = self.patch
        return the_dict

    @classmethod
    def from_dict(cls, the_dict: DataChunk) -> DataChunk:
        return cls(**the_dict)

    @classmethod
    def from_dataframe(cls, dataframe: DataFrame) -> DataChunk:
        return cls(**dataframe_to_numpy_dict(dataframe))


class IndexMapper:
    def __init__(self, indices: NDArray[np.int_]) -> None:
        self.reset()
        self.idx = indices

    def reset(self) -> None:
        self.recorded = 0

    def map(self, data: Sized) -> NDArray[np.int_]:
        # slide the index window according to the input data sample
        start = self.recorded
        end = start + len(data)
        self.recorded = end  # future state
        # pick the indices that fall into the current data range
        indices = np.compress((self.idx >= start) & (self.idx < end), self.idx)
        return indices - start


def check_optional_args(
    optional_expected: bool,
    optional_provided: bool,
    annotation: str,
) -> None:
    if optional_expected == optional_provided:
        return
    elif optional_expected and not optional_provided:
        raise ValueError(f"'{annotation}' expected but not provided")
    else:
        raise ValueError(f"got unexpected optional '{annotation}'")


def check_arrays_matching_shape(
    array: NDArray,
    *arrays: NDArray | None,
    ndim: int | None = None,
) -> None:
    if ndim is not None and array.ndim != ndim:
        raise IndexError(f"expected {ndim}-dim array")
    shape = array.shape
    for arr in arrays:
        if arr is None:
            continue
        if arr.shape != shape:
            raise IndexError("array lengths do not match")


def read_pickle(path: TypePathStr) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def write_pickle(path: TypePathStr, data: Any) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def patch_path_from_id(cache_directory: TypePathStr, patch_id: int) -> Path:
    return Path(cache_directory) / f"patch_{patch_id:03d}"


def patch_id_from_path(directory: TypePathStr) -> int:
    if not directory.match("patch_*"):
        raise ValueError(f"'directory' does not match 'patch_*': {directory}")
    _, id_str = directory.name.split("_")
    return int(id_str)
