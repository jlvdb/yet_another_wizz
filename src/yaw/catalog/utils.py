from __future__ import annotations

import pickle
from collections.abc import Iterable, Sized
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator

import numpy as np

from yaw.core.utils import TypePathStr

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import ArrayLike, DTypeLike, NDArray

__all__ = [
    "memmap_init",
    "memmap_load",
    "memmap_resize",
    "concat_numpy_dicts",
    "groupby_arrays",
    "DataChunk",
    "IndexMapper",
    "check_optional_args",
    "check_arrays_matching_shape",
    "read_pickle",
    "write_pickle",
    "patch_path_from_id",
    "patch_id_from_path",
]


# wrap C extensions and define python fallback function

try:
    from ._groupby import _groupby_arrays

    def groupby_arrays(
        patch: NDArray[np.float64],
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.float64] | None = None,
        redshift: NDArray[np.float64] | None = None,
    ) -> tuple[
        dict[int, NDArray[np.float64]],
        dict[int, NDArray[np.float64]],
        dict[int, NDArray[np.float64]] | None,
        dict[int, NDArray[np.float64]] | None,
    ]:
        # ensure types
        patch = patch.astype(np.int64, copy=False, casting="same_kind")
        ra = ra.astype(np.float64, copy=False)
        dec = dec.astype(np.float64, copy=False)
        # TODO: handling of optional arguments
        if weight is None:
            weight = np.empty(len(patch))
        else:
            weight = weight.astype(np.float64, copy=False)
        if redshift is None:
            redshift = np.empty(len(patch))
        else:
            redshift = redshift.astype(np.float64, copy=False)
        result = _groupby_arrays(patch, ra, dec, weight, redshift)
        return (
            result[0],
            result[1],
            result[2] if weight is not None else None,
            result[3] if redshift is not None else None,
        )

except ImportError:
    import warnings

    warnings.warn("compiled ._groupby extension not availble, performance degraded")

    def groupby_arrays(
        patch: NDArray[np.float64],
        ra: NDArray[np.float64],
        dec: NDArray[np.float64],
        weight: NDArray[np.float64] | None = None,
        redshift: NDArray[np.float64] | None = None,
    ) -> tuple[
        dict[int, NDArray[np.float64]],
        dict[int, NDArray[np.float64]],
        dict[int, NDArray[np.float64]] | None,
        dict[int, NDArray[np.float64]] | None,
    ]:
        order = patch.argsort()
        items, _split = np.unique(patch[order], return_index=True)
        split = _split[1:]
        grouped = [
            {pid: pdata for pid, pdata in zip(split, ra[order])},
            {pid: pdata for pid, pdata in zip(split, dec[order])},
        ]
        if weight is not None:
            grouped.append({pid: pdata for pid, pdata in zip(split, weight[order])})
        if redshift is not None:
            grouped.append({pid: pdata for pid, pdata in zip(split, redshift[order])})
        return tuple(grouped)


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


def concat_numpy_dicts(dicts: Iterable[dict[str, NDArray]]) -> dict[str, NDArray]:
    chunk_iter = iter(dicts)
    chunk_dict = {key: [data] for key, data in next(chunk_iter).items()}
    for chunk in chunk_iter:
        for col, chunk_list in chunk_dict.items():
            chunk_list.append(chunk[col])
    return {col: np.concatenate(data) for col, data in chunk_dict.items()}


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

    def set_patch(self, patch: ArrayLike) -> None:
        patch_ids = np.asarray(patch)
        if patch_ids.shape != (len(self),):
            raise ValueError(
                f"failed to convert patch IDs to array with shape ({len(self)},)"
            )
        self.patch = patch_ids

    def groupby(self, ordered: bool = False) -> Generator[tuple[int, DataChunk]]:
        # run the groupby and construct the final result
        grp_ra, grp_dec, grp_weight, grp_redshift = groupby_arrays(
            self.patch, self.ra, self.dec, self.weight, self.redshift
        )
        # the keys are guaranteed to be the same
        patch_ids = sorted(grp_ra.keys()) if ordered else grp_ra.keys()
        for patch_id in patch_ids:
            weight = None if self.weight is None else grp_weight[patch_id]
            redshift = None if self.redshift is None else grp_redshift[patch_id]
            chunk = self.__class__(
                ra=grp_ra[patch_id],
                dec=grp_dec[patch_id],
                weight=weight,
                redshift=redshift,
            )
            yield patch_id, chunk

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


class IndexMapper:
    def __init__(self, indices: NDArray[np.int64]) -> None:
        self.reset()
        self.idx = indices

    def reset(self) -> None:
        self.recorded = 0

    def map(self, data: Sized) -> NDArray[np.int64]:
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
