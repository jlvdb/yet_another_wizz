from __future__ import annotations

from typing import TYPE_CHECKING, Generator

import numpy as np

from yaw.core.coordinates import Coord3D, Coordinate, DistSky

from ._utils import _compute_center, _compute_radius, _minmax

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike, NDArray

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
