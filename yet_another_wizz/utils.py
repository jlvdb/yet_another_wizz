from __future__ import annotations

import json
import operator
import os
from collections.abc import Iterable, Iterator
from datetime import timedelta
from timeit import default_timer
from typing import Any

import bloscpack
import numpy as np
from numpy.typing import NDArray


TypePatchKey = tuple[int, int]
TypeScaleKey = str


def array_groupby(
    by: NDArray,
    arrays: Iterable[NDArray | None]
) -> Iterator[tuple[Any, tuple[NDArray | None]]]:
    order = by.argsort()
    keys, idx_split = np.unique(by[order], return_index=True)
    splitted = []
    for arr in arrays:
        if arr is None:
            splitted.append([None] * len(keys))
        else:
            splitted.append(np.split(arr[order], idx_split[1:]))
    for i, key in enumerate(keys):
        yield key, tuple(split[i] for split in splitted)


def optional_attribute_path(
    directory: str | None,
    name: str,
    ext: str = ".blosc"
) -> str | None:
    if directory is None:
        return None
    else:
        return os.path.join(directory, name + ext)


def read_blosc(fpath: str) -> NDArray:
    return bloscpack.unpack_ndarray_from_file(fpath)


def write_blosc(data: NDArray, fpath: str) -> None:
    bloscpack.pack_ndarray_to_file(data, fpath)


class DynamicAttribute:

    def __init__(
        self,
        data: NDArray | None,
        path: str | None = None
    ) -> None:
        self._data = data
        self.path = path

    @classmethod
    def from_file(cls, path: str) -> DynamicAttribute:
        data = write_blosc(path)
        return cls(data, path)

    def is_loaded(self) -> bool:
        return hasattr(self, "_data")

    def load(self) -> bool:
        loaded = self.is_loaded()
        if not loaded:
            self._data = read_blosc(self.path)
        return loaded

    def unload(self) -> bool:
        loaded = self.is_loaded()
        if loaded:
            if self._data is not None and self.path is not None:
                if not os.path.exists(self.path):
                    write_blosc(self._data, self.path)
                delattr(self, "_data")

    @property
    def data(self) -> NDArray:
        loaded = self.load()
        data = self._data
        if not loaded:  # restore original state
            self.unload()
        return data


class LimitTracker:

    def __init__(self):
        self.min = +np.inf
        self.max = -np.inf

    def update(self, data: NDArray | None):
        if data is not None:
            self.min = np.minimum(self.min, np.min(data))
            self.max = np.maximum(self.max, np.max(data))

    def get(self):
        vmin = None if np.isinf(self.min) else self.min
        vmax = None if np.isinf(self.max) else self.max
        return vmin, vmax


class Timed:

    def __init__(
        self,
        msg: str | None = None,
        verbose: bool = True
    ) -> None:
        self.verbose = verbose
        self.msg = msg

    def __enter__(self) -> Timed:
        if self.verbose and self.msg is not None:
            print(f"-:--:-- {self.msg} ...", end="\r")
        self.t = default_timer()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        delta = default_timer() - self.t
        if self.verbose:
            time = str(timedelta(seconds=round(delta)))
            print(f"{time} {self.msg} - Done")


def scales_to_keys(scales: NDArray[np.float_]) -> list[TypeScaleKey]:
    return [f"kpc{scale[0]:.0f}t{scale[1]:.0f}" for scale in scales]


def load_json(path):
    with open(path) as f:
        data_dict = json.load(f)
        # convert lists to numpy arrays
        for key, value in data_dict.items():
            if type(value) is list:
                data_dict[key] = np.array(value)
    return data_dict


def dump_json(data, path, preview=False):
    kwargs = dict(indent=4, default=operator.methodcaller("tolist"))
    if preview:
        print(json.dumps(data, **kwargs))
    else:
        with open(path, "w") as f:
            json.dump(data, f, **kwargs)
