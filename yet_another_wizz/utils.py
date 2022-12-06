from __future__ import annotations

import json
import operator
from datetime import timedelta
from timeit import default_timer

import numpy as np
from numpy.typing import NDArray


TypePatchKey = tuple[int, int]
TypeScaleKey = str


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
