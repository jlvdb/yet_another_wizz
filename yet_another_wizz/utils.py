from __future__ import annotations

import functools
import json
import multiprocessing
import operator
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from datetime import timedelta
from timeit import default_timer
from typing import Any

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
            print(self.msg, end="")
        self.t = default_timer()
        return self

    def __exit__(self, *args, **kwargs) -> None:
        delta = default_timer() - self.t
        if self.verbose:
            print(" - " + str(timedelta(seconds=round(delta))))


class ArrayDict(Mapping):

    def __init__(
        self,
        keys: Collection[Any],
        array: NDArray
    ) -> None:
        if len(array) != len(keys):
            raise ValueError("number of keys and array length do not match")
        self._array = array
        self._dict = {key: idx for idx, key in enumerate(keys)}

    def __len__(self) -> int:
        return len(self._array)

    def __getitem__(self, key: Any) -> NDArray:
        idx = self._dict[key]
        return self._array[idx]

    def __iter__(self) -> Iterator[NDArray]:
        return self._dict.__iter__()

    def __contains__(self, key: Any) -> bool:
        return key in self._dict

    def items(self) -> list[tuple[Any, NDArray]]:
        # ensure that the items are ordered by the index of each key
        return sorted(self._dict.items(), key=lambda item: item[1])

    def keys(self) -> list[Any]:
        # key are ordered by their corresponding index
        return [key for key, _ in self.items()]

    def values(self) -> list[NDArray]:
        # values are returned in index order
        return [value for value in self._array]

    def get(self, key: Any, default: Any) -> Any:
        try:
            idx = self._dict[key]
        except KeyError:
            return default
        else:
            return self._array[idx]

    def sample(self, keys: Iterable[Any]) -> NDArray:
        idx = [self._dict[key] for key in keys]
        return self._array[idx]

    def as_array(self) -> NDArray:
        return self._array


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


def _threadwrapper(arg_tuple, function):
    return function(*arg_tuple)


class ParallelHelper:
    """
    Helper class to apply a series of arguments to a function using
    multiprocessing.Pool.
    """

    def __init__(
        self,
        function: Callable,
        n_items: int,
        num_threads: int | None = None
    ) -> None:
        self.function = function
        self._n_items = n_items
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self._num_threads = num_threads
        self.args = []

    def n_jobs(self) -> int:
        """
        Returns the number of expected function arguments.
        """
        return self._n_items

    def n_args(self) -> int:
        return len(self.args)

    def n_threads(self):
        return self._num_threads

    def add_constant(self, value: Any) -> None:
        """
        Append a constant argument that will be repeated for each thread.
        """
        self.args.append([value] * self._n_items)

    def add_iterable(self, iterable: Collection) -> None:
        """
        Append a variable argument that will be iterated for each thread.
        """
        if len(iterable) != self._n_items:
            raise ValueError(
                f"length of iterable argument must be {self._n_items}")
        self.args.append(iterable)

    def run(self) -> list:
        """
        Apply the accumulated arguments to a function in a pool of threads.
        The threads are blocking until all results are received.
        """
        if self._num_threads > 1:
            with multiprocessing.Pool(self._num_threads) as pool:
                results = pool.starmap(self.function, zip(*self.args))
        else:  # mimic the behaviour of pool.starmap() with map()
            results = list(map(self.function, *self.args))
        return results

    def iter_result(self) -> Iterator:
        """
        Apply the accumulated arguments to a function in a pool of threads.
        The results are processed unordered and yielded as an iterator.
        """
        function = functools.partial(_threadwrapper, function=self.function)
        if self._num_threads > 1:
            with multiprocessing.Pool(self._num_threads) as pool:
                for result in pool.imap_unordered(function, zip(*self.args)):
                    yield result
        else:  # mimic the behaviour of pool.imap_unordered()
            for result in map(self.function, *self.args):
                yield result
