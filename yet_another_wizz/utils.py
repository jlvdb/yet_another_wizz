from __future__ import annotations

import json
import multiprocessing
import operator
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import pandas as pd
from astropy.cosmology import FLRW, Planck15
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame

from yet_another_wizz.catalog import BinnedCatalog


def get_default_cosmology() -> FLRW:
    return Planck15


class CustomCosmology(ABC):
    """
    Can be used to implement a custom cosmology outside of astropy.cosmology
    """

    @abstractmethod
    def to_format(self, format: str = "mapping") -> str:
        # TODO: really necessary?
        raise NotImplementedError

    @abstractmethod
    def comoving_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    @abstractmethod
    def comoving_transverse_distance(self, z: ArrayLike) -> ArrayLike:
        raise NotImplementedError


def r_kpc_to_angle(
    r_kpc: NDArray[np.float_],
    z: float,
    cosmology: FLRW | CustomCosmology
) -> tuple[float, float]:
    f_K = cosmology.comoving_transverse_distance(z)  # for 1 radian in Mpc
    angle_rad = np.asarray(r_kpc) / 1000.0 * (1.0 + z) / f_K.value
    return np.rad2deg(angle_rad)


class UniformRandoms:

    def __init__(
        self,
        ra_min: float,
        ra_max: float,
        dec_min: float,
        dec_max: float,
        seed: int = 12345
    ) -> None:
        self.x_min, self.y_min = self.sky2cylinder(ra_min, dec_min)
        self.x_max, self.y_max = self.sky2cylinder(ra_max, dec_max)
        self.rng = np.random.SeedSequence(seed)

    @classmethod
    def from_catalogue(cls, cat: BinnedCatalog) -> UniformRandoms:
        return cls(
            np.rad2deg(cat.ra.min()),
            np.rad2deg(cat.ra.max()),
            np.rad2deg(cat.dec.min()),
            np.rad2deg(cat.dec.max()))

    @staticmethod
    def sky2cylinder(
        ra: float | NDArray[np.float_],
        dec: float | NDArray[np.float_]
    ) -> NDArray:
        x = np.deg2rad(ra)
        y = np.sin(np.deg2rad(dec))
        return np.transpose([x, y])
 
    @staticmethod
    def cylinder2sky(
        x: float | NDArray[np.float_],
        y: float | NDArray[np.float_]
    ) -> float | NDArray[np.float_]:
        ra = np.rad2deg(x)
        dec = np.rad2deg(np.arcsin(y))
        return np.transpose([ra, dec])

    def generate(
        self,
        size: int,
        names: list[str, str] | None = None,
        draw_from: dict[str, NDArray] | None = None,
        n_threads: int = 1
    ) -> DataFrame:
        seeds = self.rng.spawn(n_threads)
        if size <= 100 * n_threads:
            n_threads = 1
        sizes = np.diff(np.linspace(0, size, n_threads+1).astype(np.int_))
        args = []
        for i in range(n_threads):
            args.append([self, seeds[i], sizes[i], names, draw_from])
        with multiprocessing.Pool(n_threads) as pool:
            chunks = pool.starmap(_generate_uniform_randoms, args)
        return pd.concat(chunks)


def _generate_uniform_randoms(
    inst: UniformRandoms,
    seed: np.random.SeedSequence,
    size: int,
    names: list[str, str] | None = None,
    draw_from: dict[str, NDArray] | None = None
) -> DataFrame:
    rng = np.random.default_rng(seed)
    if names is None:
        names = ["ra", "dec"]
    # generate positions
    x = np.random.uniform(inst.x_min, inst.x_max, size)
    y = np.random.uniform(inst.y_min, inst.y_max, size)
    ra, dec = UniformRandoms.cylinder2sky(x, y).T
    rand = DataFrame({names[0]: ra, names[1]: dec})
    # generate random draw
    if draw_from is not None:
        N = None
        for col in draw_from.values():
            if N is None:
                if len(col.shape) > 1:
                    raise ValueError("data to draw from must be 1-dimensional")
                N = len(col)
            else:
                if len(col) != N:
                    raise ValueError(
                        "length of columns to draw from does not match")
        draw_idx = rng.integers(0, N, size=size)
        # draw and insert data
        for key, col in draw_from.items():
            rand[key] = col[draw_idx]
    return rand


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


class ThreadHelper(object):
    """
    ThreadHelper(threads, n_items)

    Helper class to apply a series of arguments to a function using
    multiprocessing.Pool.

    Parameters
    ----------
    threads : int
        Number of parallel threads for the pool.
    n_items : int
        Number items to be processed by the threads.
    """

    threads = multiprocessing.cpu_count()

    def __init__(
        self,
        n_items: int,
        threads: int | None = None
    ) -> None:
        self._n_items = n_items
        if threads is not None:
            self.threads = threads
        self.args = []

    def __len__(self) -> int:
        """
        Returns the number of expected function arguments.
        """
        return self._n_items

    def add_constant(self, value: Any) -> None:
        """
        add_constant(value)

        Append a constant argument that will be repeated for each thread.
        """
        self.args.append([value] * self._n_items)

    def add_iterable(
        self,
        iterable: Iterable,
        no_checks: bool = False
    ) -> None:
        """
        add_iterable(iterable, no_checks=False)

        Append a constant argument that will be repeated for each thread.
        """
        if not hasattr(iterable, "__iter__") and not no_checks:
            raise TypeError("object is not iterable: %s" % iterable)
        self.args.append(iterable)

    def map(self, callable: Callable) -> Iterable:
        """
        map(callable)

        Apply the accumulated arguments to a function in a pool of threads.
        The threads are blocking until all results are received.

        Parameters
        ----------
        callable : callable
            Callable object with the correct number of arguements.

        Returns:
        --------
        results: list
            Ordered list of return values of the called object per thread.
        """
        if self.threads > 1:
            with multiprocessing.Pool(self.threads) as pool:
                results = pool.starmap(callable, zip(*self.args))
        else:  # mimic the behaviour of pool.starmap() with map()
            results = list(map(callable, *self.args))
        return results
