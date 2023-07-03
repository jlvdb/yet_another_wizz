from __future__ import annotations

import functools
import logging
import multiprocessing
from collections.abc import Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from numpy.typing import DTypeLike, NDArray


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedArray:

    array: multiprocessing.RawArray
    shape: tuple[int]
    dtype: DTypeLike

    @staticmethod
    def _get_mp_type(dtype: DTypeLike | str) -> str:
        type_map = {
            "|b1": "b",  # no bool correspondance, but same number of bytes
            "i1": "h", "u1": "H",
            "i2": "i", "u2": "I",
            "i4": "l", "u4": "L",
            "i8": "q", "u8": "Q",
            "f4": "f", "f8": "d"}
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
        try:
            return type_map[dtype.str.strip("<>")]
        except KeyError as e:
            raise NotImplementedError(
                f"cannot convert numpy dtype '{dtype.str}' to RawArray"
            ) from e

    def __len__(self) -> int:
        return self.shape[0]

    def __del__(self) -> None:
        object.__delattr__(self, "array")

    @classmethod
    def empty(
        cls,
        shape: tuple[int],
        dtype: DTypeLike | str
    ) -> SharedArray:
        mtype = cls._get_mp_type(dtype)
        size = int(np.prod(shape))
        new = cls(
            multiprocessing.RawArray(mtype, size),
            shape=shape, dtype=np.dtype(dtype))
        return new

    @classmethod
    def from_numpy(cls, array: NDArray) -> SharedArray:
        new = cls.empty(array.shape, array.dtype)
        try:
            buffer = new.to_numpy()
            np.copyto(buffer, array)
        except Exception:
            del new
            raise
        return new

    def to_numpy(self, copy: bool = False) -> NDArray:
        arr = np.frombuffer(self.array, dtype=self.dtype).reshape(self.shape)
        if copy:
            return np.copy(arr)
        else:
            return arr


POOL_SHARE = dict()
"""Global variable ..."""


def _threadinit(
    shares_dict: dict[str, SharedArray],
    initializer: Callable | None = None,
    initargs: Iterable | None = None
) -> None:
    POOL_SHARE.update(shares_dict)
    # call the usual initializer function provided by the user
    if initializer is not None:
        initializer(*initargs)


def _threadwrapper(arg_tuple, function):
    return function(*arg_tuple)


_T = TypeVar("_T")


class ParallelHelper(Generic[_T]):
    """
    Helper class to apply a series of arguments to a function using
    multiprocessing.Pool.
    """

    def __init__(
        self,
        function: Callable[..., _T],
        n_items: int,
        num_threads: int | None = None
    ) -> None:
        self.function = function
        self._n_items = n_items
        if num_threads is None:
            num_threads = multiprocessing.cpu_count()
        self._num_threads = num_threads if n_items > num_threads else n_items
        self.shares = {}
        self.args = []

    def __enter__(self) -> ParallelHelper:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.free()

    def free(self) -> None:
        for key in list(self.shares.keys()):
            sarray = self.shares.pop(key)
            del sarray

    def n_jobs(self) -> int:
        """
        Returns the number of expected function arguments.
        """
        return self._n_items

    def n_args(self) -> int:
        return len(self.args)

    def n_threads(self):
        return self._num_threads

    def add_shared_array(self, key: str, array: NDArray | SharedArray) -> None:
        if isinstance(array, SharedArray):
            sarray = array
        else:
            sarray = SharedArray.from_numpy(array)
        self.shares[key] = sarray

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

    def _init_pool(
        self,
        initializer: Callable | None = None,
        initargs: Iterable | None = None
    ) -> multiprocessing.Pool:
        all_initargs = [self.shares]
        if initializer is not None:
            all_initargs.append(initializer)
            if isinstance(initargs, Iterable):
                all_initargs.append(initargs)
            elif initargs is None:
                all_initargs.append(None)
            else:
                raise TypeError("'initargs' must be an iterable")
        logger.debug(
            f"running {self.n_jobs()} jobs on {self.n_threads()} threads")
        return multiprocessing.Pool(
            initializer=_threadinit,
            initargs=all_initargs,
            processes=self._num_threads)

    def result(
        self,
        initializer: Callable | None = None,
        initargs: Iterable | None = None
    ) -> list[_T]:
        """
        Apply the accumulated arguments to a function in a pool of threads.
        The threads are blocking until all results are received.
        """
        with self._init_pool(initializer, initargs) as pool:
            results = pool.starmap(self.function, zip(*self.args))
        return results

    def iter_result(
        self,
        initializer: Callable | None = None,
        initargs: Iterable | None = None,
        ordered: bool = True
    ) -> Iterator[_T]:
        """
        Apply the accumulated arguments to a function in a pool of threads.
        The results are processed unordered and yielded as an iterator.
        """
        function = functools.partial(_threadwrapper, function=self.function)
        with self._init_pool(initializer, initargs) as pool:
            imap_args = (function, zip(*self.args))
            if ordered:
                for result in pool.imap(*imap_args):
                    yield result
            else:
                for result in pool.imap_unordered(*imap_args):
                    yield result
