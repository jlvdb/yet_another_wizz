"""This module implements helper classes to handle parallel computation with
:mod:`multiprocessing`.
"""

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

__all__ = ["SharedArray", "ParallelHelper"]


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SharedArray:
    """Container for shared array that can be easily shared with other
    processes.

    Stores multidimenional data boolean, integer or floating point numbers in a
    flat array in shared memory. The recommended constructors are :meth:`empty`
    and :meth:`from_numpy`.

    Args:
        array (:obj:`multiprocessing.RawArray`):
            A raw array of shared memory used to store array data.
        shape (:obj:`tuple[int]`):
            Shape of the array.
        dtype (:obj:`DTypeLike`):
            Numpy-style data type of the array.
    """

    array: multiprocessing.RawArray
    """Underlying shared memory array."""
    shape: tuple[int]
    """Shape of the array."""
    dtype: DTypeLike
    """Numpy-style data type of the array."""

    @staticmethod
    def _get_mp_type(dtype: DTypeLike | str) -> str:
        type_map = {
            "|b1": "b",  # no bool correspondance, but same number of bytes
            "i1": "h",
            "u1": "H",
            "i2": "i",
            "u2": "I",
            "i4": "l",
            "u4": "L",
            "i8": "q",
            "u8": "Q",
            "f4": "f",
            "f8": "d",
        }
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
    def empty(cls, shape: tuple[int], dtype: DTypeLike | str) -> SharedArray:
        """Create an new class instance with uninitialised memory.

        Args:
            shape (:obj:`tuple[int]`):
                Shape of the array.
            dtype (:obj:`DTypeLike`):
                Numpy-style data type of the array.

        Returns:
            :obj:`SharedArray`
        """
        mtype = cls._get_mp_type(dtype)
        size = int(np.prod(shape))
        new = cls(
            multiprocessing.RawArray(mtype, size), shape=shape, dtype=np.dtype(dtype)
        )
        return new

    @classmethod
    def from_numpy(cls, array: NDArray) -> SharedArray:
        """Create an new class instance from a numpy array.

        Args:
            array (:obj:`NDArray`):
                Input array, used to initialise the shared memory.

        Returns:
            :obj:`SharedArray`
        """
        new = cls.empty(array.shape, array.dtype)
        try:
            buffer = new.to_numpy()
            np.copyto(buffer, array)
            return new
        except Exception:
            del new
            raise

    def to_numpy(self, copy: bool = False) -> NDArray:
        """Reconstruct the original numpy array from shared memory.

        Args:
            copy (:obj:`bool`):
                Whether to copy the array values instead of wrapping the shared
                memory (zero-copy).

        Returns:
            :obj:`NDArray`
        """
        arr = np.frombuffer(self.array, dtype=self.dtype).reshape(self.shape)
        if copy:
            return np.copy(arr)
        else:
            return arr


POOL_SHARE = dict()
"""Global dictionary used to share :obj:`SharedArray` instances with other
processes."""


def _threadinit(
    shares_dict: dict[str, SharedArray],
    initializer: Callable | None = None,
    initargs: Iterable | None = None,
) -> None:
    """Initialiser function used to send shared arrays to processes."""
    POOL_SHARE.update(shares_dict)
    # call the usual initializer function provided by the user
    if initializer is not None:
        initializer(*initargs)


def _threadwrapper(arg_tuple, function):
    """Simple wrapper function that unpacks a tuple of arguments and applies to
    a function."""
    return function(*arg_tuple)


_T = TypeVar("_T")


class ParallelHelper(Generic[_T]):
    """Helper class to run a function in parallel with
    :obj:`multiprocessing.Pool`.

    The positional function arguments are registered in order of appearance.
    Arguments can be either iterable (all of the same length), which will be
    iterated for each parallel function call, or constant, which are identical
    at each function call.

    Additionally, :obj:`SharedArray` instances can be provided which will be
    distributed automatically.

    .. Warning:
        Calling the target function with keyword arguments is not supported.

    .. Tip::
        After completion, call the :meth:`free` method to free the memory of all
        shared array, or use this class as a context wrapper, which will
        automate this process on exit.
    """

    def __init__(
        self, function: Callable[..., _T], n_items: int, num_threads: int | None = None
    ) -> None:
        """Create a new instane.

        Args:
            function (Callable):
                Function which accepts an arbitray number of positional
                arguments.
            n_items (:obj:`int`):
                The total number of items/jobs to apply to the function.
            num_threads (:obj:`int`):
                The total number of threads to use for parallel processing.
                Capped at the maximum number of logical CPU cores.
        """
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
        """Free the memory of all shared arrays registered with
        :meth:`add_shared_array`."""
        for key in list(self.shares.keys()):
            sarray = self.shares.pop(key)
            del sarray

    def n_jobs(self) -> int:
        """The length of the iterable function arguments, i.e. the total number
        of jobs."""
        return self._n_items

    def n_args(self) -> int:
        """The total number of arguments passed at each function call."""
        return len(self.args)

    def n_threads(self):
        """The total number of threads to use for processing."""
        return self._num_threads

    def add_shared_array(self, key: str, array: NDArray | SharedArray) -> None:
        """Add a new shared array as function argument.

        .. Note::
            The array will be treated as constant, i.e. the same array is
            applied at each function call.

        Args:
            key (:obj:`str`):
                Identifier name at which the array is registered in the new
                processes namespace.
            array (:obj:`NDArray`, :obj:`SharedArray`):
                Array, if numpy array, converted automatically to shared array.
        """
        if isinstance(array, SharedArray):
            sarray = array
        else:
            sarray = SharedArray.from_numpy(array)
        self.shares[key] = sarray

    def add_constant(self, value: Any) -> None:
        """Add the next function argument which will be repeated at each
        function call."""
        self.args.append([value] * self._n_items)

    def add_iterable(self, iterable: Collection) -> None:
        """Add the next function argument which will be iterated at each
        function call.

        Must be an iterable with :meth:`n_jobs` items.
        """
        if len(iterable) != self._n_items:
            raise ValueError(f"length of iterable argument must be {self._n_items}")
        self.args.append(iterable)

    def _init_pool(
        self, initializer: Callable | None = None, initargs: Iterable | None = None
    ) -> multiprocessing.Pool:
        """Initialises the worker processes for the :obj:`multiprocessing.Pool`."""
        all_initargs = [self.shares]
        if initializer is not None:
            all_initargs.append(initializer)
            if isinstance(initargs, Iterable):
                all_initargs.append(initargs)
            elif initargs is None:
                all_initargs.append(None)
            else:
                raise TypeError("'initargs' must be an iterable")
        logger.debug("running %i jobs on %i threads", self.n_jobs(), self.n_threads())
        return multiprocessing.Pool(
            initializer=_threadinit, initargs=all_initargs, processes=self._num_threads
        )

    def result(
        self, initializer: Callable | None = None, initargs: Iterable | None = None
    ) -> list[_T]:
        """Apply the accumulated arguments to a function in a pool of processes.

        The call is blocking until all results are received.

        Args:
            initializer (Callable, optional):
                Custom initialiser function to call when creating the pool.
            initargs (Iterable, optional):
                Arguments applied to the initialiser function.

        Returns:
            :obj:`list`:
                List containing the return values from all function calls.
        """
        with self._init_pool(initializer, initargs) as pool:
            results = pool.starmap(self.function, zip(*self.args))
        return results

    def iter_result(
        self,
        initializer: Callable | None = None,
        initargs: Iterable | None = None,
        ordered: bool = True,
    ) -> Iterator[_T]:
        """Apply the accumulated arguments to a function in a pool of processes.

        Returns an iterator over the available thread results.

        Args:
            initializer (Callable, optional):
                Custom initialiser function to call when creating the pool.
            initargs (Iterable, optional):
                Arguments applied to the initialiser function.

        Yields:
            :obj:`list`:
                Iterator over the return values.
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
