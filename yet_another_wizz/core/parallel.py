from __future__ import annotations

import functools
import logging
import multiprocessing
from collections.abc import Callable, Collection, Iterator
from typing import Any


logger = logging.getLogger(__name__.replace(".core.", "."))


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

    def result(self) -> list:
        """
        Apply the accumulated arguments to a function in a pool of threads.
        The threads are blocking until all results are received.
        """
        logger.debug(
            f"running {self.n_jobs()} jobs on {self.n_threads()} threads")
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
        logger.debug(
            f"running {self.n_jobs()} jobs on {self.n_threads()} threads")
        function = functools.partial(_threadwrapper, function=self.function)
        if self._num_threads > 1:
            with multiprocessing.Pool(self._num_threads) as pool:
                for result in pool.imap_unordered(function, zip(*self.args)):
                    yield result
        else:  # mimic the behaviour of pool.imap_unordered()
            for result in map(self.function, *self.args):
                yield result
