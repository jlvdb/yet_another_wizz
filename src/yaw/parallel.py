from __future__ import annotations

import multiprocessing
import os
from collections.abc import Iterable, Iterator
from shutil import get_terminal_size
from typing import Callable, TypeVar

from mpi4py import MPI
from tqdm import tqdm

Targ = TypeVar("Targ")
Tresult = TypeVar("Tresult")
Titer = TypeVar("Titer")


def get_num_threads() -> int:
    system_threads = multiprocessing.cpu_count()
    num_threads = os.environ.get("YAW_NUM_THREADS", system_threads)
    return min(int(num_threads), system_threads)


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
MP_THREADS = get_num_threads()


def use_mpi() -> bool:
    return SIZE > 1


def on_root() -> bool:
    return COMM.Get_rank() == 0


class EndOfQueue:
    pass


class ParallelJob:
    def __init__(
        self, func: Callable[[Targ], Tresult], func_args: tuple, func_kwargs: dict
    ) -> None:
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs

    def __call__(self, arg: Targ) -> Tresult:
        return self.func(arg, *self.func_args, **self.func_kwargs)


def mpi_root_task(iterable: Iterable) -> Iterator:
    # first pass of assigning tasks to workers dynamically
    active_workers = 0
    for rank in range(1, SIZE):
        try:
            COMM.send(next(iterable), dest=rank, tag=1)
            active_workers += 1
        except StopIteration:
            # if more workers than tasks, shut down superflous workers
            COMM.send(EndOfQueue, dest=rank, tag=1)

    # yield results from workers and send new tasks until all have been processed
    while active_workers > 0:
        rank, result = COMM.recv(source=MPI.ANY_SOURCE, tag=2)
        yield result

        try:
            COMM.send(next(iterable), dest=rank, tag=1)
        except StopIteration:
            COMM.send(EndOfQueue, dest=rank, tag=1)
            active_workers -= 1


def mpi_worker_task(func: ParallelJob) -> None:
    rank = COMM.Get_rank()
    while (arg := COMM.recv(source=0, tag=1)) is not EndOfQueue:
        result = func(arg)
        COMM.send((rank, result), dest=0, tag=2)


def mpi_iter_unordered(
    func: Callable[[Targ], Tresult],
    iterable: Iterable[Targ],
    *,
    func_args: tuple,
    func_kwargs: dict,
) -> Iterator[Tresult]:
    if on_root():
        iterable = iter(iterable)
        yield from mpi_root_task(iterable)

    else:
        wrapped_func = ParallelJob(func, func_args, func_kwargs)
        mpi_worker_task(wrapped_func)


def multiprocessing_iter_unordered(
    func: Callable[[Targ], Tresult],
    iterable: Iterable[Targ],
    *,
    func_args: tuple,
    func_kwargs: dict,
    num_threads: int | None = None,
) -> Iterator[Tresult]:
    wrapped_func = ParallelJob(func, func_args, func_kwargs)
    with multiprocessing.Pool(num_threads) as pool:
        yield from pool.imap_unordered(wrapped_func, iterable)


class ParallelHelper:
    comm = COMM
    size = SIZE
    num_threads = MP_THREADS

    @classmethod
    def set_multiprocessing_threads(cls, num_threads: int) -> None:
        cls.num_threads = num_threads

    @classmethod
    def use_mpi(cls) -> bool:
        return use_mpi()

    @classmethod
    def get_rank(cls) -> int:
        return cls.comm.Get_rank()

    @classmethod
    def on_root(cls) -> bool:
        return on_root()

    @classmethod
    def on_worker(cls) -> bool:
        return not cls.on_root()

    @classmethod
    def print(cls, *args, **kwargs) -> None:
        if cls.on_root():
            print(*args, **kwargs)

    @classmethod
    def iter_unordered(
        cls,
        func: Callable[[Targ], Tresult],
        iterable: Iterable[Targ],
        *,
        func_args: tuple | None = None,
        func_kwargs: dict | None = None,
        progress: bool = False,
        total: int | None = None,
    ) -> Iterator[Tresult]:
        iter_kwargs = dict(
            func_args=(func_args or tuple()),
            func_kwargs=(func_kwargs or dict()),
        )
        if cls.use_mpi():
            parallel_method = mpi_iter_unordered
        else:
            parallel_method = multiprocessing_iter_unordered
            iter_kwargs["num_threads"] = cls.num_threads

        result_iter = parallel_method(func, iterable, **iter_kwargs)
        result_iter_progress_optional = tqdm(
            result_iter,
            total=total,
            ncols=min(80, get_terminal_size()[0]),
            disable=(not progress or cls.on_worker()),
        )
        yield from result_iter_progress_optional
