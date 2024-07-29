from __future__ import annotations

import multiprocessing
from collections.abc import Iterable, Iterator
from shutil import get_terminal_size
from typing import Callable, TypeVar

from mpi4py import MPI
from tqdm import tqdm

Targ = TypeVar("Targ")
Tresult = TypeVar("Tresult")

COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()


def use_mpi() -> bool:
    return SIZE > 1


class EndOfQueue:
    pass


class ParallelJob:
    def __init__(
        self, func: Callable[[Targ], Tresult], args: tuple, kwargs: dict
    ) -> None:
        self.function = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, arg: Targ) -> tuple[int, Tresult]:
        result = self.function(arg, *self.args, **self.kwargs)

        if use_mpi():
            rank = COMM.Get_rank()
        else:
            name = multiprocessing.current_process().name
            _, str_rank = name.split("-")
            rank = int(str_rank)

        return (rank, result)


def _init_mpi_jobs(job_items: Iterable) -> int:
    active_ranks = 0
    for rank in range(1, SIZE):
        try:
            COMM.send(next(job_items), dest=rank, tag=1)
            active_ranks += 1
        except StopIteration:  # stop all superflous workers
            COMM.send(EndOfQueue, dest=rank, tag=1)
    return active_ranks


def _finalise_mpi_jobs(job_items: Iterable, active_ranks: int) -> Iterator:
    while active_ranks > 0:
        worker, result = COMM.recv(source=MPI.ANY_SOURCE, tag=2)
        yield result

        try:
            COMM.send(next(job_items), dest=worker, tag=1)
        except StopIteration:
            COMM.send(EndOfQueue, dest=worker, tag=1)
            active_ranks -= 1


def _run_mpi_job(job_func: ParallelJob) -> None:
    while (args := COMM.recv(source=0, tag=1)) is not EndOfQueue:
        rank_result = job_func(args)
        COMM.send(rank_result, dest=0, tag=2)


class ParallelHelper:
    comm = COMM
    size = SIZE
    num_threads = multiprocessing.cpu_count()

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
        return cls.get_rank() == 0

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
        function: Callable[[Targ], Tresult],
        job_items: Iterable[Targ],
        *,
        job_args: tuple | None = None,
        job_kwargs: dict | None = None,
        progress: bool = True,
        n_jobs: int | None = None,
    ) -> Iterator[Tresult]:
        if job_args is None:
            job_args = tuple()
        if job_kwargs is None:
            job_kwargs = dict()

        parallel_method = cls._iter_mpi if cls.use_mpi() else cls._iter_multiprocessing
        result_iter = parallel_method(
            function, job_items, job_args=job_args, job_kwargs=job_kwargs
        )

        if progress and cls.on_root():
            if n_jobs is None:
                n_jobs = len(job_items)
            ncols = min(80, get_terminal_size()[0])
            result_iter: Iterator[Tresult] = tqdm(
                result_iter, total=n_jobs, ncols=ncols
            )

        for result in iter(result_iter):
            yield result

    @classmethod
    def _iter_multiprocessing(
        cls,
        function: Callable[[Targ], Tresult],
        job_items: Iterable[Targ],
        *,
        job_args: tuple,
        job_kwargs: dict,
    ) -> Iterator[Tresult]:
        job_func = ParallelJob(function, job_args, job_kwargs)
        with multiprocessing.Pool(cls.num_threads) as pool:
            for _, result in pool.imap_unordered(job_func, job_items):
                yield result

    @classmethod
    def _iter_mpi(
        cls,
        function: Callable[[Targ], Tresult],
        job_items: Iterable[Targ],
        *,
        job_args: tuple,
        job_kwargs: dict,
    ) -> Iterator[Tresult]:
        if cls.on_worker():
            job_func = ParallelJob(function, job_args, job_kwargs)
            _run_mpi_job(job_func)

        else:
            job_iter = iter(job_items)
            active_ranks = _init_mpi_jobs(job_iter)
            for result in _finalise_mpi_jobs(job_iter, active_ranks):
                yield result
