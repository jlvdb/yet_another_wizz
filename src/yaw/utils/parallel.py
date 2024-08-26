from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
from collections.abc import Iterable, Iterator
from typing import Callable, TypeVar

from mpi4py import MPI

__all__ = [
    "ParallelHelper",
    "get_num_threads",
    "get_physical_cores",
]

Targ = TypeVar("Targ")
Tresult = TypeVar("Tresult")
Titer = TypeVar("Titer")

logger = logging.getLogger(__name__)


def get_physical_cores() -> int:
    try:
        if os.name == "posix":
            output = subprocess.check_output("lscpu")
            for line in output.decode("utf-8").splitlines():
                if "Core(s) per socket:" in line:
                    return int(line.split(":")[1].strip())

        elif sys.platform == "darwin":  # macOS
            output = subprocess.check_output("sysctl -n hw.physicalcpu", shell=True)
            return int(output.decode("utf-8").strip())

        elif os.name == "nt":
            output = subprocess.check_output("WMIC CPU Get NumberOfCores", shell=True)
            return int(output.strip().split(b"\n")[1])

    except Exception:
        return multiprocessing.cpu_count()


def get_num_threads() -> int:
    system_threads = get_physical_cores()

    try:
        num_threads = int(os.environ["YAW_NUM_THREADS"])
        return min(num_threads, system_threads)

    except KeyError:
        return system_threads


COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
MP_THREADS = get_num_threads()


def use_mpi() -> bool:
    return SIZE > 1


def on_root() -> bool:
    return COMM.Get_rank() == 0


def mpi_ranks_on_same_node(rank: int = 0, max_workers: int | None = None) -> list[int]:
    proc_name = MPI.Get_processor_name()
    proc_names = COMM.gather(proc_name, root=rank)

    on_same_node = []
    if COMM.Get_rank() == rank:
        on_same_node = [i for i, name in enumerate(proc_names) if name == proc_name]
        if max_workers is not None:
            on_same_node = on_same_node[:max_workers]

    return COMM.bcast(on_same_node, root=rank)


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


def mpi_root_task(iterable: Iterable, ranks: Iterable[int]) -> Iterator:
    # first pass of assigning tasks to workers dynamically
    active_workers = 0
    for rank in range(1, SIZE):
        try:
            assert rank in ranks
            COMM.send(next(iterable), dest=rank, tag=1)
            active_workers += 1
        except (AssertionError, StopIteration):
            # shut down any unused workers
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
    ranks: Iterable[int],
) -> Iterator[Tresult]:
    if on_root():
        iterable = iter(iterable)
        yield from mpi_root_task(iterable, ranks)

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
        num_threads = min(int(num_threads), multiprocessing.cpu_count())
        if num_threads < 1:
            num_threads = get_physical_cores()

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
        num_threads: int | None = None,
        rank0_node_only: bool = False,
    ) -> Iterator[Tresult]:
        iter_kwargs = dict(
            func_args=(func_args or tuple()),
            func_kwargs=(func_kwargs or dict()),
        )

        if cls.use_mpi():
            if rank0_node_only:
                ranks = set(mpi_ranks_on_same_node(rank=0, max_workers=num_threads))
            else:
                ranks = set(range(min(SIZE, num_threads)))

            num_workers = len(ranks)
            iter_kwargs["ranks"] = ranks
            parallel_method = mpi_iter_unordered

        else:
            num_workers = num_threads or cls.num_threads
            iter_kwargs["num_threads"] = num_workers
            parallel_method = multiprocessing_iter_unordered

        logger.debug(f"running parallel jobs on {num_workers} workers")
        yield from parallel_method(func, iterable, **iter_kwargs)
        cls.comm.Barrier()
