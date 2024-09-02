from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Callable, Literal, TypeVar

    T = TypeVar("T")
    Targ = TypeVar("Targ")
    Tresult = TypeVar("Tresult")
    Titer = TypeVar("Titer")

__all__ = [
    "COMM",
    "get_size",
    "iter_unordered",
    "on_root",
    "on_worker",
    "ranks_on_same_node",
    "use_mpi",
]

logger = logging.getLogger(__name__)


def _get_physical_cores() -> int:
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


def _num_processes() -> int:
    system_threads = _get_physical_cores()

    try:
        num_threads = int(os.environ["YAW_NUM_THREADS"])
        return min(num_threads, system_threads)

    except KeyError:
        return system_threads


try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD

    def use_mpi() -> bool:
        return COMM.Get_size() > 1

except ImportError:

    def pass_value(value: T, *args, **kwargs) -> T:
        return value

    def do_nothing(*args, **kwargs) -> None:
        pass

    class MockComm:
        Barrier = do_nothing
        Bcast = pass_value
        bcast = pass_value
        Get_size = _num_processes

    COMM = MockComm()

    def use_mpi() -> Literal[False]:
        return False


def get_size(max_workers: int | None = None) -> int:
    if use_mpi():
        size = COMM.Get_size()
    else:
        size = _num_processes()
    max_workers = max_workers or size
    return min(max_workers, size)


def on_root() -> bool:
    return COMM.Get_rank() == 0


def on_worker() -> bool:
    return COMM.Get_rank() != 0


def ranks_on_same_node(rank: int = 0, max_workers: int | None = None) -> list[int]:
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
    __slots__ = ("func", "func_args", "func_kwargs", "unpack")

    def __init__(
        self,
        func: Callable[..., Tresult],
        func_args: tuple,
        func_kwargs: dict,
        *,
        unpack: bool = False,
    ) -> None:
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.unpack = unpack

    def __call__(self, arg: Any) -> Tresult:
        if self.unpack:
            func_args = (*arg, *self.func_args)
        else:
            func_args = (arg, *self.func_args)
        return self.func(*func_args, **self.func_kwargs)


def _mpi_root_task(iterable: Iterable, ranks: Iterable[int]) -> Iterator:
    # first pass of assigning tasks to workers dynamically
    active_workers = 0
    for rank in range(1, get_size()):
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


def _mpi_worker_task(func: ParallelJob) -> None:
    rank = COMM.Get_rank()
    while (arg := COMM.recv(source=0, tag=1)) is not EndOfQueue:
        result = func(arg)
        COMM.send((rank, result), dest=0, tag=2)


def _mpi_iter_unordered(
    func: Callable[[Targ], Tresult],
    iterable: Iterable[Targ],
    *,
    func_args: tuple,
    func_kwargs: dict,
    unpack: bool = False,
    ranks: Iterable[int],
) -> Iterator[Tresult]:
    if on_root():
        iterable = iter(iterable)
        yield from _mpi_root_task(iterable, ranks)

    else:
        wrapped_func = ParallelJob(func, func_args, func_kwargs, unpack=unpack)
        _mpi_worker_task(wrapped_func)

    COMM.Barrier()


def _multiprocessing_iter_unordered(
    func: Callable[[Targ], Tresult],
    iterable: Iterable[Targ],
    *,
    func_args: tuple,
    func_kwargs: dict,
    unpack: bool = False,
    num_processes: int | None = None,
) -> Iterator[Tresult]:
    wrapped_func = ParallelJob(func, func_args, func_kwargs, unpack=unpack)
    with multiprocessing.Pool(num_processes) as pool:
        yield from pool.imap_unordered(wrapped_func, iterable)


def iter_unordered(
    func: Callable[[Targ], Tresult],
    iterable: Iterable[Targ],
    *,
    func_args: tuple | None = None,
    func_kwargs: dict | None = None,
    unpack: bool = False,
    max_workers: int | None = None,
    rank0_node_only: bool = False,
) -> Iterator[Tresult]:
    max_workers = get_size(max_workers)
    iter_kwargs = dict(
        func_args=(func_args or tuple()),
        func_kwargs=(func_kwargs or dict()),
        unpack=unpack,
    )

    if use_mpi():
        if rank0_node_only:
            ranks = ranks_on_same_node(rank=0, max_workers=max_workers)
        else:
            ranks = range(max_workers)

        num_workers = len(ranks)
        iter_kwargs["ranks"] = set(ranks)
        parallel_method = _mpi_iter_unordered

    else:
        num_workers = max_workers
        iter_kwargs["num_processes"] = max_workers
        parallel_method = _multiprocessing_iter_unordered

    if on_root():
        logger.debug(f"running parallel jobs on {num_workers} workers")

    yield from parallel_method(func, iterable, **iter_kwargs)
