"""
Core implementation of a parallel computation model that works with both
MPI (mpi4py) or python's multiprocessing by having a shared API.

The code dynamically figures out if running in an MPI execution environment,
otherwise falls back to using multiprocessing (see use_mpi). Implements a
parallel iterator for MPI that functions similar to multiprocessing's
Pool.imap_unordered(). Also implements a mock-up of an MPI communicator that is
used as a stand-in when running with multiprocessing.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import subprocess
import sys
from abc import ABC
from functools import partial
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, Callable, Literal, TypeVar

    from mpi4py.MPI import Comm
    from numpy.typing import NDArray

    T = TypeVar("T")
    TypeArgument = TypeVar("TypeArgument")
    TypeResult = TypeVar("TypeResult")
    TypeBroadcastable = TypeVar("TypeBroadcastable", bound="Broadcastable")

__all__ = [
    "Broadcastable",
    "COMM",
    "get_size",
    "iter_unordered",
    "on_root",
    "on_worker",
    "ranks_on_same_node",
    "use_mpi",
    "world_to_comm_rank",
]

logger = logging.getLogger(__name__)


def _get_physical_cores() -> int:
    """Attempt to get the number of physical as opposed to logical CPU cores of
    the system."""
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
    """Get the number of processes to use from the ``YAW_NUM_THREADS``
    environment variable, otherwise use the number of physical cores."""
    system_threads = _get_physical_cores()

    try:
        num_threads = int(os.environ["YAW_NUM_THREADS"])
        return min(num_threads, system_threads)

    except KeyError:
        return system_threads


try:
    from mpi4py import MPI

    def use_mpi() -> bool:
        """Whether the current code is run in an MPI environment."""
        return MPI.COMM_WORLD.Get_size() > 1

except ImportError:

    def use_mpi() -> Literal[False]:
        """Whether the current code is run in an MPI environment."""
        return False


class MockComm:
    """Implements the most basic functionality of an MPI communicator if MPI is
    not installed."""

    def Barrier(self) -> None:
        """Mock-implementation of a synchronisation barrier, does nothing
        here."""
        pass

    def Bcast(self, value: T, *args, **kwargs) -> T:
        """Mock-implementation of broadcasting array buffers, returns value
        here."""
        return value

    def bcast(self, value: T, *args, **kwargs) -> T:
        """Mock-implementation of broadcasting python objects, returns value
        here."""
        return value

    def Get_size(self) -> int:
        """Mock-implementation of getting number of parallel workers, here the
        number of processes."""
        return _num_processes()

    def Get_rank(self) -> int:
        """Mock-implementation of getting the worker rank, always 0 here."""
        return 0


if use_mpi():
    COMM = MPI.COMM_WORLD
    """The default communicator that `yet_another_wizz` uses (world comm)."""
else:
    COMM = MockComm()
    """The default communicator that `yet_another_wizz` uses (mock-up comm)."""


def get_size(max_workers: int | None = None, comm: Comm = COMM) -> int:
    """Get the smaller value of ``max_workers`` or the size of the
    communicator."""
    if use_mpi():
        size = comm.Get_size()
    else:
        size = _num_processes()
    max_workers = max_workers or size
    return min(max_workers, size)


def on_root(comm: Comm = COMM) -> bool:
    """Whether currently on the root worker with rank 0."""
    return comm.Get_rank() == 0


def on_worker(comm: Comm = COMM) -> bool:
    """Whether currently on a non-root worker."""
    return comm.Get_rank() != 0


def ranks_on_same_node(
    rank: int = 0, max_workers: int | None = None, comm: Comm = COMM
) -> set[int]:
    """Get the set of MPI ranks that are associated with the same CPU as the
    root rank."""
    proc_name = MPI.Get_processor_name()
    proc_names = comm.gather(proc_name, root=rank)

    on_same_node = set()
    if comm.Get_rank() == rank:
        on_same_node = [i for i, name in enumerate(proc_names) if name == proc_name]
        if max_workers is not None:
            on_same_node = on_same_node[:max_workers]
        on_same_node = set(on_same_node)

    return comm.bcast(on_same_node, root=rank)


def world_to_comm_rank(comm: Comm, world_rank: int) -> int:
    """Get the rank of the current rank in the world rank."""
    comm_rank = None
    if MPI.COMM_WORLD.Get_rank() == world_rank:
        comm_rank = comm.Get_rank()
    return comm.bcast(comm_rank, root=world_rank)


class EndOfQueue:
    pass


class ParallelJob:
    """
    Wrapper for a function that binds arguments and keyword arguments, similar
    to ``functools.partial``. If ``unpack=True``, the positional arguments are
    unpacked when calling the function, otherwise they are passed as a tuple.
    """

    __slots__ = ("func", "func_args", "func_kwargs", "unpack")

    def __init__(
        self,
        func: Callable[..., TypeResult],
        func_args: tuple,
        func_kwargs: dict,
        *,
        unpack: bool = False,
    ) -> None:
        self.func = func
        self.func_args = func_args
        self.func_kwargs = func_kwargs
        self.unpack = unpack

    def __call__(self, arg: Any) -> TypeResult:
        if self.unpack:
            func_args = (*arg, *self.func_args)
        else:
            func_args = (arg, *self.func_args)
        return self.func(*func_args, **self.func_kwargs)


def _mpi_root_task(
    iterable: Iterable, ranks: Iterable[int], comm: Comm = COMM
) -> Iterator:
    """On the root rank, send the job arguments to the remaining ranks and
    collect the results in an iterator."""
    # first pass of assigning tasks to workers dynamically
    active_workers = 0
    for rank in range(1, get_size()):
        try:
            assert rank in ranks
            comm.send(next(iterable), dest=rank, tag=1)
            active_workers += 1
        except (AssertionError, StopIteration):
            # shut down any unused workers
            comm.send(EndOfQueue, dest=rank, tag=1)

    # yield results from workers and send new tasks until all have been processed
    while active_workers > 0:
        rank, result = comm.recv(source=MPI.ANY_SOURCE, tag=2)
        yield result

        try:
            comm.send(next(iterable), dest=rank, tag=1)
        except StopIteration:
            comm.send(EndOfQueue, dest=rank, tag=1)
            active_workers -= 1


def _mpi_worker_task(func: ParallelJob, comm: Comm = COMM) -> None:
    """On the worker rank, receive function arguments and return the function
    call results to the root rank."""
    rank = comm.Get_rank()
    while (arg := comm.recv(source=0, tag=1)) is not EndOfQueue:
        result = func(arg)
        comm.send((rank, result), dest=0, tag=2)


def _mpi_iter_unordered(
    func: Callable[[TypeArgument], TypeResult],
    iterable: Iterable[TypeArgument],
    *,
    func_args: tuple,
    func_kwargs: dict,
    unpack: bool = False,
    ranks: Iterable[int],
    comm: Comm = COMM,
) -> Iterator[TypeResult]:
    """
    Asynchronous iterator that maps arguments to a worker function using MPI
    parallelism.

    Takes a job function, an iterable of job arguments and optionally a list
    of positional and keyword arguments to bind to the job function.
    Additionally, specify if the function expects the the positional arguments
    as a single tuple or unpacked.
    """
    if on_root():
        iterable = iter(iterable)
        yield from _mpi_root_task(iterable, ranks, comm=comm)

    else:
        wrapped_func = ParallelJob(func, func_args, func_kwargs, unpack=unpack)
        _mpi_worker_task(wrapped_func, comm=comm)

    comm.Barrier()


def _multiprocessing_iter_unordered(
    func: Callable[[TypeArgument], TypeResult],
    iterable: Iterable[TypeArgument],
    *,
    func_args: tuple,
    func_kwargs: dict,
    unpack: bool = False,
    num_processes: int | None = None,
) -> Iterator[TypeResult]:
    """
    Asynchronous iterator that maps arguments to a worker function using
    multiprocessing parallelism.

    Takes a job function, an iterable of job arguments and optionally a list
    of positional and keyword arguments to bind to the job function.
    Additionally, specify if the function expects the the positional arguments
    as a single tuple or unpacked.
    """
    wrapped_func = ParallelJob(func, func_args, func_kwargs, unpack=unpack)

    if num_processes == 1:
        yield from map(wrapped_func, iterable)

    else:
        with multiprocessing.Pool(num_processes) as pool:
            yield from pool.imap_unordered(wrapped_func, iterable)


def iter_unordered(
    func: Callable[[TypeArgument], TypeResult],
    iterable: Iterable[TypeArgument],
    *,
    func_args: tuple | None = None,
    func_kwargs: dict | None = None,
    unpack: bool = False,
    max_workers: int | None = None,
    rank0_node_only: bool = False,
    comm: Comm = COMM,
) -> Iterator[TypeResult]:
    """
    Asynchronous iterator that maps arguments to a worker function, choosing
    the parallelism mechanism (MPI/multiprocessing) automatically.

    Takes a job function, an iterable of job arguments and optionally a list
    of positional and keyword arguments to bind to the job function. Specify if
    the function expects the the positional arguments as a single tuple or
    unpacked. Additionally limit the number of workers to use (also applies to
    MPI) or run only on the same node as the root worker (MPI only).
    """
    max_workers = get_size(max_workers)
    iter_kwargs = dict(
        func_args=(func_args or tuple()),
        func_kwargs=(func_kwargs or dict()),
        unpack=unpack,
    )

    if use_mpi():
        if rank0_node_only:
            ranks = ranks_on_same_node(rank=0, max_workers=max_workers, comm=comm)
        else:
            ranks = set(range(max_workers))

        num_workers = len(ranks)
        iter_kwargs["ranks"] = ranks
        iter_kwargs["comm"] = comm
        parallel_method = _mpi_iter_unordered

    else:
        num_workers = max_workers
        iter_kwargs["num_processes"] = max_workers
        parallel_method = _multiprocessing_iter_unordered

    if on_root():
        if max_workers > 1:
            logger.debug("running parallel jobs on %d workers", num_workers)
        else:
            logger.debug("running jobs sequentially")

    yield from parallel_method(func, iterable, **iter_kwargs)


class Broadcastable(ABC):
    """
    Implements a protocol that allows efficient MPI broadcasting of numpy
    arrays.

    Subclasses must implement ``__slots__``, which specify the attribte that
    must be shared with other ranks. This allows to broadcast the attributes
    individually and recursively and use MPI broadcasting of numpy arrays where
    possible and pickling otherwise.

    This solves an issue when trying to broadcast ``CorrFunc`` instances with
    many patches, which fails when using pickling.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if "__slots__" not in cls.__dict__:
            raise TypeError(
                f"{cls.__name__}: subclass of Broadcastable must implement __slots__"
            )


def new_uninitialised(cls: type[TypeBroadcastable]) -> TypeBroadcastable:
    """Create an empty instance of the class with all attributes initialised to
    ``None``."""
    inst = cls.__new__(cls)
    for attr in cls.__slots__:
        setattr(inst, attr, None)
    return inst


def bcast_array(array: NDArray, comm: Comm = COMM) -> NDArray:
    """Broadcast a numpy array to all non-root ranks. Input array may have any
    value on non-root ranks."""
    array_info = ()
    if on_root():
        array = np.ascontiguousarray(array)
        array_info = (array.shape, array.dtype)
    array_info = comm.bcast(array_info, root=0)

    if on_worker():
        shape, dtype = array_info
        array = np.empty(shape, dtype=dtype)
    comm.Bcast(array, root=0)

    return array


def get_bcast_method(inst: T, comm: Comm = COMM) -> Callable[[T], T]:
    """Determine if the object must be broadcastes by recursion, using MPI
    broadcasting mechanisms, or using pickling."""
    if isinstance(inst, Broadcastable):
        bcast_method = partial(bcast_instance, comm=comm)
    elif isinstance(inst, np.ndarray):
        bcast_method = partial(bcast_array, comm=comm)
    else:
        bcast_method = comm.bcast

    return comm.bcast(bcast_method, root=0)


def bcast_instance(inst: TypeBroadcastable, *, comm: Comm = COMM) -> TypeBroadcastable:
    """Broadcast an instance of a subclass of ``Broadcastable`` to all non-root
    ranks. Instance may have any value on non-root ranks."""
    if not use_mpi():
        return inst

    cls = comm.bcast(type(inst), root=0)
    if on_worker():
        inst = new_uninitialised(cls)

    for name in inst.__slots__:
        value = getattr(inst, name)
        bcast = get_bcast_method(value, comm=comm)
        setattr(inst, name, bcast(value))

    return inst
