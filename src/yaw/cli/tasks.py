"""
Network of task connections (#=: required, |-: optional):

cache_ref === auto_ref --+---------+
     #                   |         |
     #==== cross === estimate --- plot
     #                   |        |  |
cache_unk === auto_unk --+--------+  |
     #                               |
     #=== hist ----------------------+
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Container, Sized
from typing import TYPE_CHECKING

import yaw
from yaw.cli.utils import print_message
from yaw.config.base import ConfigError
from yaw.utils import transform_matches

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, TypeVar

    from yaw.cli.config import CatPairConfig, ProjectConfig
    from yaw.cli.directory import CacheDirectory, ProjectDirectory
    from yaw.cli.handles import CacheHandle, CorrFuncHandle
    from yaw.config import Configuration

    TypeTask = TypeVar("TypeTask", bound="Task")


class Task(ABC):
    name: str
    _tasks: dict[str, type[Task]] = {}
    _inputs: set[type[Task]]
    _optionals: set[type[Task]]

    def __init_subclass__(cls):
        # implicit naming conventions are nasty, but here we go
        if not cls.__name__.endswith("Task"):
            raise NameError("name of sublasses of 'Task' must contain 'Task'")
        name = cls.__name__.replace("Task", "")

        # transform name: MyNew -> my_new
        cls.name = transform_matches(
            name, regex=r"[A-Z]", transform=lambda match: "_" + match.lower()
        ).lstrip("_")

        cls._tasks[cls.name] = cls
        return super().__init_subclass__()

    def __init__(self) -> None:
        self.inputs: set[Task] = set()
        self.optionals: set[Task] = set()

    @classmethod
    def get(cls: type[TypeTask], name: str) -> type[TypeTask]:
        try:
            return cls._tasks[name]
        except KeyError as err:
            raise ValueError(f"no tasked with name '{name}'") from err

    def connect_input(self, task: Task) -> bool:
        if type(task) in self._inputs:
            self.inputs.add(task)
            return True

        if type(task) in self._optionals:
            self.optionals.add(task)
            return True

        return False

    def check_inputs(self) -> None:
        expect = set(t.name for t in self._inputs)
        have = set(t.name for t in self.inputs)
        for name in expect - have:
            raise ConfigError(f"missing input '{name}' for task '{self.name}'")

    def completed(self) -> bool:
        return all(handle.exists() for handle in self.outputs)

    @abstractmethod
    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        pass


def create_catalog(
    global_cache: CacheDirectory,
    cache_handle: CacheHandle,
    inputs: CatPairConfig,
    num_patches: int | None,
) -> None:
    columns = inputs.columns
    paths = {
        cache_handle.rand.path: inputs.path_rand,
        cache_handle.data.path: inputs.path_data,
    }

    for cache_path, input_path in paths.items():
        if input_path is None:
            continue

        cat = yaw.Catalog.from_file(
            cache_directory=cache_path,
            path=input_path,
            ra_name=columns.ra,
            dec_name=columns.dec,
            weight_name=columns.weight,
            redshift_name=columns.redshift,
            patch_centers=global_cache.get_patch_centers(),
            patch_name=columns.patches,
            patch_num=num_patches,
            # overwrite=...,
            # progress=...,
            # max_workers=...,
        )
        try:
            global_cache.set_patch_centers(cat.get_centers())
        except RuntimeError:
            pass


class CacheRefTask(Task):
    _inputs = set()
    _optionals = set()

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        create_catalog(
            global_cache=directory.cache,
            cache_handle=directory.cache.reference,
            inputs=config.inputs.reference,
            num_patches=config.inputs.num_patches,
        )


class CacheUnkTask(Task):
    _inputs = set()
    _optionals = set()

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        for idx, unk_config in config.inputs.unknown.iter_bins():
            print_message(f"processing bin {idx}", colored=True, bold=False)
            create_catalog(
                global_cache=directory.cache,
                cache_handle=directory.cache.unknown[idx],
                inputs=unk_config,
                num_patches=config.inputs.num_patches,
            )


def run_autocorr(
    cache_handle: CacheHandle,
    corrfunc_handle: CorrFuncHandle,
    config: Configuration,
) -> None:
    data, rand = cache_handle.load()
    (corr,) = yaw.autocorrelate(
        config,
        data,
        rand,
        # progress=...,
        # max_workers=...,
    )
    corr.to_file(corrfunc_handle.path)


class AutoRefTask(Task):
    _inputs = {CacheRefTask}
    _optionals = set()

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        run_autocorr(
            cache_handle=directory.cache.reference,
            corrfunc_handle=directory.paircounts.auto_ref,
            config=config.correlation,
        )


class AutoUnkTask(Task):
    _inputs = {CacheUnkTask}
    _optionals = set()

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        for idx, handle in directory.cache.unknown.items():
            print_message(f"processing bin {idx}", colored=True, bold=False)
            run_autocorr(
                cache_handle=handle,
                corrfunc_handle=directory.paircounts.auto_unk[idx],
                config=config.correlation,
            )


class CrossCorrTask(Task):
    _inputs = {CacheRefTask, CacheUnkTask}
    _optionals = set()

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        ref_data, ref_rand = directory.cache.reference.load()

        for idx, handle in directory.cache.unknown.items():
            print_message(f"processing bin {idx}", colored=True, bold=False)
            unk_data, unk_rand = handle.load()

            (corr,) = yaw.crosscorrelate(
                config.correlation,
                ref_data,
                unk_data,
                ref_rand=ref_rand,
                unk_rand=unk_rand,
                # progress=...,
                # max_workers=...,
            )
            corr.to_file(directory.paircounts.cross[idx].path)


class EstimateTask(Task):
    _inputs = {CrossCorrTask}
    _optionals = {AutoRefTask, AutoUnkTask}

    def run(self, directory: ProjectDirectory, config: ProjectConfig) -> None:
        if directory.paircounts.auto_ref.exists():
            auto_ref = directory.paircounts.auto_ref.load().sample()
            auto_ref.to_files(directory.estimate.auto_ref.template)
        else:
            auto_ref = None

        for idx, cross_handle in directory.paircounts.cross.items():
            print_message(f"processing bin {idx}", colored=True, bold=False)
            auto_handle = directory.paircounts.auto_unk[idx]
            if auto_handle.exists():
                auto_unk = auto_handle.load().sample()
                auto_unk.to_files(directory.estimate.auto_unk[idx].template)
            else:
                auto_unk = None

            cross = cross_handle.load().sample()
            ncc = yaw.RedshiftData.from_corrdata(cross, auto_ref, auto_unk)
            ncc.to_files(directory.estimate.nz_est[idx].template)


class HistTask(Task):
    _inputs = {CacheUnkTask}
    _optionals = set()

    def run(self) -> None:
        raise NotImplementedError


class PlotTask(Task):
    _inputs = set()
    _optionals = {AutoRefTask, AutoUnkTask, EstimateTask, HistTask}

    def run(self) -> None:
        raise NotImplementedError


def has_child(task: Task, candidates: set[Task]) -> bool:
    for candidate in candidates:
        if task in candidate.inputs | candidate.optionals:
            return True
    return False


def find_chain_ends(tasks: set[Task]) -> set[Task]:
    return {task for task in tasks if not has_child(task, tasks)}


def move_to_front(queue: deque, item: Any) -> None:
    queue.remove(item)
    queue.appendleft(item)


def build_chain(end: Task, chain: deque[Task] | None = None) -> deque[Task]:
    chain = chain or deque((end,))

    for parent in end.inputs | end.optionals:
        while parent in chain:
            chain.remove(parent)

        similar_tasks = tuple(task for task in chain if task.name == parent.name)
        for task in similar_tasks:
            move_to_front(chain, task)

        chain.appendleft(parent)
        chain = build_chain(parent, chain)

    return chain


def remove_duplicates(tasks: Iterable[Task]) -> deque[Task]:
    seen = set()
    uniques = deque()
    for task in tasks:
        if task not in seen:
            uniques.append(task)
            seen.add(task)
    return uniques


class TaskList(Container, Sized):
    _tasks: set[Task]
    _queue: deque[Task]
    _history: list[Task]

    def __init__(self, tasks: Iterable[Task]) -> None:
        self.clear()
        self._tasks = set(tasks)
        self._link_tasks()

    @classmethod
    def from_list(cls, task_names: Iterable[str]) -> TaskList:

        tasks = [Task.get(name) for name in task_names]
        return cls(task() for task in tasks)

    def to_list(self) -> list[str]:
        return [task.name for task in self._schedule(resume=False)]

    def __len__(self) -> int:
        return len(self._queue)

    def __contains__(self, task: Task) -> bool:
        return task in self._queue

    def __str__(self) -> str:
        tasks = Counter(task.name for task in self._queue)
        return " -> ".join(
            name if count == 1 else f"{name}[{count}]" for name, count in tasks.items()
        )

    def _link_tasks(self) -> None:
        for task in self._tasks:
            for candidate in self._tasks:
                if candidate == task:
                    continue
                task.connect_input(candidate)

            task.check_inputs()

    def _schedule(self, *, resume: bool = False) -> deque[Task]:
        chains = [build_chain(end) for end in find_chain_ends(self._tasks)]
        queue = remove_duplicates(itertools.chain(*chains))
        if resume:
            return deque(task for task in queue if not task.completed())
        return queue

    def clear(self) -> None:
        self._queue = deque()
        self._history = list()

    def schedule(self, resume: bool = False) -> None:
        self.clear()
        self._queue = self._schedule(resume=resume)

    def pop(self) -> Task:
        task = self._queue.popleft()
        self._history.append(task)
        return task
