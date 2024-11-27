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
from yaw.config.base import ConfigError
from yaw.utils import transform_matches

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, TypeVar

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
        self._completed = False

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
        return self._completed
        return all(handle.exists() for handle in self.outputs)

    @abstractmethod
    def run(self) -> None:
        print(f"running {self.name}")
        self._completed = True


class CacheRefTask(Task):
    _inputs = set()
    _optionals = set()

    def run(self) -> None:
        super().run()

        cache = self.project.cache.reference
        for path in (cache.data.path, cache.rand.path):
            yaw.Catalog.from_file(
                cache_directory=...,
                path=path,
                ra_name=...,
                dec_name=...,
                weight_name=...,
                redshift_name=...,
                patch_centers=...,
                patch_name=...,
                patch_num=...,
                overwrite=...,
                progress=...,
                max_workers=...,
            )


class CacheUnkTask(Task):
    _inputs = set()
    _optionals = set()

    def run(self) -> None:
        super().run()

        for idx in self.project.indices:
            cache = self.project.cache.unknown[idx]
            for path in (cache.data.path, cache.rand.path):
                yaw.Catalog.from_file(
                    cache_directory=...,
                    path=path,
                    ra_name=...,
                    dec_name=...,
                    weight_name=...,
                    redshift_name=...,
                    patch_centers=...,
                    patch_name=...,
                    patch_num=...,
                    overwrite=...,
                    progress=...,
                    max_workers=...,
                )


class AutoRefTask(Task):
    _inputs = {CacheRefTask}
    _optionals = set()

    def run(self) -> None:
        super().run()

        data, random = self.project.cache.reference.load()
        (corr,) = yaw.autocorrelate(
            self.project.correlation.config,
            data,
            random,
            progress=self.project.progress,
            max_workers=self.project.correlation.config.max_workers,
        )
        path = self.project.paircounts.auto_ref.path
        corr.to_file(path)


class AutoUnkTask(Task):
    _inputs = {CacheUnkTask}
    _optionals = set()

    def run(self) -> None:
        super().run()

        for idx, handle in self.project.cache.unknown.items():
            data, random = handle.load()
            (corr,) = yaw.autocorrelate(
                self.project.correlation.config,
                data,
                random,
                progress=self.project.progress,
                max_workers=self.project.correlation.config.max_workers,
            )
            path = self.project.paircounts.auto_unk[idx].path
            corr.to_file(path)


class CrossCorrTask(Task):
    _inputs = {CacheRefTask, CacheUnkTask}
    _optionals = set()

    def run(self) -> None:
        super().run()

        ref_data, ref_rand = self.project.cache.reference.load()
        for idx, handle in self.project.cache.unknown.items():
            unk_data, unk_rand = handle.load()
            (corr,) = yaw.crosscorrelate(
                self.project.correlation.config,
                ref_data,
                unk_data,
                ref_rand=ref_rand,
                unk_rand=unk_rand,
                progress=self.project.progress,
                max_workers=self.project.correlation.config.max_workers,
            )
            path = self.project.paircounts.cross[idx].path
            corr.to_file(path)


class EstimateTask(Task):
    _inputs = {CrossCorrTask}
    _optionals = {AutoRefTask, AutoUnkTask}

    def run(self) -> None:
        super().run()

        paircounts = self.project.paircounts
        estimate = self.project.estimate

        if paircounts.auto_ref.exists():
            auto_ref = paircounts.auto_ref.load().sample()
            path = estimate.auto_ref.template
            auto_ref.to_files(path)
        else:
            auto_ref = None

        for idx, handle in paircounts.cross.items():
            cross = handle.load()
            auto_pairs = paircounts.auto_unk[idx]
            if auto_pairs.exists():
                auto_unk = auto_pairs.load().sample()
                path = estimate.auto_unk[idx].template
                auto_unk.to_files(path)
            else:
                auto_unk = None

            ncc = yaw.RedshiftData.from_corrdata(cross.sample(), auto_ref, auto_unk)
            path = estimate[idx].template
            ncc.to_files(path)


class HistTask(Task):
    _inputs = {CacheUnkTask}
    _optionals = set()

    def run(self) -> None:
        super().run()

        raise NotImplementedError


class PlotTask(Task):
    _inputs = set()
    _optionals = {AutoRefTask, AutoUnkTask, EstimateTask, HistTask}

    def run(self) -> None:
        # NOTE: there may be nothing to do
        super().run()

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
        return [task.name for task in self._history]

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

    def clear(self) -> None:
        self._queue = deque()
        self._history = list()

    def schedule(self, resume: bool = False) -> None:
        self.clear()

        chains = [build_chain(end) for end in find_chain_ends(self._tasks)]
        queue = remove_duplicates(itertools.chain(*chains))
        if resume:
            queue = deque(task for task in queue if not task.completed())
        self._queue = queue

    def pop(self) -> Task:
        task = self._queue.popleft()
        self._history.append(task)
        return task


if __name__ == "__main__":
    task_list = [
        "hist",
        "cross_corr",
        "auto_ref",
        "estimate",
        "cache_ref",
        "cache_unk",
        "estimate",
        "estimate",
        "plot",
    ]
    tasks = TaskList.from_list(task_list)
    tasks.schedule()

    print(tasks)
    while tasks:
        tasks.pop()
        print(tasks)

    print(tasks.to_list())
