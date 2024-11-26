"""
Network of task connections (#=: required, |-: optional):

CacheRef === AutoRef ---+
    #                   |
    #==== Cross === Estimate === Plot
    #                   |         |
CacheUnk --- AutoUnk ---+         |
    #                             |
    #== True ---------------------+
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import yaw
from yaw.cli.directory import ProjectDirectory
from yaw.options import NotSet

if TYPE_CHECKING:
    from typing import TypeVar

    TypeTask = TypeVar("TypeTask", bound="Task")


class Task(ABC):
    _tasks: dict[str, type[Task]] = {}
    name: str
    inputs: tuple[Task, ...]
    optionals: tuple[Task, ...]

    def __init_subclass__(cls):
        cls.name = cls.__name__.replace("Task", "").lower()
        cls._tasks[cls.name] = cls
        return super().__init_subclass__()

    def __new__(cls: type[TypeTask]) -> TypeTask:
        new = super().__new__(cls)
        new._inputs = {task: NotSet for task in cls.inputs}
        new._optionals = {task: NotSet for task in cls.optionals}
        return new

    def __init__(self, project: ProjectDirectory) -> None:
        self.project = project

    @classmethod
    def get(cls: type[TypeTask], name: str) -> type[TypeTask]:
        try:
            return cls._tasks[name]
        except KeyError as err:
            raise ValueError(f"no tasked with name '{name}'") from err

    def connect_input(self, task: Task) -> bool:
        if type(task) in self.inputs:
            self._inputs[type(task)] = task
            return True

        if type(task) in self.optionals:
            self._optionals[type(task)] = task
            return True

        return False

    def check_inputs(self) -> None:
        for task, inst in self._inputs.items():
            if inst is NotSet:
                raise ValueError(f"missing input '{task.name}' for task '{self.name}'")

    def completed(self) -> bool:
        return all(handle.exists() for handle in self.outputs)

    def execute(self) -> None:
        if self.completed():
            return

        self.check_inputs()
        for parent in self._inputs.values():
            parent.execute()

        for parent in self._optionals.values():
            if parent is not NotSet:
                parent.execute()

        self._run()

    @abstractmethod
    def _run(self) -> None:
        print(f"running {self.name}")
        self._completed = True


class CacheRefTask(Task):
    inputs = ()
    optionals = ()

    def _run(self) -> None:
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
                overwrite=True,
                progress=...,
                max_workers=...,
            )


class CacheUnkTask(Task):
    inputs = ()
    optionals = ()

    def _run(self) -> None:
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
                    overwrite=True,
                    progress=...,
                    max_workers=...,
                )


class AutoRefTask(Task):
    inputs = (CacheRefTask,)
    optionals = ()

    def _run(self) -> None:
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
    inputs = (CacheUnkTask,)
    optionals = ()

    def _run(self) -> None:
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


class CrossTask(Task):
    inputs = (CacheRefTask, CacheUnkTask)
    optionals = ()

    def _run(self) -> None:
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
    inputs = (CrossTask,)
    optionals = (AutoRefTask, AutoUnkTask)

    def _run(self) -> None:
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


class TrueTask(Task):
    inputs = (CacheUnkTask,)
    optionals = ()

    def _run(self) -> None:
        raise NotImplementedError


class PlotTask(Task):
    inputs = (EstimateTask,)
    optionals = (TrueTask,)

    def _run(self) -> None:
        raise NotImplementedError
