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
from yaw.config.base import ConfigError

if TYPE_CHECKING:
    from typing import TypeVar

    TypeTask = TypeVar("TypeTask", bound="Task")


class Task(ABC):
    _tasks: dict[str, type[Task]] = {}
    name: str
    _inputs: set[type[Task]]
    _optionals: set[type[Task]]

    def __init_subclass__(cls):
        cls.name = cls.__name__.replace("Task", "").lower()
        cls._tasks[cls.name] = cls
        return super().__init_subclass__()

    def __init__(self, project: ProjectDirectory) -> None:
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
    _inputs = ()
    _optionals = ()

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
                overwrite=True,
                progress=...,
                max_workers=...,
            )


class CacheUnkTask(Task):
    _inputs = ()
    _optionals = ()

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
                    overwrite=True,
                    progress=...,
                    max_workers=...,
                )


class AutoRefTask(Task):
    _inputs = (CacheRefTask,)
    _optionals = ()

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
    _inputs = (CacheUnkTask,)
    _optionals = ()

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


class CrossTask(Task):
    _inputs = (CacheRefTask, CacheUnkTask)
    _optionals = ()

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
    _inputs = (CrossTask,)
    _optionals = (AutoRefTask, AutoUnkTask)

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


class TrueTask(Task):
    _inputs = (CacheUnkTask,)
    _optionals = ()

    def run(self) -> None:
        super().run()

        raise NotImplementedError


class PlotTask(Task):
    _inputs = (EstimateTask,)
    _optionals = (TrueTask,)

    def run(self) -> None:
        super().run()

        raise NotImplementedError
