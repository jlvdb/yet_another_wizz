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

import logging
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Container, Sized
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from typing import TYPE_CHECKING

import yaw
from yaw.cli import plotting
from yaw.cli.handles import load_optional_data
from yaw.config.base import ConfigError, TextIndenter, format_yaml_record_commented
from yaw.utils import parallel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TypeVar

    from yaw.cli.config import CatPairConfig, ProjectConfig
    from yaw.cli.directory import CacheDirectory, ProjectDirectory
    from yaw.cli.handles import CacheHandle, CorrFuncHandle
    from yaw.config import Configuration

    TypeTask = TypeVar("TypeTask", bound="Task")
    T = TypeVar("T")

logger = logging.getLogger("yaw.cli.tasks")


def bin_iter_progress(iterable: Iterable[T]) -> Iterator[T]:
    try:
        N = len(iterable)
    except TypeError:
        iterable = tuple(iterable)
        N = len(iterable)

    for i, item in enumerate(iterable, 1):
        if parallel.on_root():
            logger.info(f"processing bin {i} / {N}")
        yield item


REGISTERED_TASKS: dict[str, type[Task]] = {}


def get_all_tasks() -> tuple[type[Task]]:
    return tuple(REGISTERED_TASKS.values())


def get_autolink_tasks() -> tuple[type[Task]]:
    return tuple(filter(lambda task: task.properties.auto_link, get_all_tasks()))


def get_user_tasks() -> tuple[type[Task]]:
    return tuple(filter(lambda task: not task.properties.auto_link, get_all_tasks()))


@dataclass(frozen=True)
class TaskProperties:
    name: str
    help: str
    inputs: set[type[Task]] = field(default_factory=set, kw_only=True)
    optionals: set[type[Task]] = field(default_factory=set, kw_only=True)
    auto_link: bool = field(default=False, kw_only=True)


class Task(ABC):
    properties: TaskProperties
    name: str

    def __init_subclass__(cls):
        REGISTERED_TASKS[cls.properties.name] = cls
        cls.name = cls.properties.name
        return super().__init_subclass__()

    def __init__(self) -> None:
        self.inputs: set[Task] = set()
        self.optionals: set[Task] = set()

    def __hash__(self):
        return hash(self.__class__)

    def connect_input(self, task: Task) -> bool:
        if type(task) in self.properties.inputs:
            self.inputs.add(task)
            return True

        if type(task) in self.properties.optionals:
            self.optionals.add(task)
            return True

        return False

    def check_inputs(self) -> None:
        expect = set(t.name for t in self.properties.inputs)
        have = set(t.name for t in self.properties.inputs)
        for name in expect - have:
            raise ConfigError(f"missing input '{name}' for task '{self.name}'")

    @abstractmethod
    @parallel.broadcasted
    def completed(self, directory: ProjectDirectory) -> bool:
        pass

    @abstractmethod
    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        pass

    @classmethod
    def to_yaml(cls, padding: int = 20) -> str:
        return f"{cls.name:{padding}s}# {cls.properties.help}"


def create_catalog(
    global_cache: CacheDirectory,
    cache_handle: CacheHandle,
    inputs: CatPairConfig,
    num_patches: int | None,
    *,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    columns = inputs.get_columns()
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
            ra_name=columns["ra"],
            dec_name=columns["dec"],
            weight_name=columns["weight"],
            redshift_name=columns["redshift"],
            patch_centers=global_cache.get_patch_centers(),
            patch_name=columns["patches"],
            patch_num=num_patches,
            overwrite=True,
            progress=progress,
            max_workers=max_workers,
        )
        try:
            if parallel.on_root():
                global_cache.set_patch_centers(cat.get_centers())
        except RuntimeError:
            pass
        parallel.COMM.Barrier()


def run_autocorr(
    cache_handle: CacheHandle,
    corrfunc_handle: CorrFuncHandle,
    config: Configuration,
    *,
    progress: bool = False,
) -> None:
    data, rand = cache_handle.load()
    if rand is None:
        raise ValueError("could not load randoms")

    (corr,) = yaw.autocorrelate(
        config,
        data,
        rand,
        progress=progress,
        max_workers=config.max_workers,
    )
    corr.to_file(corrfunc_handle.path)


class LoadRefTask(Task):
    properties = TaskProperties(
        name="load_ref",
        help="Load and cache the reference data and randoms.",
        auto_link=True,
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.cache.reference.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        create_catalog(
            global_cache=directory.cache,
            cache_handle=directory.cache.reference,
            inputs=config.inputs.reference,
            num_patches=config.inputs.num_patches,
            progress=progress,
            max_workers=config.correlation.max_workers,
        )


class LoadUnkTask(Task):
    properties = TaskProperties(
        name="load_unk",
        help="Load and cache the unknown data and randoms.",
        optionals={LoadRefTask},
        auto_link=True,
    )
    # has no inputs, but we want to enforce LoadRefTask to be run first if
    # possible to determine patch centers

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.cache.unknown.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        for idx, unk_config in bin_iter_progress(config.inputs.unknown.iter_bins()):
            create_catalog(
                global_cache=directory.cache,
                cache_handle=directory.cache.unknown[idx],
                inputs=unk_config,
                num_patches=config.inputs.num_patches,
                progress=progress,
                max_workers=config.correlation.max_workers,
            )


class AutoRefTask(Task):
    properties = TaskProperties(
        name="auto_ref",
        help="Run the pair counting for the reference autocorrelation function.",
        inputs={LoadRefTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.auto_ref.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        run_autocorr(
            cache_handle=directory.cache.reference,
            corrfunc_handle=directory.paircounts.auto_ref,
            config=config.correlation,
            progress=progress,
        )


class AutoUnkTask(Task):
    properties = TaskProperties(
        name="auto_unk",
        help="Run the pair counting for the unknown autocorrelation functions.",
        inputs={LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.auto_unk.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        for idx, handle in bin_iter_progress(directory.cache.unknown.items()):
            run_autocorr(
                cache_handle=handle,
                corrfunc_handle=directory.paircounts.auto_unk[idx],
                config=config.correlation,
                progress=progress,
            )


class CrossCorrTask(Task):
    properties = TaskProperties(
        name="cross_corr",
        help="Run the pair counting for the cross-correlation functions.",
        inputs={LoadRefTask, LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.cross.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        ref_data, ref_rand = directory.cache.reference.load()

        for idx, handle in bin_iter_progress(directory.cache.unknown.items()):
            unk_data, unk_rand = handle.load()
            if ref_rand is None and unk_rand is None:
                raise ValueError("could not load any randoms")

            (corr,) = yaw.crosscorrelate(
                config.correlation,
                ref_data,
                unk_data,
                ref_rand=ref_rand,
                unk_rand=unk_rand,
                progress=progress,
                max_workers=config.correlation.max_workers,
            )
            corr.to_file(directory.paircounts.cross[idx].path)


class EstimateTask(Task):
    properties = TaskProperties(
        name="estimate",
        help="Compute the clustering redshift estimate and use autocorrelations to mitigate galaxy bias.",
        inputs={CrossCorrTask},
        optionals={AutoRefTask, AutoUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return (
            directory.estimate.auto_ref.exists()
            | directory.estimate.auto_unk.exists()
            | directory.estimate.nz_est.exists()
        )

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        auto_ref = load_optional_data(directory.paircounts.auto_ref)
        if auto_ref:
            auto_ref = auto_ref.sample()
            auto_ref.to_files(directory.estimate.auto_ref.template)

        for idx, cross_handle in bin_iter_progress(directory.paircounts.cross.items()):
            auto_unk = load_optional_data(directory.paircounts.auto_unk[idx])
            if auto_unk:
                auto_unk = auto_unk.sample()
                auto_unk.to_files(directory.estimate.auto_unk[idx].template)

            cross = cross_handle.load().sample()
            ncc = yaw.RedshiftData.from_corrdata(cross, auto_ref, auto_unk)
            ncc.to_files(directory.estimate.nz_est[idx].template)


class HistTask(Task):
    properties = TaskProperties(
        name="hist",
        help="Compute redshift histograms from unknown data if a redshift column is configured.",
        inputs={LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.true.unknown.exists()

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        for idx, handle in bin_iter_progress(directory.cache.unknown.items()):
            hist = yaw.HistData.from_catalog(
                handle.load_data(),
                config.correlation,
                progress=progress,
                max_workers=config.correlation.max_workers,
            )
            hist.to_files(directory.true.unknown[idx].template)


class PlotTask(Task):
    properties = TaskProperties(
        name="plot",
        help="Plot all available autocorrelation functions and redshift estimates.",
        optionals={AutoRefTask, AutoUnkTask, EstimateTask, HistTask},
    )

    def completed(self, directory) -> bool:
        if len(self.optionals) == 0:
            return True  # nothing to do

        for _ in directory.plot.path.iterdir():
            return True
        return False

    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        auto_ref = load_optional_data(directory.estimate.auto_ref)
        auto_unks = [
            load_optional_data(w) for w in directory.estimate.auto_unk.values()
        ]
        nz_ests = [load_optional_data(w) for w in directory.estimate.nz_est.values()]
        hists = [load_optional_data(w) for w in directory.true.unknown.values()]

        plot_created = plotting.plot_wss(directory.plot.auto_ref_path, auto_ref)
        plot_created |= plotting.plot_wpp(directory.plot.auto_unk_path, auto_unks)
        plot_created |= plotting.plot_nz(directory.plot.redshifts_path, nz_ests, hists)

        if not plot_created and parallel.on_root():
            logger.warning("no data to plot")


class TaskList(Container, Sized):
    _tasks: set[Task]
    _queue: deque[Task]
    _history: list[Task]

    def __init__(self, tasks: Iterable[Task]) -> None:
        self.clear()
        self._tasks = set(tasks)
        self._link_tasks()

    @classmethod
    def format_yaml_doc(
        self,
        *,
        indentation: TextIndenter | None = None,
        padding: int = 24,
    ) -> str:
        indentation = indentation or TextIndenter()
        list_indent = indentation.indent[:-2]  # manual fix for nicer list format

        lines = [
            format_yaml_record_commented(
                "tasks", comment="List of pipeline tasks to execute", padding=padding
            )
        ]
        lines.extend(
            list_indent + "- " + task.to_yaml(padding) for task in get_user_tasks()
        )

        return "\n".join(lines)

    @classmethod
    def from_list(cls, task_names: Iterable[str]) -> TaskList:
        tasks = [REGISTERED_TASKS[name] for name in task_names]
        return cls(task() for task in tasks)

    def to_list(self) -> list[str]:
        return [task.name for task in self._schedule()]

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
        auto_links = set(task() for task in get_autolink_tasks())

        for task in self._tasks:
            for candidate in self._tasks | auto_links:
                if candidate == task:
                    continue
                task.connect_input(candidate)

            task.check_inputs()

    def _schedule(self, directory: ProjectDirectory | None = None) -> deque[Task]:
        graph = {task: task.inputs | task.optionals for task in self._tasks}
        sorter = TopologicalSorter(graph)
        queue = deque(sorter.static_order())
        if directory is not None:  # check for completed tasks
            return deque(task for task in queue if not task.completed(directory))
        return queue

    def clear(self) -> None:
        self._queue = deque()
        self._history = list()

    def schedule(self, directory: ProjectDirectory, *, resume: bool = False) -> None:
        self.clear()
        if resume:
            args = (directory,)
        else:
            args = ()
        self._queue = self._schedule(*args)

    def pop(self) -> Task:
        task = self._queue.popleft()
        self._history.append(task)
        return task
