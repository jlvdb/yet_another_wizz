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
import logging
from abc import ABC, abstractmethod
from collections import Counter, deque
from collections.abc import Container, Sized
from typing import TYPE_CHECKING

import yaw
from yaw.cli.plotting import WrappedFigure
from yaw.config.base import ConfigError
from yaw.utils import transform_matches

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Any, TypeVar

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
        logger.info(f"processing bin {i} / {N}")
        yield item


class Task(ABC):
    name: str
    help: str
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

    def __hash__(self):
        return hash(self.__class__)

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

    @abstractmethod
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
        return f"{cls.name:{padding}s}# {cls.help}"


def create_catalog(
    global_cache: CacheDirectory,
    cache_handle: CacheHandle,
    inputs: CatPairConfig,
    num_patches: int | None,
    *,
    progress: bool = False,
    max_workers: int | None = None,
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
            overwrite=True,
            progress=progress,
            max_workers=max_workers,
        )
        try:
            global_cache.set_patch_centers(cat.get_centers())
        except RuntimeError:
            pass


class LoadRefTask(Task):
    help = "Load and cache the reference data and randoms."
    _inputs = set()
    _optionals = set()

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
    help = "Load and cache the unknown data and randoms."
    # has no inputs, but we want to enforce LoadRefTask to be run first if
    # possible to determine patch centers
    _inputs = set()
    _optionals = {LoadRefTask}

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


def run_autocorr(
    cache_handle: CacheHandle,
    corrfunc_handle: CorrFuncHandle,
    config: Configuration,
    *,
    progress: bool = False,
) -> None:
    data, rand = cache_handle.load()
    (corr,) = yaw.autocorrelate(
        config,
        data,
        rand,
        progress=progress,
        max_workers=config.max_workers,
    )
    corr.to_file(corrfunc_handle.path)


class AutoRefTask(Task):
    help = "Run the pair counting for the reference autocorrelation function."
    _inputs = {LoadRefTask}
    _optionals = set()

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
    help = "Run the pair counting for the unknown autocorrelation functions."
    _inputs = {LoadUnkTask}
    _optionals = set()

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
    help = "Run the pair counting for the cross-correlation functions."
    _inputs = {LoadRefTask, LoadUnkTask}
    _optionals = set()

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
    help = "Compute the clustering redshift estimate and use autocorrelations to mitigate galaxy bias."
    _inputs = {CrossCorrTask}
    _optionals = {AutoRefTask, AutoUnkTask}

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
        if directory.paircounts.auto_ref.exists():
            auto_ref = directory.paircounts.auto_ref.load().sample()
            auto_ref.to_files(directory.estimate.auto_ref.template)
        else:
            auto_ref = None

        for idx, cross_handle in bin_iter_progress(directory.paircounts.cross.items()):
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
    help = "Compute redshift histograms from unknown data if a redshift column is configured."
    _inputs = {LoadUnkTask}
    _optionals = set()

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
    help = "Plot all available autocorrelation functions and redshift estimates."
    _inputs = set()
    _optionals = {AutoRefTask, AutoUnkTask, EstimateTask, HistTask}

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
        plot_dir = directory.plot
        indices = config.get_bin_indices()
        num_bins = len(indices)

        auto_ref_exists = directory.estimate.auto_ref.exists()
        auto_unk_exists = directory.estimate.auto_unk.exists()
        nz_est_exists = directory.estimate.nz_est.exists()
        hist_exists = directory.true.unknown.exists()

        if not (auto_ref_exists | auto_unk_exists | nz_est_exists | hist_exists):
            logger.warning("no data to plot")

        if auto_ref_exists:
            ylabel = r"$w_{\rm ss}$"
            with WrappedFigure(plot_dir.auto_ref_path, 1, ylabel) as fig:
                auto_ref = directory.estimate.auto_ref
                auto_ref.load().plot(ax=fig.axes[0], indicate_zero=True)

        if auto_unk_exists:
            ylabel = r"$w_{\rm pp}$"
            with WrappedFigure(plot_dir.auto_unk_path, num_bins, ylabel) as fig:
                for ax, auto_unk in zip(fig.axes, directory.estimate.auto_unk.values()):
                    auto_unk.load().plot(ax=ax, indicate_zero=True)

        if nz_est_exists or hist_exists:
            ylabel = r"Density estimate"
            with WrappedFigure(plot_dir.redshifts_path, num_bins, ylabel) as fig:
                for ax, idx in zip(fig.axes, indices):
                    is_last_bin = idx == indices[-1]

                    if nz_est_exists:
                        nz_est = directory.estimate.nz_est[idx].load()

                    if hist_exists:
                        hist = directory.true.unknown[idx].load().normalised()
                        if nz_est_exists:
                            nz_est = nz_est.normalised(hist)

                    if hist_exists:
                        label = "Histogram" if is_last_bin else None
                        hist.plot(ax=ax, indicate_zero=True, label=label)
                    if nz_est_exists:
                        label = "CC $p(z)$" if is_last_bin else None
                        nz_est.plot(ax=ax, indicate_zero=True, label=label)

                    if is_last_bin:
                        ax.legend()


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
        for task in self._tasks:
            for candidate in self._tasks:
                if candidate == task:
                    continue
                task.connect_input(candidate)

            task.check_inputs()

    def _schedule(self, directory: ProjectDirectory | None = None) -> deque[Task]:
        chains = [build_chain(end) for end in find_chain_ends(self._tasks)]
        queue = remove_duplicates(itertools.chain(*chains))
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
