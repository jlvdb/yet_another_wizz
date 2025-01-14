"""
Implements the indivual tasks of the pipeline.

A task is a unit with possible required or optional inputs that performs a
specific processing tasks. Tasks can be executed once linked to its required
parents and providing a project configuration and directory.

The pipeline tasks are automatically linked by a scheduler according to this
directed graph (#=: required, |-: optional):

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
from yaw.config.base import ConfigError, TextIndenter, format_yaml_record_commented
from yaw.utils import parallel

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import TypeVar

    from yaw.cli.config import CatPairConfig, ProjectConfig
    from yaw.cli.directory import CacheDirectory, ProjectDirectory
    from yaw.cli.handles import CacheHandle, CorrFuncHandle, Handle
    from yaw.config import Configuration
    from yaw.config.base import BaseConfig

    TypeTask = TypeVar("TypeTask", bound="Task")
    T = TypeVar("T")

logger = logging.getLogger(__name__)


def create_catalog(
    global_cache: CacheDirectory,
    cache_handle: CacheHandle,
    inputs: CatPairConfig,
    num_patches: int | None,
    *,
    progress: bool = False,
    max_workers: int | None = None,
) -> None:
    """
    Creates a new pair of data and random catalogs.

    Args:
        global_cache:
            The pipeline's cache directory.
        cache_handle:
            The actual cache handle in the cache directory used to locate the
            catalog cache directories.
        inputs:
            Input configuration containing file paths and column names.
        num_patches:
            Whether to generate patch centers automatically (preferrentially)
            from the random catalog.

    Keyword Args:
        progress:
            Whether to show a progress bar on terminal.
        max_workers:
            Restrict to using at most this number of parallel workers.
    """
    columns = inputs.get_columns()
    paths = {
        cache_handle.rand.path: inputs.path_rand,
        cache_handle.data.path: inputs.path_data,
    }

    for cache_path, input_path in paths.items():
        if input_path is None:
            logger.info("skipping unconfigured random catalog")
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
    """
    Measure an autocorrelation function.

    Args:
        cache_handle:
            The cache handle which contains the data and random catalogs.
        corrfunc_handle:
            The handle used to locate the output file for the pair counts.
        config:
            `yet_another_wizz` configuration use for correlation measurements.

    Keyword Args:
        progress:
            Whether to show a progress bar on terminal.
    """
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


@parallel.broadcasted
def load_optional_data(handle: Handle[T]) -> T | None:
    """Load and return data from a handle and return ``None`` if this data does
    not exist."""
    if not handle.exists():
        return None
    return handle.load()


def bin_iter_progress(iterable: Iterable[T]) -> Iterator[T]:
    """Iterates an iterable and prints a progress indication line to terminal."""
    try:
        N = len(iterable)
    except TypeError:
        iterable = tuple(iterable)
        N = len(iterable)

    for i, item in enumerate(iterable, 1):
        if parallel.on_root():
            logger.client(f"processing bin {i} / {N}")
        yield item


REGISTERED_TASKS: dict[str, type[Task]] = {}
"""Mapping from task name to task class."""


def get_all_tasks() -> tuple[type[Task]]:
    """Get a tuple of all implemented task classes."""
    return tuple(REGISTERED_TASKS.values())


def get_autolink_tasks() -> tuple[type[Task]]:
    """Get a tuple of all implemented task classes that are automatically linked
    into the pipeline when required."""
    return tuple(filter(lambda task: task.properties.auto_link, get_all_tasks()))


def get_user_tasks() -> tuple[type[Task]]:
    """Get a tuple of all implemented task classes that can be required by the
    user in the pipeline configuration."""
    return tuple(filter(lambda task: not task.properties.auto_link, get_all_tasks()))


class TaskError(Exception):
    """Special exception that can be used to indicate invalid task
    configurations."""

    def __init__(self, task: Task, msg: str):
        super().__init__(f"in tasks: {task.properties.name}: {msg}")


@dataclass(frozen=True)
class TaskProperties:
    """Wrapper class for global properties of a task."""

    name: str
    """Name by which the task can be identified."""
    help: str
    """Descriptive help string of the task."""
    inputs: set[type[Task]] = field(default_factory=set, kw_only=True)
    """Set of required parents of this task."""
    optionals: set[type[Task]] = field(default_factory=set, kw_only=True)
    """Set of optional parents of this task."""
    auto_link: bool = field(default=False, kw_only=True)
    """Whether the scheduler will link this task automatically into the pipeline
    when a task requires it."""


class Task(ABC):
    """
    Base class for all pipeline tasks.

    Implements the functionality that ensures that the task is properly linked
    to its required parents and that specific pipeline configuration criteria
    are met. The latter allows to identify invalid pipeline configurations
    prior to executing any task.

    Tasks do not return their outputs directly but write them to a provided
    project directory.

    .. Note:
        Subclasses are automatically registered in :obj:`REGISTERED_TASKS`.
    """

    properties: TaskProperties
    """Task properties instance."""
    name: str
    """Task name."""

    def __init_subclass__(cls):
        REGISTERED_TASKS[cls.properties.name] = cls
        cls.name = cls.properties.name
        return super().__init_subclass__()

    def __init__(self) -> None:
        self.inputs: set[Task] = set()
        self.optionals: set[Task] = set()

    def __hash__(self):
        return hash(self.name)

    def connect_input(self, task: Task) -> bool:
        """
        To set the given task as parent for this task.

        Checks only, if the provided task has the correct type.

        Args:
            task:
                Task that may be linked as required or optional input.

        Returns:
            Whether the linking was successful.
        """
        if type(task) in self.properties.inputs:
            self.inputs.add(task)
            return True

        if type(task) in self.properties.optionals:
            self.optionals.add(task)
            return True

        return False

    def _require_config_item(self, config: BaseConfig, path: str) -> None:
        """
        Convenience function that requires having a specific setup configuration
        item set.

        The configuration item is specified as a string of YAML field names
        joined by a dot, e.g. ``section: {item: value}`` is addressed as
        ``section.item.value``.

        Args:
            config:
                Configuration class to perform the check on.
            path:
                Path that identifies (hierarchical) name of required item.

        Raises:
            TaskError:
                If the requested item is ``None``.
        """
        current_path = ""
        for attr in path.split("."):
            current_path = f"{current_path}.{attr}"
            config = getattr(config, attr)
            if config is None:
                current_path = current_path.strip(".")
                raise TaskError(self, f"requries '{current_path}'")

    def check_config_requirements(self, config: ProjectConfig) -> None:
        """
        Function that performs check for configuration requirements specific
        to this task.

        Should raise a :obj:`TaskError` if any requirement is not met.
        """
        pass

    def check_inputs(self) -> None:
        """Checks that the task has all its required parents linked and raises
        an exception otherwise."""
        expect = set(t.name for t in self.properties.inputs)
        have = set(t.name for t in self.properties.inputs)
        for name in expect - have:
            raise ConfigError(f"missing input '{name}' for task '{self.name}'")

    @abstractmethod
    @parallel.broadcasted
    def completed(self, directory: ProjectDirectory) -> bool:
        """Checks whether the file outputs of this task already exist in the
        given project directory and synchronises result between MPI workers if
        needed."""
        pass

    @abstractmethod
    def run(
        self,
        directory: ProjectDirectory,
        config: ProjectConfig,
        *,
        progress: bool = False,
    ) -> None:
        """
        Perform the actual computation of the task.

        Task results are automatically written to the project directory on
        successful completion.

        Args:
            directory:
                Project directory to use.
            config:
                Project configuration to use.

        Keyword Args:
            progress:
                Whether to show a progress bar on terminal if available for
                task.
        """
        pass

    @classmethod
    def to_yaml(cls, padding: int = 20) -> str:
        return f"{cls.name:{padding}s}# {cls.properties.help}"


class LoadRefTask(Task):
    """
    Loads the reference sample input catalogs and caches them.

    Task is automatically linked if required by any other task in the pipeline.
    """

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
    """
    Loads the tomographic unknown sample input catalogs and caches them.

    Task is automatically linked if required by any other task in the pipeline.
    """

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
    """Runs pair counting for the reference sample autocorrelation function."""

    properties = TaskProperties(
        name="auto_ref",
        help="Run the pair counting for the reference autocorrelation function.",
        inputs={LoadRefTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.auto_ref.exists()

    def check_config_requirements(self, config: ProjectConfig) -> None:
        self._require_config_item(config, "inputs.reference.path_rand")

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
    """Runs pair counting for the unknown sample autocorrelation function in
    each tomographic bin."""

    properties = TaskProperties(
        name="auto_unk",
        help="Run the pair counting for the unknown autocorrelation functions.",
        inputs={LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.auto_unk.exists()

    def check_config_requirements(self, config: ProjectConfig) -> None:
        self._require_config_item(config, "inputs.unknown.path_rand")
        self._require_config_item(config, "inputs.unknown.redshift")

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
    """Runs pair counting for crosscorrelation function in each tomographic
    bin."""

    properties = TaskProperties(
        name="cross_corr",
        help="Run the pair counting for the cross-correlation functions.",
        inputs={LoadRefTask, LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.paircounts.cross.exists()

    def check_config_requirements(self, config: ProjectConfig) -> None:
        paths = ("inputs.reference.path_rand", "inputs.unknown.path_rand")
        for path in paths:
            try:
                return self._require_config_item(config, path)
            except TaskError:
                pass
        else:
            raise TaskError(self, f"requries '{paths[0]}' and/or '{paths[1]}'")

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
    """
    Computes all correlation functions from pair counts as well as the final
    redshift estimate.

    Requries ``CrossCorrTask``, optional autocorrelation functions are used to
    mitigate galaxy bias.
    """

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

        for idx in bin_iter_progress(directory.indices):
            auto_unk = load_optional_data(directory.paircounts.auto_unk[idx])
            if auto_unk:
                auto_unk = auto_unk.sample()
                auto_unk.to_files(directory.estimate.auto_unk[idx].template)

            cross = load_optional_data(directory.paircounts.cross[idx])
            if cross:
                ncc = yaw.RedshiftData.from_corrdata(cross.sample(), auto_ref, auto_unk)
                ncc.to_files(directory.estimate.nz_est[idx].template)


class HistTask(Task):
    """
    Computes redshift histograms unknown catalogs.

    Requries providing redshifts for unknown catalogs.
    """

    properties = TaskProperties(
        name="hist",
        help="Compute redshift histograms from unknown data if a redshift column is configured.",
        inputs={LoadUnkTask},
    )

    @parallel.broadcasted
    def completed(self, directory) -> bool:
        return directory.true.unknown.exists()

    def check_config_requirements(self, config: ProjectConfig) -> None:
        self._require_config_item(config, "inputs.unknown.redshift")

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
    """Automatically generates checkplots for all available output data
    products."""

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
    """
    Container class for a list of class, scheduled in a valid order.

    Performs task linking and configuration checks on initialisation. Tasks
    must be scheduled with an appropriate method call before first usage.

    Args:
        tasks:
            An iterable of task instances to schedule.
    """

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
        """
        Serialise the ordered list of task names to a multi-line YAML string.

        Args:
            indentation:
                Indentation helper class that manages the indentation level and
                indentation characters.
            padding:
                The minimum number of spaces between the start of the task name
                and the following comment containing the task description/help.

        Returns:
            Formatted YAML string.
        """
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
        """Create a new instance from an iterable of task names."""
        tasks = [REGISTERED_TASKS[name] for name in task_names]
        return cls(task() for task in tasks)

    def to_list(self) -> list[str]:
        """Convert the ordered list of tasks to a list of task names."""
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
        """Runs the linking stage for all tasks and verifies that all tasks are
        linked to their required parents."""
        auto_links = set(task() for task in get_autolink_tasks())

        for task in self._tasks:
            for candidate in self._tasks | auto_links:
                if candidate == task:
                    continue
                task.connect_input(candidate)

            task.check_inputs()

    def _schedule(self, directory: ProjectDirectory | None = None) -> deque[Task]:
        """
        Return a task queue by analysing the graph of tasks.

        If a project directory is provided, completed tasks are removed from the
        queue.

        Args:
            directory:
                Optional project directory instance used to remove completed
                tasks from the queue.

        Returns:
            Correctly ordered tasks in a :obj:`deque`.
        """
        graph = {task: task.inputs | task.optionals for task in self._tasks}
        sorter = TopologicalSorter(graph)
        queue = deque(sorter.static_order())
        if directory is not None:  # check for completed tasks
            return deque(task for task in queue if not task.completed(directory))
        return queue

    def check_config_requirements(self, config: ProjectConfig) -> None:
        """Run the configuration checks for each task."""
        for task in self._tasks:
            task.check_config_requirements(config)

    def clear(self) -> None:
        """Clear any queued tasks and the internal history book-keeping."""
        self._queue = deque()
        self._history = list()

    def schedule(self, directory: ProjectDirectory, *, resume: bool = False) -> None:
        """
        Reschedule the input tasks in the correct order.

        Args:
            directory:
                The project directory to operate on.

        Keyword Args:
            resume:
                Whether to remove completed tasks from the queue.
        """
        self.clear()
        if resume:
            args = (directory,)
        else:
            args = ()
        self._queue = self._schedule(*args)

    def pop(self) -> Task:
        """Return the next task from the queue and move it internally to the
        task history."""
        task = self._queue.popleft()
        self._history.append(task)
        return task
